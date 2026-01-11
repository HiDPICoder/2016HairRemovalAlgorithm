#define INTERMEDIATE_IMAGES 1


#if MPI_SUPPORT

#include <mpi.h>

static int mpi_rank = 0;
static int mpi_nproc = 0;

#define MPI_IS_ROOT (mpi_rank == 0)

#else

#define MPI_IS_ROOT 1

#endif



#include "include/skelft.h"
#include "include/field.h"
#include "include/image.h"

#if GUI_SUPPORT
#include "include/vis.h"
#endif

#include <cuda_runtime_api.h>
#include <math.h>
#include <assert.h>
#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <time.h>

#include "color.h"
#include "connected.h"
#include "inpainting.h"
#include "api.h"
#include "pgm.h"
#include <filesystem>
#include <locale>

using namespace std;


#define INDEX(i,j) (i)+fboSize*(j)

// These are lists specifying coordinate offsets into all neigbouring pixels, in a specific order.
// Clockwise is clockwise, starting from the left bottom. Blockwise starts with all 4-connected
// neighbours and only then follows with the diagonal neighbours.
Coord __clockwise[8] = { Coord(-1,-1), Coord(0,-1), Coord(1,-1), Coord(1,0), Coord(1,1), Coord(0,1), Coord(-1,1), Coord(-1,0) };
Coord __blockwise[8] = { Coord(0,-1), Coord(1,0), Coord(0,1), Coord(-1,0), Coord(-1,-1), Coord(1,-1), Coord(1,1), Coord(-1,1) };

// Global options, see the structure in `include/api.h`
Options options;

// xm and ym are always zero, as an image always start in the top left. xM and yM represent the width and height
// of the input image, which may be less than the dimensions of the fboSize. The fboSize is the maximum of the first
// power of two of xM and yM, as required by the PBA algoritm.
int xm = 0, ym = 0, xM,yM,fboSize;
std::vector<CComponent> components;

std::vector<Coord> skeletonEndpoints;

// CPU buffers used for various images.
IMAGE<float>* rgbImage = NULL;
IMAGE<float>* hsvImage = NULL; // Used for converting RGB to grayscale image, and inpaining in HSV space
IMAGE<float>* inpaintedImage = NULL;
FIELD<float>* mask = NULL;
FIELD<float>* skeleton = NULL;

const char* filePrefix = "";

unsigned char* inputFBO; //Input image, 0-padded to fboSize^2
float siteParamLength;
float* siteParam; //Boundary parameterization (fboSize^2). value(i,j) = param of (i,j) if on boundary, else ALIVE or FAR_AWAY

// The following are CUDA buffers for several images.
unsigned char* imageDetectedGaps;
unsigned char* imageHairMask;
unsigned char* imageInvertedHairMask;
unsigned char* gapsSkeleton;
unsigned char* simplifiedSkeleton;

// These track maximum distances between junctions. Used to guess which formats outputs the most likely structure of hairs.
float globalMaxDistance;
float hairMaskMaxDistance;
float invertedHairMaskMaxDistance;

void allocateCudaMem(int size)
{
	skelft2DInitialization(size);

	cudaMallocHost((void**)&inputFBO,size*size*sizeof(unsigned char));
	cudaMallocHost((void**)&siteParam,size*size*sizeof(float));
	cudaMallocHost((void**)&imageDetectedGaps,size*size*sizeof(unsigned char));
	cudaMallocHost((void**)&imageHairMask,size*size*sizeof(unsigned char));
	cudaMallocHost((void**)&imageInvertedHairMask,size*size*sizeof(unsigned char));
	cudaMallocHost((void**)&gapsSkeleton,size*size*sizeof(unsigned char));
	cudaMallocHost((void**)&simplifiedSkeleton,size*size*sizeof(unsigned char));
}

void deallocateCudaMem()
{
	skelft2DDeinitialization();

	cudaFreeHost(inputFBO);
	cudaFreeHost(siteParam);
	cudaFreeHost(imageDetectedGaps);
	cudaFreeHost(imageHairMask);
	cudaFreeHost(imageInvertedHairMask);
	cudaFreeHost(gapsSkeleton);
	cudaFreeHost(simplifiedSkeleton);
}

// All these function prototypes are implemented in CUDA, see `gaps.cu` for implementation
void allocateBuffers(int xM, int yM, int size);
void deallocateBuffers();
void loadImage(unsigned char* image);
void invertImage();
void clearDetectedGaps();
bool computeThreshold(int level);
void computeMorphs(unsigned char *out, float radius);
void computeDetectedGaps(int level, float lambda);
void copyDetectedGapsD2H(unsigned char* out);
void copyImage(int which, void *output);
void computeInflation(unsigned char* result, unsigned char* reconstruction, unsigned char* skeleton);



// Used in Connected Component algorithm to determine if some pixel values are in the same component
struct is_same_component
{
	bool operator() (const unsigned char& x, const unsigned char& y) const
	{
		// I experimented with separating connected components even there was quite a large
		// difference in colors, but I have not found that this approach improves the results.
		// return fabs(x-y) < 50;
		return x>0 && y>0;
	}
};

struct is_background
{
	bool operator() (const unsigned char& x) const { return x == 0; }
};

struct is_not_background
{
	bool operator() (const unsigned char& x) const { return x != 0; }
};

void ensureOutputDir()
{
	// Use standard ternary operator instead of GCC extension '?:'
	const char* output = options.output ? options.output : "./output";

	try {
		// Automatically handles nested directories (like "mkdir -p")
		std::filesystem::create_directories(output);
	}
	catch (const std::exception& e) {
		if (MPI_IS_ROOT) {
			cerr << "Error creating output directory: " << e.what() << endl;
		}
	}
}

char* outputdir(const char* file)
{
	// Buffer overflow
	static char name[1024];

	sprintf(name, "%s/%s%s", options.output ? options.output : "./output", filePrefix, file);

	return name;
}

void printUsage(char **argv)
{
	if (MPI_IS_ROOT)
	{
		cout << "Hair detection" << endl << endl;
		cout << "Usage: " << endl;
		cout << argv[0] << " [options]" << endl;
		cout << endl;
		cout << "\tOption\tArg\tDefault\tDescription" << endl;
		cout << "\t------------------------------------------------------------------" << endl;
		cout << "\t-f\tpath\t\tInput PPM file, required." << endl;
		cout << "\t-w\t0..3\t3\t0 = original, 1 = inverted, 2 = combined, 3 = likeliest." << endl;
		cout << endl;
		cout << "\t-p\t[0,1]\t" << options.boundaryPercentage << "\tSkeleton prune as percentage from boundary." << endl;
		cout << "\t-b\t[1,..)\t" << options.minSkeletonLevel << "\tMinimum prune parameter." << endl;
		cout << "\t-B\t[1,..)\t" << options.maxSkeletonLevel << "\tMaximum prune parameter." << endl;
		cout << endl;
		cout << "\t-s\t[0,1]\t" << options.maxDistanceScaling << "\tPercentage of longest branch to require as minimal branch length." << endl;
		cout << "\t-d\t[1,..)\t" << options.minDistanceThreshold << "\tMinimum bound on branch length." << endl;
		cout << "\t-D\t[1,..)\t" << options.maxDistanceThreshold << "\tMaximum bound on branch length." << endl;
		cout << endl;
		cout << "\t-J\t[0,1]\t" << options.junctionRatioThreshold << "\tMaximum junction ratio." << endl;
		cout << "\t-S\t[1,..)\t" << options.skeletonLevel << "\tSkeletonization simplification level." << endl;
		cout << "\t-r\t[1,..)\t" << options.morphRadius << "\tRadius used for opening/closings." << endl;
		cout << "\t-l\t[0,1]\t" << options.lambda << "\tLambda, 0 = open-close, 1 = close-open, in-between for linear combination." << endl;
		cout << endl;
		cout << "\t-o\tpath\t\tOutput directory." << endl;
		cout << "\t-v\t    \t\tVerbose, output information." << endl;
#if GUI_SUPPORT
		cout << "\t-g\t    \t\tOpen GUI with results." << endl;
#endif
	}

	exit(2);
}

void parseOptions(int argc, char** argv)
{
	if (argc < 2) printUsage(argv);

	for (int i = 1; i < argc; ++i)
	{
		// Check if the argument is an option (starts with '-')
		if (argv[i][0] == '-')
		{
			char optionChar = argv[i][1];

			// 1. Handle flags that do NOT take arguments
			if (optionChar == 'v') {
				options.verbose = true;
				continue;
			}
			if (optionChar == 'g') {
				options.gui = true;
				continue;
			}

			// 2. Handle options that REQUIRE arguments
			// Ensure there is a next argument available
			if (i + 1 >= argc) {
				if (MPI_IS_ROOT) cerr << "Missing argument for option -" << optionChar << endl;
				printUsage(argv);
			}

			// Advance to the next argument to get the value
			char* optArg = argv[++i];

			switch (optionChar)
			{
			case 'f': options.file = optArg; break;
			case 'S': options.skeletonLevel = atof(optArg); break;
			case 'r': options.morphRadius = atof(optArg); break;
			case 'l': options.lambda = atof(optArg); break;
			case 'J': options.junctionRatioThreshold = atof(optArg); break;

			case 'p': options.boundaryPercentage = atof(optArg); break;
			case 'b': options.minSkeletonLevel = atof(optArg); break;
			case 'B': options.maxSkeletonLevel = atof(optArg); break;

			case 's': options.maxDistanceScaling = atof(optArg); break;
			case 'd': options.minDistanceThreshold = atof(optArg); break;
			case 'D': options.maxDistanceThreshold = atof(optArg); break;

			case 'o': options.output = optArg; break;

			case 'w':
				switch (atoi(optArg))
				{
				case 0: options.format = Options::ORIGINAL; break;
				case 1: options.format = Options::INVERTED; break;
				case 2: options.format = Options::BOTH; break;
				case 3: options.format = Options::LIKELIEST; break;
				default: printUsage(argv); break;
				}
				break;

			default:
				if (MPI_IS_ROOT) cerr << "Unknown option -" << optionChar << endl;
				printUsage(argv);
			}
		}
		else
		{
			// Handle positional arguments or excessive arguments
			if (MPI_IS_ROOT) cerr << "Excessive arguments are ignored: " << argv[i] << endl;
		}
	}

	// Post-parsing validation
	if (!options.file)
	{
		if (MPI_IS_ROOT) cerr << "No input file given." << endl;
		exit(3);
	}

#if MPI_SUPPORT
	if (mpi_nproc > 1 && options.gui)
	{
		if (MPI_IS_ROOT) cerr << "MPI support is only available when no GUI is used" << endl;
		exit(4);
	}
#endif

	ensureOutputDir();
}
void showInfo()
{
	if (!MPI_IS_ROOT || !options.verbose) return;

	cout << "====================================================" << endl;
	cout << "                   fbo size:  " << fboSize << endl;
	cout << endl;
	cout << "----------------------------------------------------" << endl;
	cout << "                GAP DETECTION PARAMS " << endl;
	cout << "----------------------------------------------------" << endl;
	cout << "  [-l]               lambda:  " << options.lambda << endl;
	cout << "  [-r]         morph radius:  " << options.morphRadius << endl;
	cout << "  [-S]   gap skeleton level:  " << options.skeletonLevel << endl;
	cout << endl;
	cout << "---------------------------------------------------" << endl;
	cout << "                   FILTERING PARAMS " << endl;
	cout << "---------------------------------------------------" << endl;
	cout << "  [-p]  boundary percentage:  " << options.boundaryPercentage << endl;
	cout << "  [-b]    min pruning level:  " << options.minSkeletonLevel << endl;
	cout << "  [-B]    max pruning level:  " << options.maxSkeletonLevel << endl;
	cout << endl;
	cout << "  [-s]         dist scaling:  " << options.maxDistanceScaling << endl;
	cout << "  [-d]    min required dist:  " << options.minDistanceThreshold << endl;
	cout << "  [-D]    max required dist:  " << options.maxDistanceThreshold << endl;
	cout << endl;
	cout << "  [-J]   max junction ratio:  " << options.junctionRatioThreshold << endl;
	cout << "===================================================" << endl;

	cout << endl;

	switch (options.format)
	{
		case Options::ORIGINAL: cout << "Hair detection using original image." << endl; break;
		case Options::INVERTED: cout << "Hair detection using inverted image." << endl; break;
		case Options::BOTH: cout << "Hair detection using original and inverted image." << endl; break;
		case Options::LIKELIEST: cout << "Hair detection using both images, then using likeliest." << endl; break;
	}
}

void convertRGB2HSV(IMAGE<float>* image)
{
	for (float* r = image->r.data(), *g = image->g.data(), *b = image->b.data(), *rend = image->r.data()+image->dimX()*image->dimY(); r < rend; ++r, ++g, ++b)
	{
		HSV hsv = RGB(*r,*g,*b).toHSV();
		*r = hsv.h, *g = hsv.s, *b = hsv.v;
	}
}

void convertHSV2RGB(IMAGE<float>* image)
{
	for (float* r = image->r.data(), *g = image->g.data(), *b = image->b.data(), *rend = image->r.data()+image->dimX()*image->dimY(); r < rend; ++r, ++g, ++b)
	{
		RGB rgb = HSV(*r,*g,*b).toRGB();
		*r = rgb.r, *g = rgb.g, *b = rgb.b;
	}
}

void readImage()
{
	rgbImage = IMAGE<float>::read(options.file);

	if (!rgbImage)
	{
		if (MPI_IS_ROOT) cerr << "Input file could not be read." << endl;
		exit(1);
	}

	xM = rgbImage->dimX();
	yM = rgbImage->dimY();
	fboSize = skelft2DSize(xM,yM);

	hsvImage = new IMAGE<float>(*rgbImage);
	convertRGB2HSV(hsvImage);

	inpaintedImage = new IMAGE<float>(xM, yM);
	mask = new FIELD<float>(xM, yM);
	skeleton = new FIELD<float>(xM, yM);

	allocateCudaMem(fboSize);

	memset(inputFBO,0,fboSize*fboSize*sizeof(unsigned char));
	for (int j=0; j<yM; ++j)
	for (int i=0; i<xM; ++i)
		inputFBO[INDEX(i,j)] = hsvImage->b(i,j) * 0xff; // Read value component of HSV color

	allocateBuffers(xM, yM, fboSize);
}

// This function processes a single level, i.e. threshold the image based on the level, calculate
// open-close and close-open (in CUDA), perform gap detection from Sobiecki et al. (Gap-sensitive
// segmentation and restoration of digital images. In: Proc. CGVC. pp. 136–144. Eurographics (2014))
void processLevel(int level)
{
	bool changed = computeThreshold(level);

	if (!changed) return;

	computeMorphs(imageDetectedGaps, options.morphRadius);

	// Parameterize the boundary required for the AFMM for skeleton calculation
	float length = skelft2DMakeBoundary(imageDetectedGaps, xm, ym, xM, yM, siteParam, fboSize, 0, false);

	// If the boundary is empty, don't bother processing the level
	if (!length) return;

	// Calculate Feature Transform as needed for skeletonization (buffers only necessary in GPU mem) and then
	// calculate skeleton itself.
	skelft2DFT(NULL, siteParam, xm, ym, xM, yM, fboSize);
	skelft2DSkeleton(NULL, length, options.skeletonLevel, xm, ym, xM, yM);

	// Actually perform the hair-detection algorithm, in CUDA
	computeDetectedGaps(level, options.lambda);
}




// Merge gap mask from MPI child process into the parent's mask
void mergeDetectedGaps(unsigned char* client)
{
	unsigned char *f = client, *t = imageDetectedGaps;

	for (int i = 0; i < fboSize * fboSize; ++i, ++f, ++t)
	{
		if (*f) *t = *f;
	}
}

// Compute the hair mask for all thresholds in the image
void computeGaps()
{
	clearDetectedGaps();

#if MPI_SUPPORT
	for (int level = mpi_rank; level <= 0xff; level += mpi_nproc)
#else
	for (int level = 0; level <= 0xff; ++level)
#endif
	{
		processLevel(level);
	}

	// Copy GPU memory to CPU
	copyDetectedGapsD2H(imageDetectedGaps);

#if MPI_SUPPORT
	if (MPI_IS_ROOT)
	{
		unsigned char* buffer = new unsigned char[fboSize*fboSize];

		// Receive data from all processors
		for (int i = 1; i < mpi_nproc; ++i)
		{
			MPI_Recv(buffer, fboSize * fboSize, MPI_UNSIGNED_CHAR, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			mergeDetectedGaps(buffer);
		}

		delete[] buffer;
	}
	else
	{
		MPI_Send(imageDetectedGaps, fboSize * fboSize, MPI_UNSIGNED_CHAR, 0, 1, MPI_COMM_WORLD);
	}
#endif

#if INTERMEDIATE_IMAGES

	// Set any non-zero mask pixel to 0xff, so that we output a blackwhite. The actual color information
	// is not currently used, but may be used to improve the ability to distinguish hairs from false,
	// positives, in which case this has to be removed!
	for (int i=0;i<fboSize*fboSize;++i) imageDetectedGaps[i] = imageDetectedGaps[i] ? 0xff : 0;

	writePGM(imageDetectedGaps, outputdir("gaps.pgm"), xM,yM,fboSize);
#endif
}

// Calculate dimensions of a connected components, by taking the
// width/height of the smallest square (orthogonal to image axes).
void calculateDimensions(std::vector<int>& pixels, int& w, int& h)
{
	int minX = -1, maxX = -1;
	int minY = -1, maxY = -1;

	// Count number of junctions
	for (std::vector<int>::iterator p = pixels.begin(); p != pixels.end(); ++p)
	{
		int i = *p % fboSize, j = *p / fboSize;

		if (minX < 0 || i < minX) minX = i;
		if (maxX < 0 || i > maxX) maxX = i;
		if (minY < 0 || j < minY) minY = j;
		if (maxY < 0 || j > maxY) maxY = j;
	}

	w = maxX - minX + 1; h = maxY - minY + 1;
}

void replacePixels(std::vector<int>& pixels, unsigned char* image, unsigned char v)
{
	for (std::vector<int>::iterator p = pixels.begin(); p != pixels.end(); ++p)
	{
		image[*p] = v;
	}
}

void replacePixels(std::vector<Coord>& pixels, unsigned char* image, unsigned char v)
{
	for (std::vector<Coord>::iterator p = pixels.begin(); p != pixels.end(); ++p)
	{
		image[INDEX(p->i,p->j)] = v;
	}
}

void copyPixels(std::vector<int>& pixels, unsigned char* from, unsigned char* to)
{
	for (std::vector<int>::iterator p = pixels.begin(); p != pixels.end(); ++p)
	{
		to[*p] = from[*p];
	}
}

// To avoid superfluous skeleton branches, fill component holes smaller than 1% of the image size
void fillSmallHoles()
{
	float* labeled = new float[fboSize * fboSize];

	// Calculate background connected components and check their size, small background areas are replaced by foreground
	ConnectedComponents().connected(imageDetectedGaps, labeled, components, xM, yM, fboSize, std::equal_to<unsigned char>(), is_not_background(), constant<bool,false>());

	// Fill small holes
	for (std::vector<CComponent>::iterator it = components.begin(); it != components.end(); ++it)
	{
		int w, h;
		calculateDimensions(it->pixels, w, h);

		if (w <= 0.01*xM && h <= 0.01*yM) replacePixels(it->pixels, imageDetectedGaps, 0xff);
	}

	delete[] labeled;
}

// Calculates connected components in the hair mask
void extractComponents()
{
	float* labeled = new float[fboSize * fboSize];

	ConnectedComponents().connected(imageDetectedGaps, labeled, components, xM, yM, fboSize, is_same_component(), is_background(), constant<bool,false>());

	// Remove small components to avoid processing noise, this significantly reduces execution time
	for (std::vector<CComponent>::iterator it = components.begin(); it != components.end();)
	{
		int w, h;
		calculateDimensions(it->pixels, w, h);

		if (w <= 0.01*xM && h <= 0.01*yM)
		{
			replacePixels(it->pixels, imageDetectedGaps, 0);

			it = components.erase(it);
		}
		else ++it;
	}

	delete[] labeled;
}

void parameterizeBoundary()
{
	siteParamLength = skelft2DMakeBoundary(imageDetectedGaps, xm, ym, xM, yM, siteParam, fboSize, 0, false);
}

// Calculate the boundary length per component, which is used to derive skeleton pruning parameter from
void calculateComponentBoundaryLengths()
{
	enum State { BLANK, BOUNDARY, VISITED };

	State *state = new State[fboSize*fboSize];
	memset(state, BLANK, fboSize*fboSize*sizeof(State));

	// Iterate over all connected components to determine boundary length per component
	for (std::vector<CComponent>::iterator it = components.begin(); it != components.end(); ++it)
	{
		std::vector<Coord> stack;

		// Push boundary pixels of this component onto the stack
		for (std::vector<int>::iterator p = it->pixels.begin(); p != it->pixels.end(); ++p)
		{
			int i = *p % fboSize, j = *p / fboSize;
			unsigned char val = imageDetectedGaps[*p];

			if (
				(i>=xm && imageDetectedGaps[INDEX(i-1,j)] != val) ||
				(i<xM && imageDetectedGaps[INDEX(i+1,j)] != val) ||
				(j>=ym && imageDetectedGaps[INDEX(i,j-1)] != val) ||
				(j<yM && imageDetectedGaps[INDEX(i,j+1)] != val)
			)
			{
				state[*p] = BOUNDARY;
				stack.push_back(Coord(i,j));
			}
		}

		it->boundaryLength = 0.0;

		while (!stack.empty())
		{
			Coord pt = stack[stack.size()-1]; stack.pop_back();

			if (state[INDEX(pt.i,pt.j)] == VISITED) continue;

			// Find preceding visited neighbour to calculate distance from
			int ci = pt.i, cj = pt.j;
			for (int dj=-1;dj<2;++dj)
			for (int di=-1;di<2;++di)
			{
				int ii=pt.i+di, jj=pt.j+dj;
				if (ii>=xm && ii<xM && jj>=ym && jj<yM && state[INDEX(ii,jj)] == VISITED)
					ci = ii, cj = jj;
			}

			it->boundaryLength += sqrt(float((pt.i-ci)*(pt.i-ci)+(pt.j-cj)*(pt.j-cj)));
			state[INDEX(pt.i,pt.j)] = VISITED;

			// Push neighbourhood onto stack
			for (int dj=-1;dj<2;++dj)
			for (int di=-1;di<2;++di)
			{
				if (di==0 && dj==0) continue;

				int ii=pt.i+di, jj=pt.j+dj;
				if (ii>=xm && ii<xM && jj>=ym && jj<yM && state[INDEX(ii,jj)] == BOUNDARY)
					stack.push_back(Coord(ii,jj));
			}
		}
	}

	delete[] state;
}

// Find skeleton pixels which have at least three branches, i.e. the junctions in the skeleton
void junctionMarker(unsigned char* skeleton, int i, int j, std::vector<Coord>& junctions)
{
	if (!skeleton[INDEX(i,j)]) return;

	bool north = j>0 ? !!skeleton[INDEX(i,j-1)] : 0;
	bool east = i<xM-1 ? !!skeleton[INDEX(i+1,j)] : 0;
	bool south = j<yM-1 ? !!skeleton[INDEX(i,j+1)] : 0;
	bool west = i>0 ? !!skeleton[INDEX(i-1,j)] : 0;

	int neighbours = north + east + south + west;

	if (
		(neighbours > 2) || // If more than 2 direct neighbours, definitely a junction
		(
			(neighbours == 2 && i>0 && j>0 && i<xM-1 && j<yM-1) && // If precisely two neighbours, check opposite diagonal
			(
				(north && west && skeleton[INDEX(i+1,j+1)]) ||
				(north && east && skeleton[INDEX(i-1,j+1)]) ||
				(south && west && skeleton[INDEX(i+1,j-1)]) ||
				(south && east && skeleton[INDEX(i-1,j-1)])
			)
		)
	)
	{
		junctions.push_back(Coord(i,j));
	}
}

// Find junctions only in specific set of pixels
std::vector<Coord> findJunctions(unsigned char *skeleton, std::vector<int>& pixels)
{
	std::vector<Coord> junctions;

	for (std::vector<int>::iterator p = pixels.begin(); p != pixels.end(); ++p)
	{
		int i = *p % fboSize, j = *p / fboSize;

		junctionMarker(skeleton, i,j, junctions);
	}

	return junctions;
}

// Find junctions in full skeleton image
std::vector<Coord> findJunctions(unsigned char* skeleton)
{
	std::vector<Coord> junctions;

	for (int j=0; j<yM; ++j)
	for (int i=0; i<xM; ++i)
	{
		junctionMarker(skeleton, i,j, junctions);
	}

	return junctions;
}

// Finds skeleton endpoints
void endpointMarker(unsigned char* skeleton, int i, int j, std::vector<Coord>& endpoints)
{
	if (!skeleton[INDEX(i,j)]) return;

	bool north = j>0 ? !!skeleton[INDEX(i,j-1)] : 0;
	bool east = i<xM-1 ? !!skeleton[INDEX(i+1,j)] : 0;
	bool south = j<yM-1 ? !!skeleton[INDEX(i,j+1)] : 0;
	bool west = i>0 ? !!skeleton[INDEX(i-1,j)] : 0;

	bool northeast = i>0 ? !!skeleton[INDEX(i+1,j-1)] : 0;
	bool southeast = i<xM-1 ? !!skeleton[INDEX(i+1,j+1)] : 0;
	bool southwest = j>0 ? !!skeleton[INDEX(i-1,j+1)] : 0;
	bool northwest = j<yM-1 ? !!skeleton[INDEX(i-1,j-1)] : 0;

	int neighbours = north + east + south + west + northeast + southeast + southwest + northwest;

	if (
		(neighbours <= 1) || // If only a single neighbour
		(
			(neighbours == 2) && // If precisely two neighbours, check to see if they are adjacent
			(
				(north && northeast) ||
				(northeast && east) ||
				(east && southeast) ||
				(southeast && south) ||
				(south && southwest) ||
				(southwest && west) ||
				(west && northwest) ||
				(northwest && north)
			)
		)
	)
	{
		endpoints.push_back(Coord(i,j));
	}
}

// Only endpoints in specific set of pixels
std::vector<Coord> findEndpoints(unsigned char *skeleton, std::vector<int>& pixels)
{
	std::vector<Coord> endpoints;

	for (std::vector<int>::iterator p = pixels.begin(); p != pixels.end(); ++p)
	{
		int i = *p % fboSize, j = *p / fboSize;

		endpointMarker(skeleton, i,j, endpoints);
	}

	return endpoints;
}

// Skeleton endpoints throughout the whole image
std::vector<Coord> findEndpoints(unsigned char* skeleton)
{
	std::vector<Coord> endpoints;

	for (int j=0; j<yM; ++j)
	for (int i=0; i<xM; ++i)
	{
		endpointMarker(skeleton, i,j, endpoints);
	}

	return endpoints;
}

// Remove single-pixel branches from a skeleton, they mess up the max distance determination
void removeSkeletonArtifacts(std::vector<int>& pixels, unsigned char* skeleton)
{
	for (std::vector<int>::iterator p = pixels.begin(); p != pixels.end(); ++p)
	{
		int i = *p % fboSize, j = *p / fboSize;

		if (i<1 || j<1 || i>=xM-1 || j>=yM-1 || !skeleton[INDEX(i,j)]) continue;

		int clockwise = 0, counterclockwise = 0;
		bool gap = false;

		// Count clockwise non-skel pixels until a skel pixel is found
		for (int c=0; c<8; ++c)
		{
			if (!skeleton[INDEX(i+__clockwise[c].i,j+__clockwise[c].j)]) clockwise++;
			else break;
		}

		// Count counterclockwise until a skel pixel is found
		for (int c=7; c>=0; --c)
		{
			if (!skeleton[INDEX(i+__clockwise[c].i,j+__clockwise[c].j)]) counterclockwise++;
			else break;
		}

		// Verify that in-between every pixel is occupied by the skeleton
		for (int c=clockwise; c<8-counterclockwise; ++c)
		{
			if (!skeleton[INDEX(i+__clockwise[c].i,j+__clockwise[c].j)]) gap = true;
		}

		// If a streak of 5 pixels and no other gaps, we have a match
		if (!gap && clockwise+counterclockwise == 5) skeleton[INDEX(i,j)] = 0;
	}
}

// Sort helper to sort components based on their boundary length.
struct CComponentLengthComparator
{
	bool operator() (const CComponent& a, const CComponent& b)
	{
		return a.boundaryLength < b.boundaryLength;
	}
};

void computeSimplifiedSkeleton()
{
	memset(simplifiedSkeleton, 0, fboSize*fboSize*sizeof(unsigned char));

	float prevSkeletonLevel = 0.0;

	// Sort components by boundary length so that we avoid computing a new skeleton
	// per component, but rather reuse the previous one if the threshold stays the same.
	std::sort(components.begin(), components.end(), CComponentLengthComparator());

	// Precompute FT necessary for skeletonization
	skelft2DFT(NULL, siteParam, xm, ym, xM, yM, fboSize);

	for (std::vector<CComponent>::iterator it = components.begin(); it != components.end(); ++it)
	{
		float skeletonLevel = fmin(fmax(options.minSkeletonLevel, floor(it->boundaryLength * options.boundaryPercentage)), options.maxSkeletonLevel);

		// Only recompute the skeleton if it has changed
		if (skeletonLevel != prevSkeletonLevel)
		{
			skelft2DSkeleton(gapsSkeleton, siteParamLength, skeletonLevel, xm, ym, xM, yM);

			prevSkeletonLevel = skeletonLevel;
		}

		// Copy only skeleton pixels from the current connected component
		copyPixels(it->pixels, gapsSkeleton, simplifiedSkeleton);
	}

#if INTERMEDIATE_IMAGES
	writePGM(simplifiedSkeleton, outputdir("skel.pgm"), xM,yM,fboSize);
#endif
}

// Trace a branch we know is in (i,j) to its junctions, and mark the state buffer
// accordingly to know which skeleton pixels have been processed.
SBranch traceBranch(unsigned char* skeleton, SBranch::State* state, int i, int j)
{
	SBranch branch;

next:
	state[INDEX(i,j)] = SBranch::VISITED;
	branch.pixels.push_back(Coord(i,j));

	for (int c=0; c<8; ++c)
	{
		int di = __blockwise[c].i, dj = __blockwise[c].j;
		int ii = i+di, jj=j+dj;

		if (ii<xm || ii>=xM || jj<ym || jj>=yM) continue;

		if (
			skeleton[INDEX(ii,jj)] && // Should be in skeleton
			state[INDEX(ii,jj)] == SBranch::BLANK && // Not visited yet
			(
				// Only for diagonal neighbours, check if neither of the direct neighbours is a junction
				// If this is the case we should not go into this branch, as it belongs to the branch
				// started from that junction
				c<4 ||
				(state[INDEX(ii,j)] != SBranch::JUNCTION && state[INDEX(i,jj)] != SBranch::JUNCTION)
			)
		)
		{
			i = ii, j = jj;
			goto next;
		}
	}

	return branch;
}

// Extract all branches in a skeleton (limited to a certain component as specified by `pixels`) from the skeleton's junctions.
std::vector<SBranch> extractBranches(unsigned char* skeleton, std::vector<int>& pixels, std::vector<Coord>& junctions)
{
	std::vector<SBranch> branches;
	SBranch::State* state = new SBranch::State[fboSize*fboSize];
	memset(state, SBranch::BLANK, fboSize*fboSize*sizeof(SBranch::State));

	// Mark every junction accordingly
	for (std::vector<Coord>::iterator it = junctions.begin(); it != junctions.end(); ++it)
	{
		state[INDEX(it->i,it->j)] = SBranch::JUNCTION;
	}

	// For each junction, trace a branch for each of its neighbour skeleton pixels
	for (std::vector<Coord>::iterator it = junctions.begin(); it != junctions.end(); ++it)
	{
		for (int c=0; c<8; ++c)
		{
			int di = __blockwise[c].i, dj = __blockwise[c].j;
			int ii = it->i+di, jj=it->j+dj;

			if (ii<xm || ii>=xM || jj<ym || jj>=yM) continue;

			if (skeleton[INDEX(ii,jj)] && state[INDEX(ii,jj)] == SBranch::BLANK)
			{
				branches.push_back(traceBranch(skeleton, state, ii,jj));
			}
		}
	}

	// Now that we have traced all branches from the junctions, also trace them from
	// endpoints, if they have not been visited yet. This may occur when the skeleton
	// is split in multiple parts where some parts do not contain any junction at all.
	std::vector<Coord> endpoints = findEndpoints(skeleton, pixels);

	for (std::vector<Coord>::iterator it = endpoints.begin(); it != endpoints.end(); ++it)
	{
		if (state[INDEX(it->i,it->j)] == SBranch::VISITED) continue;

		branches.push_back(traceBranch(skeleton, state, it->i,it->j));
	}

	delete[] state;

	return branches;
}

// Calculated the maximum branch distance, as required for skeleton-based filtering.
unsigned long calculateMaxDistance(std::vector<SBranch>& branches)
{
	unsigned long max = 0;

	for (std::vector<SBranch>::iterator it = branches.begin(); it != branches.end(); ++it)
	{
		if (it->pixels.size() > max) max = it->pixels.size();
	}

	if (max > globalMaxDistance) globalMaxDistance = max;

	return max;
}

// Count total number of pixels in the skeleton, also used in skeleton-based filtering.
unsigned long countSkeletonPixels(unsigned char* skeleton, std::vector<int>& pixels)
{
	unsigned long count = 0;

	for (std::vector<int>::iterator p = pixels.begin(); p != pixels.end(); ++p)
	{
		if (skeleton[*p]) count++;
	}

	return count;
}

// Analyze a single component, i.e. detect its junctions and all of branches
void analyzeComponent(CComponent& component)
{
	component.junctions = findJunctions(simplifiedSkeleton, component.pixels);
	component.branches = extractBranches(simplifiedSkeleton, component.pixels, component.junctions);

	component.maxDistance = calculateMaxDistance(component.branches);
	component.skelPixels = countSkeletonPixels(simplifiedSkeleton, component.pixels);
}

void detectSkeletonEndpoints()
{
	// Detect endpoints in simplified skeleton before filtering. We use these later when extending
	// the simplified skeleton with a less simplified version, in order to glue branch endings back on.
	skeletonEndpoints = findEndpoints(simplifiedSkeleton);
}

// Analyzes all connected components, i.e. the skeleton metrics we need during filtering.
void analyzeSkeleton()
{
	globalMaxDistance = 0.0f;

	for (std::vector<CComponent>::iterator it = components.begin(); it != components.end(); ++it)
	{
		removeSkeletonArtifacts(it->pixels, simplifiedSkeleton);

		analyzeComponent(*it);
	}
}

// Determines if a components has to be rejected, based on its maximum distance between junctions.
bool filterComponent(CComponent& component)
{
	float requiredDistance = fmin(fmax(options.minDistanceThreshold, globalMaxDistance * options.maxDistanceScaling), options.maxDistanceThreshold);

	if (
		// If the maximum distance between junctions is not enough
		component.maxDistance < requiredDistance ||

		// If the skeleton contains too many junctions, i.e. if the average distance between junctions is too small
		float(component.junctions.size())/component.skelPixels > options.junctionRatioThreshold
	)
	{
		replacePixels(component.pixels, simplifiedSkeleton, 0);

		return true;
	}

	return false;
}

// Determines for each connected component if it has to be filtered out
void filterSkeleton()
{
	if (!options.filteringEnabled) return;

	for (std::vector<CComponent>::iterator it = components.begin(); it != components.end(); ++it)
	{
		filterComponent(*it);
	}
}

// When calculating skeletons for filtering purposes, we may use quite extensive pruning to simplify the skeleton.
// This shortens branches near the endpoints (up to half the pruning parameter) which may be up to 10-20 pixels.
// When we were to inflate this pruned skeleton, all endpoints will not reach into the end of hairs, so that the
// endpoints of hairs are not removed. Recompute a much less pruned skeleton and copy its branch endings over to
// the simplified skeleton, so that the endpoints are inflated up until the end of all hairs.
void extendSimplifiedSkeleton()
{
	unsigned char* fullSkeleton = new unsigned char[fboSize*fboSize];
	memset(fullSkeleton, 0, fboSize*fboSize*sizeof(unsigned char));

	// Now that we have a skeleton which has been simplified specifically per
	// connected component, also compute a complete skeleton so that the endings
	// of the overly simplified skeleton may be reconstructed from this skeleton.
	skelft2DFT(NULL, siteParam, xm, ym, xM, yM, fboSize);
	skelft2DSkeleton(fullSkeleton, siteParamLength, 3.0, xm, ym, xM, yM);

	// From every endpoint, copy its 8-connected pixels from the complete skeleton
	// over to the simplified skeleton
	for (std::vector<Coord>::iterator it = skeletonEndpoints.begin(); it != skeletonEndpoints.end(); ++it)
	{
		// Verify that the endpoint has not been filtered out
		if (!simplifiedSkeleton[INDEX(it->i,it->j)]) continue;

		std::vector<Coord> stack;
		stack.push_back(Coord(it->i,it->j));

		while (!stack.empty())
		{
			Coord pt = stack[stack.size()-1]; stack.pop_back();

			for (int ii=-1; ii<2; ++ii)
			for (int jj=-1; jj<2; ++jj)
			{
				if (ii==0 && jj==0) continue;

				int i = pt.i+ii, j = pt.j+jj, idx = INDEX(i,j);

				if (i<xm || i>=xM || j<ym || j>=yM) continue;

				// If in complete skeleton but not in the filtered one, copy it over to the simplified skeleton
				if (fullSkeleton[idx] && !simplifiedSkeleton[idx] && imageDetectedGaps[idx])
				{
					simplifiedSkeleton[idx] = 0xff;
					stack.push_back(Coord(i,j));
				}
			}
		}
	}

	delete[] fullSkeleton;
}

// From the simplified, filtered skeleton, compute the actual inpainting hair mask by inflating the skeleton
// based on the original Feature Transform.
void reconstructFromSimplifiedSkeleton(unsigned char* result)
{
	// CUDA operation.
	computeInflation(result, imageDetectedGaps, simplifiedSkeleton);

#if INTERMEDIATE_IMAGES
	writePGM(result, outputdir("mask.pgm"), xM,yM,fboSize);
#endif
}

// First phase of the pipeline includes gap-detection on the full
// threshold-set and extracting connected components from the resulting mask.
void executePipelinePhase1()
{
	computeGaps();

	if (MPI_IS_ROOT)
	{
		fillSmallHoles();
		extractComponents();
		parameterizeBoundary();
		calculateComponentBoundaryLengths();
	}
}

// Second phase is about dynamic skeleton calculation, analyzayion and filtering
void executePipelinePhase2(unsigned char* result)
{
	if (MPI_IS_ROOT)
	{
		computeSimplifiedSkeleton();
		detectSkeletonEndpoints();
		analyzeSkeleton();
		filterSkeleton();
		extendSimplifiedSkeleton();
		reconstructFromSimplifiedSkeleton(result);
	}
}

void executePipeline(unsigned char* result)
{
	executePipelinePhase1();
	executePipelinePhase2(result);
}

// Merge the resulting hair mask from both the original image and the inverted image into
// a single mask
void mergeHairMasks()
{
	for (int i=0; i<fboSize*fboSize; ++i)
	{
		imageHairMask[i] = imageHairMask[i] || imageInvertedHairMask[i] ? 0xff : 0;
	}
}

// Guess which mask most likely contains hairs.
void selectLikeliestHairMask()
{
	if (invertedHairMaskMaxDistance > hairMaskMaxDistance)
	{
		memcpy(imageHairMask, imageInvertedHairMask, fboSize*fboSize*sizeof(unsigned char));
	}
}

// Copy basic C arrays into IMAGE<> fields
void prepareBuffers()
{
	for (int j=0; j<yM; ++j)
	for (int i=0; i<xM; ++i)
	{
		(*mask)(i,j) = imageHairMask[INDEX(i,j)] ? 1.0f : 0.0f;
		(*skeleton)(i,j) = simplifiedSkeleton[INDEX(i,j)] ? 1.0f : 0.0f;
	}
}

// Executes the complete pipeline
void computeHairMask()
{
	loadImage(inputFBO);

	switch (options.format)
	{
		case Options::ORIGINAL:
			executePipeline(imageHairMask);
			break;
		case Options::INVERTED:
			invertImage();
			executePipeline(imageHairMask);
			break;
		case Options::BOTH:
			filePrefix = "light_";
			executePipeline(imageHairMask);
			invertImage();
			filePrefix = "dark_";
			executePipeline(imageInvertedHairMask);
			if (MPI_IS_ROOT) mergeHairMasks();
			break;
		case Options::LIKELIEST:
			filePrefix = "light_";
			executePipeline(imageHairMask);
			hairMaskMaxDistance = globalMaxDistance;
			invertImage();
			filePrefix = "dark_";
			executePipeline(imageInvertedHairMask);
			invertedHairMaskMaxDistance = globalMaxDistance;
			if (MPI_IS_ROOT) selectLikeliestHairMask();
			break;
	}

	prepareBuffers();
}

// This is a helper function that only executes the second part of the pipeline,
// i.e. filtering of the skeleton. This is only available when a single mask is
// computed, because the state at this point is only valid for the latest calculated
// format. Used from the GUI when only filtering related parameters have changed,
// in which case doing a full theshold-set decomposition is not necessary.
void computeHairMaskOnlyPhase2()
{
	assert(options.format == Options::ORIGINAL || options.format == Options::INVERTED);

	executePipelinePhase2(imageHairMask);
	prepareBuffers();
}

// Inpaint the original image in HSV space based on the hair mask we calculated.
void inpaintImage()
{
	*inpaintedImage = *hsvImage;

	inpaint(inpaintedImage, mask);

	convertHSV2RGB(inpaintedImage);
}

// Save results to disk
void writeResults()
{
	filePrefix = "";

	mask->writePGM(outputdir("hairmask.pgm"));
	inpaintedImage->writePPM(outputdir("inpainted.ppm"));

#if INTERMEDIATE_IMAGES
	rgbImage->writePPM(outputdir("original.ppm"));
#endif
}

void handleResults(int argc, char** argv)
{
	writeResults();

#if GUI_SUPPORT
	if (options.gui)
	{
		Display(rgbImage, inpaintedImage, mask, skeleton, &options, argc, argv).show();
	}
#endif
}


int verifyCudaAvailability()
{
	int count;
	switch (cudaGetDeviceCount(&count))
	{
		case cudaErrorNoDevice:
			cerr << "No CUDA capable device found!" << endl;
			exit(1);
		case cudaErrorInsufficientDriver:
			cerr << "CUDA drivers not available!" << endl;
			exit(1);
		default: break;
	}

	return count;
}

void cleanup()
{
	deallocateBuffers();
	deallocateCudaMem();

	delete rgbImage;
	delete hsvImage;
	delete inpaintedImage;
	delete mask;

#if MPI_SUPPORT
	MPI_Finalize();
#endif
}

int main(int argc,char **argv)
{
	std::locale::global(std::locale("C"));
	int cudaDevices = verifyCudaAvailability();

	atexit(cleanup);

#if MPI_SUPPORT
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_nproc);

	cudaSetDevice(mpi_rank % cudaDevices);
#endif

	parseOptions(argc, argv);
	readImage();
	showInfo();
	computeHairMask();

	if (MPI_IS_ROOT)
	{
		inpaintImage();
		handleResults(argc, argv);
	}

	return 0;
}
