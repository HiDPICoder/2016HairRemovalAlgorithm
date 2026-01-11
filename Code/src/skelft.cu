#include <device_functions.h>
#include "include/skelft.h"
#include <stdio.h>



// Parameters for CUDA kernel executions; more or less optimized for a 1024x1024 image.
#define BLOCKX		16
#define BLOCKY		16
#define BLOCKSIZE	64
#define TILE_DIM	32
#define BLOCK_ROWS	16



/****** Global Variables *******/
short2* pbaTextureWork;
short2* pbaTextureFT;
short* pbaTextureDT;
short2* pbaTextureWorkTopo;
short2* pbaTextureSkeletonFT;

unsigned char* pbaTextureThreshDT;
unsigned char* pbaTextureThreshSkel;
unsigned char* pbaTextureTopo;

float*			pbaTexSiteParam;		// Stores boundary parameterization
int				pbaTexSize;				// Texture size (squared) actually used in all computations
int				floodBand  = 4,			// Various FT computation parameters; defaults are good for an 1024x1024 image.
				maurerBand = 4,
				colorBand  = 4;

cudaTextureObject_t pbaTexColor = 0;
cudaTextureObject_t pbaTexColor2 = 0;
cudaTextureObject_t pbaTexLinks = 0;
cudaTextureObject_t pbaTexParam = 0;
cudaTextureObject_t pbaTexGray = 0;
cudaTextureObject_t pbaTexOpenClosed = 0;
cudaTextureObject_t pbaTexOriginal = 0;

#if __CUDA_ARCH__ < 110					// We cannot use atomic intrinsics on SM10 or below. Thus, we define these as nop.
// #define atomicInc(a,b) 0				// The default will be that some code e.g. endpoint detection will thus not do anything.
#endif



/********* Kernels ********/
#include "include/skelftKernel.h"

template <typename T>
cudaTextureObject_t bindTextureObject(T* devPtr, size_t sizeInBytes) {
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = devPtr;
	resDesc.res.linear.desc = cudaCreateChannelDesc<T>();
	resDesc.res.linear.sizeInBytes = sizeInBytes;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;

	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
	return texObj;
}

// Initialize necessary memory (CPU/GPU sides)
// - textureSize: The max size of any image we will process until re-initialization
void skelft2DInitialization(int maxTexSize)
{
	cudaMalloc((void**)&pbaTextureWork, maxTexSize * maxTexSize * sizeof(short2));
	cudaMalloc((void**)&pbaTextureFT, maxTexSize * maxTexSize * sizeof(short2));
	cudaMalloc((void**)&pbaTextureDT, maxTexSize * maxTexSize * sizeof(short));
	cudaMalloc((void**)&pbaTextureWorkTopo, maxTexSize * maxTexSize * sizeof(short2));
	cudaMalloc((void**)&pbaTextureSkeletonFT, maxTexSize * maxTexSize * sizeof(short2));
	cudaMalloc((void**)&pbaTextureThreshDT, maxTexSize * maxTexSize * sizeof(unsigned char));
	cudaMalloc((void**)&pbaTextureThreshSkel, maxTexSize * maxTexSize * sizeof(unsigned char));
	cudaMalloc((void**)&pbaTextureTopo, maxTexSize * maxTexSize * sizeof(unsigned char));
	cudaMalloc((void**)&pbaTexSiteParam, maxTexSize * maxTexSize * sizeof(float));
}




// Deallocate all allocated memory
void skelft2DDeinitialization()
{
	cudaFree(pbaTextureWork); cudaFree(pbaTextureFT); cudaFree(pbaTextureDT);
	cudaFree(pbaTextureWorkTopo); cudaFree(pbaTextureSkeletonFT);
	cudaFree(pbaTextureThreshDT); cudaFree(pbaTextureThreshSkel); cudaFree(pbaTextureTopo);
	cudaFree(pbaTexSiteParam);
}



__global__ void kernelSiteParamInit(short2* inputVoro, int size)							//Initialize the Voronoi textures from the sites' encoding texture (parameterization)
{																							//REMARK: we interpret 'inputVoro' as a 2D texture, as it's much easier/faster like this
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	if (tx < size && ty < size) {
		int i = TOID(tx, ty, size);
		float param = tex1Dfetch<float>(pbaTexParam, i);
		short2& v = inputVoro[i];
		v.x = v.y = MARKER;
		if (param) { v.x = tx; v.y = ty; }
	}
}



__global__ void kernelSiteParamInitChar(short2* inputVoro, int size)							//Initialize the Voronoi textures from the sites' encoding texture (parameterization)
{																							//REMARK: we interpret 'inputVoro' as a 2D texture, as it's much easier/faster like this
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	if (tx < size && ty < size) {
		int i = TOID(tx, ty, size);
		unsigned char param = tex1Dfetch<unsigned char>(pbaTexGray, i);
		short2& v = inputVoro[i];
		v.x = v.y = MARKER;
		if (param) { v.x = tx; v.y = ty; }
	}
}



__global__ void kernelSiteParamInitCharInverse(short2* inputVoro, int size)							//Initialize the Voronoi textures from the sites' encoding texture (parameterization)
{																							//REMARK: we interpret 'inputVoro' as a 2D texture, as it's much easier/faster like this
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	if (tx < size && ty < size) {
		int i = TOID(tx, ty, size);
		unsigned char param = tex1Dfetch<unsigned char>(pbaTexGray, i);
		short2& v = inputVoro[i];
		v.x = v.y = MARKER;
		if (!param) { v.x = tx; v.y = ty; }
	}
}



// Update skelft2DInitializeInput
void skelft2DInitializeInput(float* sites, int size) {
	pbaTexSize = size;
	size_t bytes = pbaTexSize * pbaTexSize * sizeof(float);
	cudaMemcpy(pbaTexSiteParam, sites, bytes, cudaMemcpyHostToDevice);
	cudaTextureObject_t texObj = bindTextureObject(pbaTexSiteParam, bytes);
	dim3 block(BLOCKX, BLOCKY), grid(pbaTexSize / block.x, pbaTexSize / block.y);
	kernelSiteParamInit << <grid, block >> > (pbaTextureWork, pbaTexSize, texObj);
	cudaDestroyTextureObject(texObj);
}

// Update pba2DTranspose
void pba2DTranspose(short2* texture) {
	dim3 block(TILE_DIM, BLOCK_ROWS), grid(pbaTexSize / TILE_DIM, pbaTexSize / TILE_DIM);
	cudaTextureObject_t texObj = bindTextureObject(texture, pbaTexSize * pbaTexSize * sizeof(short2));
	kernelTranspose << <grid, block >> > (texture, pbaTexSize, texObj);
	cudaDestroyTextureObject(texObj);
}

void pba2DPhase1(int m1, short xm, short ym, short xM, short yM) {
	dim3 block(BLOCKSIZE);
	dim3 grid(pbaTexSize / block.x, m1);
	size_t sz = pbaTexSize * pbaTexSize * sizeof(short2);

	// Bind current textures to temporary objects
	cudaTextureObject_t texWork = bindTextureObject(pbaTextureWork, sz);
	cudaTextureObject_t texFT = bindTextureObject(pbaTextureFT, sz);

	// PASS THE OBJECTS TO THE KERNELS
	kernelFloodDown << < grid, block >> > (pbaTextureFT, pbaTexSize, pbaTexSize / m1, texWork);
	kernelFloodUp << < grid, block >> > (pbaTextureFT, pbaTexSize, pbaTexSize / m1, texFT);

	// Phase where we update vertical links needs both
	kernelUpdateVertical << < grid, block >> > (pbaTextureFT, pbaTexSize, m1, pbaTexSize / m1, texFT, texWork);

	// Clean up objects
	cudaDestroyTextureObject(texWork);
	cudaDestroyTextureObject(texFT);
}

void pba2DPhase2(int m2) {
	dim3 block(BLOCKSIZE), grid(pbaTexSize / block.x, m2);
	size_t bytes = pbaTexSize * pbaTexSize * sizeof(short2);

	cudaTextureObject_t texColor = bindTextureObject(pbaTextureFT, bytes);
	kernelProximatePoints << <grid, block >> > (pbaTextureWork, pbaTexSize, pbaTexSize / m2, texColor);
	

	cudaTextureObject_t texLinks = bindTextureObject(pbaTextureWork, bytes);
	kernelCreateForwardPointers << <grid, block >> > (pbaTextureWork, pbaTexSize, pbaTexSize / m2, texLinks);

	for (int noBand = m2; noBand > 1; noBand /= 2) {
		grid = dim3(pbaTexSize / block.x, noBand / 2);
		// Passing both texColor and texLinks
		kernelMergeBands << < grid, block >> > (pbaTextureWork, pbaTexSize, pbaTexSize / noBand, texColor, texLinks);
	}

	grid = dim3(pbaTexSize / block.x, pbaTexSize);
	kernelDoubleToSingleList << < grid, block >> > (pbaTextureWork, pbaTexSize, texColor, texLinks);
	cudaDestroyTextureObject(texColor);
	cudaDestroyTextureObject(texLinks);
}

void pba2DPhase3(int m3) {
	dim3 block(BLOCKSIZE / m3, m3), grid(pbaTexSize / block.x);
	cudaTextureObject_t texColor = bindTextureObject(pbaTextureWork, pbaTexSize * pbaTexSize * sizeof(short2));
	kernelColor << <grid, block >> > (pbaTextureFT, pbaTexSize, texColor);
	cudaDestroyTextureObject(texColor);
}



void skel2DFTCompute(short xm, short ym, short xM, short yM, int floodBand, int maurerBand, int colorBand)
{
    pba2DPhase1(floodBand,xm,ym,xM,yM);										//Vertical sweep

    pba2DTranspose(pbaTextureFT);											//

    pba2DPhase2(maurerBand);												//Horizontal coloring

    pba2DPhase3(colorBand);													//Row coloring

    pba2DTranspose(pbaTextureFT);
}





__global__ void kernelThresholdDT(unsigned char* output, int size, float threshold2, short xm, short ym, short xM, short yM)
//Input:    pbaTexColor: closest-site-ids per pixel, i.e. FT
//Output:   output: thresholded DT
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	if (tx > xm && ty > ym && tx < xM && ty < yM) {
		int id = TOID(tx, ty, size);
		short2 voroid = tex1Dfetch<short2>(pbaTexColor, id);
		float d2 = (float)(tx - voroid.x) * (tx - voroid.x) + (float)(ty - voroid.y) * (ty - voroid.y);
		output[id] = (d2 <= threshold2) ? 0xff : 0;
	}
}


__global__ void kernelThresholdDTInverse(unsigned char* output, int size, float threshold2, short xm, short ym, short xM, short yM)
//Input:    pbaTexColor: closest-site-ids per pixel, i.e. FT
//Output:   output: thresholded DT
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	if (tx > xm && ty > ym && tx < xM && ty < yM) {
		int id = TOID(tx, ty, size);
		short2 voroid = tex1Dfetch<short2>(pbaTexColor, id);
		float d2 = (float)(tx - voroid.x) * (tx - voroid.x) + (float)(ty - voroid.y) * (ty - voroid.y);
		output[id] = (short)sqrtf(d2);
	}
}



__global__ void kernelDT(short* output, int size, float threshold2, short xm, short ym, short xM, short yM)
//Input:    pbaTexColor: closest-site-ids per pixel, i.e. FT
//Output:   output: DT
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if (tx>xm && ty>ym && tx<xM && ty<yM)									//careful not to index outside the image..
	{
  	  int    id     = TOID(tx, ty, size);
	  short2 voroid = tex1Dfetch<short2>(pbaTexColor,id);							//get the closest-site to tx,ty into voroid.x,.y
	  float  d2     = (tx-voroid.x)*(tx-voroid.x)+(ty-voroid.y)*(ty-voroid.y);
	  output[id]    = sqrtf(d2);											//save the Euclidean DT
    }
}


__global__ void kernelSkel(unsigned char* output, short xm, short ym,
						   short xM, short yM, short size, float threshold, float length)
																			//Input:    pbaTexColor: closest-site-ids per pixel
																			//			pbaTexParam: labels for sites (only valid at site locations)
{																			//Output:	output: binary thresholded skeleton
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	if (tx > xm && ty > ym && tx < xM && ty < yM) {
		int id = TOID(tx, ty, size);
		short2 voroid = tex1Dfetch<short2>(pbaTexColor, id);
		float imp = tex1Dfetch<float>(pbaTexParam, TOID(voroid.x, voroid.y, size));

		short2 voroid_r = tex1Dfetch<short2>(pbaTexColor, id + 1);
		float imp_r = tex1Dfetch<float>(pbaTexParam, TOID(voroid_r.x, voroid_r.y, size));

		short2 voroid_u = tex1Dfetch<short2>(pbaTexColor, id + size);
		float imp_u = tex1Dfetch<float>(pbaTexParam, TOID(voroid_u.x, voroid_u.y, size));

		float Imp = fmaxf(fabsf(imp_r - imp), fabsf(imp_u - imp));
		Imp = fminf(Imp, fabsf(length - Imp));
		if (Imp >= threshold) output[id] = 0xff;
	}

	//WARNING: this kernel may sometimes create 2-pixel-thick branches.. Study the AFMM original code to see if this is correct.
}



#define X 0xff

__constant__ const															//REMARK: put following constants (for kernelTopology) in CUDA constant-memory, as this gives a huge speed difference
unsigned char topo_patterns[][9] =		{ {0,0,0,							//These are the 3x3 templates that we use to detect skeleton endpoints
										   0,X,0,							//(with four 90-degree rotations for each)
										   0,X,0},
										  {0,0,0,
										   0,X,0,
										   0,0,X},
										  {0,0,0,
										   0,X,0,
										   0,X,X},
										  {0,0,0,
										   0,X,0,
										   X,X,0}
										};

#define topo_NPATTERNS  4														//Number of patterns we try to match (for kernelTopology)
																				//REMARK: #define faster than __constant__

__constant__ const unsigned char topo_rot[][9] = { {0,1,2,3,4,5,6,7,8}, {2,5,8,1,4,7,0,3,6}, {8,7,6,5,4,3,2,1,0}, {6,3,0,7,4,1,8,5,2} };
																				//These encode the four 90-degree rotations of the patterns (for kernelTopology);

__device__ unsigned int topo_gc			= 0;
// __device__ unsigned int topo_gc_last	= 0;


__global__ void kernelTopology(unsigned char* output, short2* output_set, short xm, short ym, short xM, short yM, short size, int maxpts)
{
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned char t[9];

	if (tx>xm && ty>ym && tx<xM-1 && ty<yM-1)									//careful not to index outside the image; take into account the template size too
	{
	   int    id = TOID(tx, ty, size);
	   unsigned char  p  = tex1Dfetch<unsigned char>(pbaTexGray,id);							//get the skeleton pixel at tx,ty
	   if (p)																	//if the pixel isn't skeleton, nothing to do
	   {
	     unsigned char idx=0;
		 for(int j=ty-1;j<=ty+1;++j)											//read the template into t[] for easier use
		 {
		   int id = TOID(tx-1, j, size);
	       for(int i=0;i<=2;++i,++id,++idx)
		      t[idx] = tex1Dfetch<unsigned char>(pbaTexGray,id);								//get the 3x3 template centered at the skel point tx,ty
		 }

		 for(unsigned char r=0;r<4;++r)											//try to match all rotations of a pattern:
		 {
		   const unsigned char* rr = topo_rot[r];
	       for(unsigned char p=0;p<topo_NPATTERNS;++p)							//try to match all patterns:
	       {
	         const unsigned char* pat = topo_patterns[p];
			 unsigned char j = (p==0)? 0 : 7;									//Speedup: for all patterns except 1st, check only last 3 entries, the first 6 are identical for all patterns
			 for(;j<9;++j)														//try to match rotated pattern vs actual pattern
			   if (pat[j]!=t[rr[j]]) break;										//this rotation failed
			 if (j<6) break;													//Speedup: if we have a mismatch on the 1st 6 pattern entries, then none of the patterns can match
																				//		   since all templates have the same first 6 entries.

			 if (j==9)															//this rotation succeeded: mark the pixel as a topology event and we're done
			 {
				int crt_gc = atomicInc(&topo_gc,maxpts);						//REMARK: this serializes (compacts) all detected endpoints in one array.
				output_set[crt_gc] = make_short2(tx,ty);						//To do this, we use an atomic read-increment-return on a global counter,
																				//which is guaranteed to give all threads unique consecutive indexes in the array.
			    output[id] = 0xff;													//Also create the topology image
				return;
			 }
		   }
		 }
	   }
	}
	// else																		//Last thread: add zero-marker to the output point-set, so the reader knows how many points are really in there
	// if (tx==xM-1 && ty==yM-1)													//Also reset the global vector counter topo_gc, for the next parallel-run of this function
	// { topo_gc_last = topo_gc; topo_gc = 0; }									//We do this in the last thread so that no one modifies topo_gc from now on.
	// 																			//REMARK: this seems to be the only way I can read a __device__ variable back to the CPU
}




void skelft2DParams(int floodBand_, int maurerBand_, int colorBand_)		//Set up some params of the FT algorithm
{
  floodBand   = floodBand_;
  maurerBand  = maurerBand_;
  colorBand   = colorBand_;
}





// Compute 2D FT / Voronoi diagram of a set of sites
// siteParam:   Site parameterization. 0 = non-site points; >0 = site parameter value.
// output:		FT. The (x,y) at (i,j) are the coords of the closest site to (i,j)
// size:        Texture size (pow 2)
void skelft2DFT(short* output, float* siteParam, short xm, short ym, short xM, short yM, int size)
{
    skelft2DInitializeInput(siteParam,size);								    // Initialization of already-allocated data structures

    skel2DFTCompute(xm, ym, xM, yM, floodBand, maurerBand, colorBand);			// Compute FT

    // Copy FT to CPU, if required
    if (output) cudaMemcpy(output, pbaTextureFT, size*size*sizeof(short2), cudaMemcpyDeviceToHost);
}



__global__ void kernelGatherSkelPixels(short2* output, int size, short xm, short ym, short xM, short yM)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	if (tx > xm && ty > ym && tx < xM && ty < yM) {
		int id = TOID(tx, ty, size);
		unsigned char s = tex1Dfetch<unsigned char>(pbaTexGray, id);
		if (s) {
			int crt_gc = atomicInc(&topo_gc, size * size);
			output[crt_gc] = make_short2(tx, ty);
		}
	}					//REMARK: this seems to be the only way I can read a __device__ variable back to the CPU
}

__global__ void kernelSplat(unsigned char* output, int size, int numpts, short xm, short ym, short xM, short yM)
{
	int offs = blockIdx.x * blockDim.x + threadIdx.x;
	if (offs < numpts) {
		short2 skel = tex1Dfetch<short2>(pbaTexColor, offs);
		int tx = skel.x, ty = skel.y;
		int id = TOID(tx, ty, size);
		short2 voroid = tex1Dfetch<short2>(pbaTexLinks, id);
		float d2 = (float)(tx - voroid.x) * (tx - voroid.x) + (float)(ty - voroid.y) * (ty - voroid.y);
		short d = sqrtf(d2);
		short jmin = max(ty - d, ym), jmax = min(ty + d, yM);
		for (short j = jmin; j <= jmax; ++j) {
			short w = sqrtf(max(0.0f, d2 - (j - ty) * (j - ty)));
			short imin = max(tx - w, xm), imax = min(tx + w, xM);
			int start_id = TOID(imin, j, size);
			for (short i = imin; i <= imax; ++i) output[start_id++] = 0xff;
		}
	}
}


int skelft2DInflate(unsigned char* result, short xm, short ym, short xM, short yM) {
	dim3 block(BLOCKX, BLOCKY), grid(pbaTexSize / block.x, pbaTexSize / block.y);
	unsigned int num_pts = 0;
	cudaMemcpyToSymbol(topo_gc, &num_pts, sizeof(unsigned int));

	cudaTextureObject_t texGray = bindTextureObject(pbaTextureThreshSkel, pbaTexSize * pbaTexSize * sizeof(unsigned char));
	kernelGatherSkelPixels << <grid, block >> > (pbaTextureWorkTopo, pbaTexSize, xm, ym, xM, yM, texGray);
	cudaDestroyTextureObject(texGray);

	cudaMemcpyFromSymbol(&num_pts, topo_gc, sizeof(unsigned int));
	cudaMemset(pbaTextureThreshDT, 0, pbaTexSize * pbaTexSize * sizeof(unsigned char));

	block = dim3(BLOCKX * BLOCKY);
	grid = dim3((num_pts + block.x - 1) / block.x);
	cudaTextureObject_t texSkelPts = bindTextureObject(pbaTextureWorkTopo, pbaTexSize * pbaTexSize * sizeof(short2));
	cudaTextureObject_t texFT = bindTextureObject(pbaTextureFT, pbaTexSize * pbaTexSize * sizeof(short2));
	kernelSplat << <grid, block >> > (pbaTextureThreshDT, pbaTexSize, num_pts, xm, ym, xM, yM, texSkelPts, texFT);
	cudaDestroyTextureObject(texSkelPts);
	cudaDestroyTextureObject(texFT);

	if (result) cudaMemcpy(result, pbaTextureThreshDT, pbaTexSize * pbaTexSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	return num_pts;
}







__global__ void kernelGatherSkelPixelsMasked(short2* output, int size, short xm, short ym, short xM, short yM)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int maxpts = size*size;

	if (tx>xm && ty>ym && tx<xM && ty<yM)									//careful not to index outside the image..
	{
  	  int    id       = TOID(tx, ty, size);
	  unsigned char s = tex1Dfetch<unsigned char>(pbaTexGray,id);							//skeleton at tx,ty
	  unsigned char o = tex1Dfetch<unsigned char>(pbaTexOriginal,id);						//original at rx,ty
	  unsigned char oc = tex1Dfetch<unsigned char>(pbaTexOpenClosed,id);					//open-closed at rx,ty
	  if ((s && !oc) && o)													// only inflate points inside masked (of open-closed) skeleton and outside image (o>0 is background)
	  {
		 int crt_gc = atomicInc(&topo_gc,maxpts);							//REMARK: this serializes (compacts) all detected skelpoints in one array.
		 output[crt_gc] = make_short2(tx,ty);								//To do this, we use an atomic read-increment-return on a global counter,
																			//which is guaranteed to give all threads unique consecutive indexes in the array.
	  }
	}
	// else																	//Last thread: add zero-marker to the output point-set, so the reader knows how many points are really in there
	// if (tx==xM && ty==yM)													//Also reset the global vector counter topo_gc, for the next parallel-run of this function
	// { topo_gc_last = topo_gc; topo_gc = 0; }								//We do this in the last thread so that no one modifies topo_gc from now on.
																			//REMARK: this seems to be the only way I can read a __device__ variable back to the CPU
}




void skelft2DDT(short* outputDT, float threshold, short xm, short ym, short xM, short yM) {
	dim3 block(BLOCKX, BLOCKY), grid(pbaTexSize / block.x, pbaTexSize / block.y);
	cudaTextureObject_t texColor = bindTextureObject(pbaTextureFT, pbaTexSize * pbaTexSize * sizeof(short2));

	if (threshold >= 0) {
		kernelThresholdDT << <grid, block >> > (pbaTextureThreshDT, pbaTexSize, threshold * threshold, xm - 1, ym - 1, xM + 1, yM + 1, texColor);
		if (outputDT) cudaMemcpy(outputDT, pbaTextureThreshDT, pbaTexSize * pbaTexSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	}
	else {
		kernelDT << <grid, block >> > (pbaTextureDT, pbaTexSize, threshold * threshold, -1, -1, pbaTexSize, pbaTexSize, texColor);
		if (outputDT) cudaMemcpy(outputDT, pbaTextureDT, pbaTexSize * pbaTexSize * sizeof(short), cudaMemcpyDeviceToHost);
	}
	cudaDestroyTextureObject(texColor);
}



void skelft2DSkeleton(unsigned char* outputSkel, float length, float threshold, short xm, short ym, short xM, short yM) {
	dim3 block(BLOCKX, BLOCKY), grid(pbaTexSize / block.x, pbaTexSize / block.y);
	cudaTextureObject_t texColor = bindTextureObject(pbaTextureFT, pbaTexSize * pbaTexSize * sizeof(short2));
	cudaTextureObject_t texParam = bindTextureObject(pbaTexSiteParam, pbaTexSize * pbaTexSize * sizeof(float));

	cudaMemset(pbaTextureThreshSkel, 0, pbaTexSize * pbaTexSize * sizeof(unsigned char));
	kernelSkel << <grid, block >> > (pbaTextureThreshSkel, xm, ym, xM - 1, yM - 1, pbaTexSize, threshold, length, texColor, texParam);

	cudaDestroyTextureObject(texColor);
	cudaDestroyTextureObject(texParam);
	if (outputSkel) cudaMemcpy(outputSkel, pbaTextureThreshSkel, pbaTexSize * pbaTexSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
}




__global__ void kernelTopology(unsigned char* output, short2* output_set, short xm, short ym, short xM, short yM, short size, int maxpts, cudaTextureObject_t pbaTexGray)
{
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned char t[9];

	if (tx > xm && ty > ym && tx < xM - 1 && ty < yM - 1)
	{
		int id = TOID(tx, ty, size);
		unsigned char p = tex1Dfetch<unsigned char>(pbaTexGray, id);

		if (p) // If the pixel is part of the skeleton
		{
			unsigned char idx = 0;
			for (int j = ty - 1; j <= ty + 1; ++j)
			{
				int fetch_id = TOID(tx - 1, j, size);
				for (int i = 0; i <= 2; ++i, ++fetch_id, ++idx)
				{
					t[idx] = tex1Dfetch<unsigned char>(pbaTexGray, fetch_id);
				}
			}

			for (unsigned char r = 0; r < 4; ++r)
			{
				const unsigned char* rr = topo_rot[r];
				for (unsigned char p_idx = 0; p_idx < topo_NPATTERNS; ++p_idx)
				{
					const unsigned char* pat = topo_patterns[p_idx];
					unsigned char j = (p_idx == 0) ? 0 : 7;
					for (; j < 9; ++j)
						if (pat[j] != t[rr[j]]) break;

					if (j < 6) break;

					if (j == 9)
					{
						int crt_gc = atomicInc(&topo_gc, maxpts);
						output_set[crt_gc] = make_short2((short)tx, (short)ty);
						output[id] = 0xff;
						return;
					}
				}
			}
		}
	}
}




__global__ void kernelSiteFromSkeleton(short2* outputSites, int size)						//Initialize the Voronoi textures from the sites' encoding texture (parameterization)
{																							//REMARK: we interpret 'inputVoro' as a 2D texture, as it's much easier/faster like this
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx<size && ty<size)																	//Careful not to go outside the image..
	{
	  int i = TOID(tx,ty,size);
	  unsigned char param = tex1Dfetch<unsigned char>(pbaTexGray,i);										//The sites-param has non-zero (parameter) values precisely on non-boundary points

	  short2& v = outputSites[i];
	  v.x = v.y = MARKER;																	//Non-boundary points are marked as 0 in the parameterization. Here we will compute the FT.
	  if (param)																			//These are points which define the 'sites' to compute the FT/skeleton (thus, have FT==identity)
	  {																						//We could use an if-then-else here, but it's faster with an if-then
	     v.x = tx; v.y = ty;
	  }
	}
}




__global__ void kernelSkelInterpolate(float* output, int size)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	if (tx < size && ty < size) {
		int id = TOID(tx, ty, size);
		short2 vid = tex1Dfetch<short2>(pbaTexColor, id);
		float T = sqrtf((tx - vid.x) * (tx - vid.x) + (ty - vid.y) * (ty - vid.y));
		short2 vid2 = tex1Dfetch<short2>(pbaTexColor2, id);
		float D = sqrtf((tx - vid2.x) * (tx - vid2.x) + (ty - vid2.y) * (ty - vid2.y));
		float B = ((D) ? fminf(T / 2 / D, 0.5f) : 0.5f) + 0.5f * ((T) ? fmaxf(1 - D / T, 0.0f) : 0);
		output[id] = B;
	}
}




void skel2DSkeletonDT(float* outputSkelDT, short xm, short ym, short xM, short yM) {
	dim3 block(BLOCKX, BLOCKY), grid(pbaTexSize / block.x, pbaTexSize / block.y);
	cudaTextureObject_t texGray = bindTextureObject(pbaTextureThreshSkel, pbaTexSize * pbaTexSize * sizeof(unsigned char));
	kernelSiteFromSkeleton << <grid, block >> > (pbaTextureWork, pbaTexSize, texGray);
	cudaDestroyTextureObject(texGray);

	cudaMemcpy(pbaTextureWorkTopo, pbaTextureFT, pbaTexSize * pbaTexSize * sizeof(short2), cudaMemcpyDeviceToDevice);
	skel2DFTCompute(xm, ym, xM, yM, floodBand, maurerBand, colorBand);
	cudaMemcpy(pbaTextureSkeletonFT, pbaTextureFT, pbaTexSize * pbaTexSize * sizeof(short2), cudaMemcpyDeviceToDevice);
	cudaMemcpy(pbaTextureFT, pbaTextureWorkTopo, pbaTexSize * pbaTexSize * sizeof(short2), cudaMemcpyDeviceToDevice);

	cudaTextureObject_t texColor = bindTextureObject(pbaTextureFT, pbaTexSize * pbaTexSize * sizeof(short2));
	cudaTextureObject_t texColor2 = bindTextureObject(pbaTextureSkeletonFT, pbaTexSize * pbaTexSize * sizeof(short2));
	kernelSkelInterpolate << <grid, block >> > ((float*)pbaTextureWork, pbaTexSize, texColor, texColor2);
	cudaDestroyTextureObject(texColor);
	cudaDestroyTextureObject(texColor2);

	if (outputSkelDT) cudaMemcpy(outputSkelDT, pbaTextureWork, pbaTexSize * pbaTexSize * sizeof(float), cudaMemcpyDeviceToHost);
}




__device__  bool fill_gc;														//Indicates if a fill-sweep did fill anything or not


__global__ void kernelFill(unsigned char* output, int size, unsigned char bg, unsigned char fg, short xm, short ym, short xM, short yM, bool ne)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	if (tx > xm && ty > ym && tx < xM && ty < yM) {
		int id0 = TOID(tx, ty, size);
		if (tex1Dfetch<unsigned char>(pbaTexGray, id0) == fg) {
			bool fill = false;
			int id = id0;
			if (ne) {
				for (short x = tx + 1; x < xM; ++x) {
					if (tex1Dfetch<unsigned char>(pbaTexGray, ++id) != bg) break;
					output[id] = fg; fill = true;
				}
				id = id0;
				for (short y = ty - 1; y > ym; --y) {
					if (tex1Dfetch<unsigned char>(pbaTexGray, id -= size) != bg) break;
					output[id] = fg; fill = true;
				}
			}
			else {
				for (short x = tx - 1; x > xm; --x) {
					if (tex1Dfetch<unsigned char>(pbaTexGray, --id) != bg) break;
					output[id] = fg; fill = true;
				}
				id = id0;
				for (short y = ty + 1; y < yM; ++y) {
					if (tex1Dfetch<unsigned char>(pbaTexGray, id += size) != bg) break;
					output[id] = fg; fill = true;
				}
			}
			if (fill) fill_gc = true;
		}
	}
}




__global__ void kernelFillHoles(unsigned char* output, int size, unsigned char bg, unsigned char fg, unsigned char fill_fg)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	if (tx >= 0 && ty >= 0 && tx < size && ty < size) {
		int id = TOID(tx, ty, size);
		unsigned char val = tex1Dfetch<unsigned char>(pbaTexGray, id);
		if (val == fill_fg) output[id] = bg;
		else if (val == bg) output[id] = fg;
	}
}


int skelft2DFill(unsigned char* outputFill, short sx, short sy, short xm, short ym, short xM, short yM, unsigned char fill_value) {
	dim3 block(BLOCKX, BLOCKY), grid(pbaTexSize / block.x, pbaTexSize / block.y);
	unsigned char background;
	cudaMemcpy(&background, pbaTextureThreshDT + (sy * pbaTexSize + sx), sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaMemset(pbaTextureThreshDT + (sy * pbaTexSize + sx), fill_value, sizeof(unsigned char));

	int iter = 0; bool xy = true;
	for (;; ++iter, xy = !xy) {
		bool filled = false;
		cudaMemcpyToSymbol(fill_gc, &filled, sizeof(bool));
		cudaTextureObject_t texGray = bindTextureObject(pbaTextureThreshDT, pbaTexSize * pbaTexSize * sizeof(unsigned char));
		kernelFill << <grid, block >> > (pbaTextureThreshDT, pbaTexSize, background, fill_value, xm, ym, xM, yM, xy, texGray);
		cudaDestroyTextureObject(texGray);
		cudaMemcpyFromSymbol(&filled, fill_gc, sizeof(bool));
		if (!filled) break;
	}
	if (outputFill) cudaMemcpy(outputFill, pbaTextureThreshDT, pbaTexSize * pbaTexSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	return iter;
}



int skelft2DFillHoles(unsigned char* outputFill, short sx, short sy, unsigned char foreground)
{
	unsigned char background, fill_value = 128;
	cudaMemcpy(&background, pbaTextureThreshDT + (sy * pbaTexSize + sx), sizeof(unsigned char), cudaMemcpyDeviceToHost);
	int iter = skelft2DFill(0, sx, sy, 0, 0, pbaTexSize, pbaTexSize, fill_value);

	dim3 block(BLOCKX, BLOCKY), grid(pbaTexSize / block.x, pbaTexSize / block.y);
	cudaTextureObject_t texGray = bindTextureObject(pbaTextureThreshDT, pbaTexSize * pbaTexSize * sizeof(unsigned char));
	kernelFillHoles << <grid, block >> > (pbaTextureThreshDT, pbaTexSize, background, foreground, fill_value, texGray);
	cudaDestroyTextureObject(texGray);

	if (outputFill) cudaMemcpy(outputFill, pbaTextureThreshDT, pbaTexSize * pbaTexSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	return iter;
}
