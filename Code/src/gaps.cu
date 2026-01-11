#include "skelft.cu"

// Various image buffers storing intermediate results.
unsigned char* cudaImageOriginal;
unsigned char* cudaImageThresholded;
unsigned char* cudaImageOpenClose;
unsigned char* cudaImageCloseOpen;
unsigned char* cudaImageDetectedGaps;

int sizeX, sizeY;

void allocateBuffers(int xM, int yM, int size)
{
	sizeX = xM; sizeY = yM;
	pbaTexSize = size;

	cudaMalloc(&cudaImageOriginal, size * size * sizeof(unsigned char));
	cudaMalloc(&cudaImageThresholded, size * size * sizeof(unsigned char));
	cudaMalloc(&cudaImageOpenClose, size * size * sizeof(unsigned char));
	cudaMalloc(&cudaImageCloseOpen, size * size * sizeof(unsigned char));
	cudaMalloc(&cudaImageDetectedGaps, size * size * sizeof(unsigned char));

	cudaMemset(cudaImageDetectedGaps, 0, size * size * sizeof(unsigned char));
}

void deallocateBuffers()
{
	cudaFree(cudaImageOriginal);
	cudaFree(cudaImageThresholded);
	cudaFree(cudaImageOpenClose);
	cudaFree(cudaImageCloseOpen);
	cudaFree(cudaImageDetectedGaps);
}

// Copy grayscale input image into GPU memory only once for all all thresholds
void loadImage(unsigned char* image)
{
	cudaMemcpy(cudaImageOriginal, image, pbaTexSize * pbaTexSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
}

__global__ void kernelInvertImage(unsigned char* output, int xM, int yM, int size)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if (tx < xM && ty < yM)
	{
		int id = TOID(tx, ty, size);
		unsigned char val = tex1Dfetch(pbaTexGray, id);

		output[id] = 0xff - val;
	}
}

void invertImage()
{
	dim3 block = dim3(BLOCKX,BLOCKY);
	dim3 grid  = dim3(pbaTexSize/block.x,pbaTexSize/block.y);

	cudaBindTexture(0, pbaTexGray, cudaImageOriginal);
	kernelInvertImage<<< grid, block >>>(cudaImageOriginal, sizeX, sizeY, pbaTexSize);
	cudaUnbindTexture(pbaTexGray);
}

void clearDetectedGaps()
{
	cudaMemset(cudaImageDetectedGaps, 0, pbaTexSize*pbaTexSize*sizeof(unsigned char));
}

__global__ void kernelThreshold(unsigned char* output, unsigned char *changed, int level, int xM, int yM, int size)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if (tx < xM && ty < yM)
	{
		int id = TOID(tx, ty, size);
		unsigned char val = tex1Dfetch(pbaTexGray, id);

		output[id] = val >= level ? 0xff : 0;
		if (val == level) *changed = 1;
	}
}

bool computeThreshold(int level)
{
	dim3 block = dim3(BLOCKX,BLOCKY);
	dim3 grid  = dim3(pbaTexSize/block.x,pbaTexSize/block.y);

	cudaMemset(pbaTextureThreshDT, 0, sizeof(unsigned char));

	cudaBindTexture(0, pbaTexGray, cudaImageOriginal);
	kernelThreshold<<< grid, block >>>(cudaImageThresholded, pbaTextureThreshDT, level, sizeX, sizeY, pbaTexSize);
	cudaUnbindTexture(pbaTexGray);

	unsigned char changed;
	cudaMemcpy(&changed, pbaTextureThreshDT, sizeof(unsigned char), cudaMemcpyDeviceToHost);

	return changed;
}

// Performs morphological dilation by tresholding EDT with the dilation circle kernel radius.
void morphDilate(unsigned char* out, unsigned char* image, float radius)
{
	dim3 block = dim3(BLOCKX,BLOCKY);
	dim3 grid  = dim3(pbaTexSize/block.x,pbaTexSize/block.y);

	int xm = 0, ym = 0, xM = sizeX, yM = sizeY;

	xM += radius; if (xM > pbaTexSize-1) xM = pbaTexSize-1;
	yM += radius; if (yM > pbaTexSize-1) yM = pbaTexSize-1;

	// Invert image into pbaTextureWork
	cudaBindTexture(0, pbaTexGray, image);
	kernelSiteParamInitChar<<<grid,block>>>(pbaTextureWork,pbaTexSize);
	cudaUnbindTexture(pbaTexGray);

	skel2DFTCompute(xm, ym, xM, yM, floodBand, maurerBand, colorBand);

	// Compute dilation from FT
	cudaBindTexture(0, pbaTexColor, pbaTextureFT);
	kernelThresholdDT<<< grid, block >>>(out, pbaTexSize, radius*radius, xm-1, ym-1, xM+1, yM+1);
	cudaUnbindTexture(pbaTexColor);
}

// Performs morphological erosion by tresholding EDT with the erosion circle kernel radius.
void morphErode(unsigned char* out, unsigned char* image, float radius)
{
	dim3 block = dim3(BLOCKX,BLOCKY);
	dim3 grid  = dim3(pbaTexSize/block.x,pbaTexSize/block.y);

	int xm = 0, ym = 0, xM = sizeX, yM = sizeY;

	xM += radius; if (xM > pbaTexSize-1) xM = pbaTexSize-1;
	yM += radius; if (yM > pbaTexSize-1) yM = pbaTexSize-1;

	// Copy image into pbaTextureWork
	cudaBindTexture(0, pbaTexGray, image);
	kernelSiteParamInitCharInverse<<<grid,block>>>(pbaTextureWork,pbaTexSize);
	cudaUnbindTexture(pbaTexGray);

	skel2DFTCompute(xm, ym, xM, yM, floodBand, maurerBand, colorBand);

	// Compute erosion from FT
	cudaBindTexture(0, pbaTexColor, pbaTextureFT);
	kernelThresholdDTInverse<<< grid, block >>>(out, pbaTexSize, radius*radius, xm-1, ym-1, xM+1, yM+1);
	cudaUnbindTexture(pbaTexColor);
}

// Compute both open-close and close-open and keep in CUDA memory as needed later.
void computeMorphs(unsigned char *out, float radius)
{
	morphDilate(cudaImageOpenClose, cudaImageThresholded, radius);
	morphErode(cudaImageOpenClose, cudaImageOpenClose, radius*2.0f);
	morphDilate(cudaImageOpenClose, cudaImageOpenClose, radius);

	morphErode(cudaImageCloseOpen, cudaImageThresholded, radius);
	morphDilate(cudaImageCloseOpen, cudaImageCloseOpen, radius*2.0f);
	morphErode(cudaImageCloseOpen, cudaImageCloseOpen, radius);

	cudaMemcpy(out, cudaImageOpenClose, pbaTexSize * pbaTexSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
}

// Only selects inflated pixels which were not in the original image, i.e. the gaps we are interested in
__global__ void kernelReconstruct(unsigned char* output, int size,int level)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	int id = TOID(tx, ty, size);
	unsigned char infl = tex1Dfetch(pbaTexGray, id); // inflated pixels from skeleton
	unsigned char im = tex1Dfetch(pbaTexOriginal, id); // original image

	if (im && infl) output[id] = level; // Outside of image but inside inflation are part of reconstruction
}

// From a list of skeleton points in `pbaTexColor`, draw circles with a radius determined by a lineair
// combination from the EDTs to both open-close and close-open image. This is a very expensive operation!
__global__ void kernelSplatInterpolate(unsigned char* output, float lambda, int size, int numpts, short xm, short ym, short xM, short yM)
{
    int offs = blockIdx.x * blockDim.x + threadIdx.x;

	if (offs < numpts)														//careful not to index outside the skel-vector..
	{
		short2     skel = tex1Dfetch(pbaTexColor,offs);						//splat the skel-point whose coords are at location offs in pbaTexColor[]
		int          tx = skel.x, ty = skel.y;
		int          id = TOID(tx,ty,size);
		short2 voroidOC = tex1Dfetch(pbaTexLinks,id);						//voroid of open-close to calculate distance from
		short2 voroidCO = tex1Dfetch(pbaTexColor2,id);						//voroid of close-open to calculate distance from

		float      dOC2 = (tx-voroidOC.x)*(tx-voroidOC.x)+(ty-voroidOC.y)*(ty-voroidOC.y);
		float      dCO2 = (tx-voroidCO.x)*(tx-voroidCO.x)+(ty-voroidCO.y)*(ty-voroidCO.y);
		float        d2 = (1-lambda) * dOC2 + lambda * dCO2;
		short         d = sqrtf(d2);

		short jmin = max(ty-d,ym);
		short jmax = min(ty+d,yM);

		for (short j=jmin;j<=jmax;++j)										//bounding-box of the splat at 'skel'
		{
			short w = sqrtf(d2-(j-ty)*(j-ty));
			short imin = max(tx-w,xm);
			short imax = min(tx+w,xM);

			int id = TOID(imin,j,size);

			for (short i=imin;i<=imax;++i)
			{
				output[id++] = 0xff;
			}
		}
    }
}

// Also draws circles centered at skeleton pixels with radius determined by EDT to open-close and close-open image.
// Different from above function in that for every pixel in the image, it iterates over all skeleton pixels to
// determine if it is close enough to fall in its circle. This seems somewhat faster when there are few skel pixels.
__global__ void kernelSplatInterpolateFullScan(unsigned char* output,float lambda, int size, int numpts, short xm, short ym, short xM, short yM)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	int id = TOID(tx,ty,size);

	if (tx>xm && ty>ym && tx<xM && ty<yM)										//careful not to index outside the skel-vector..
	{
		for (int i=0; i<numpts; i++)
		{
			short2 skel = tex1Dfetch(pbaTexColor,i);
			int sid = TOID(skel.x,skel.y,size);

			short2 voroidOC = tex1Dfetch(pbaTexLinks,sid);						//voroid of open-close to calculate distance from
			short2 voroidCO = tex1Dfetch(pbaTexColor2,sid);						//voroid of close-open to calculate distance from

			float      dOC2 = (skel.x-voroidOC.x)*(skel.x-voroidOC.x)+(skel.y-voroidOC.y)*(skel.y-voroidOC.y);
			float      dCO2 = (skel.x-voroidCO.x)*(skel.x-voroidCO.x)+(skel.y-voroidCO.y)*(skel.y-voroidCO.y);
			float      dft2 = (1-lambda) * dOC2 + lambda * dCO2;

			float ds2 = (skel.x-tx)*(skel.x-tx)+(skel.y-ty)*(skel.y-ty); // distance from skel to pixel

			if (ds2 <= dft2)
			{
				output[id] = 0xff;
				break;
			}
		}
    }
}

// Inflate the skeleton currently in `pbaTextureThreshSkel` based on the DFT to open-close and close-open.
int skelft2DInflateInterpolate(unsigned char* result,float lambda,short xm,short ym,short xM,short yM)
{
	dim3 block = dim3(BLOCKX,BLOCKY);
	dim3 grid  = dim3(pbaTexSize/block.x,pbaTexSize/block.y);

	unsigned int num_pts = 0;													//Set topo_gc to 0
	cudaMemcpyToSymbol(topo_gc,&num_pts,sizeof(unsigned int),0,cudaMemcpyHostToDevice);

	cudaBindTexture(0, pbaTexGray, pbaTextureThreshSkel);						//This is the skeleton
	cudaBindTexture(0, pbaTexOriginal, cudaImageThresholded);						//This is the original image, used as mask
	cudaBindTexture(0, pbaTexOpenClosed, cudaImageOpenClose);						//This is the open-closed image, used as skeleton mask
    kernelGatherSkelPixelsMasked<<< grid, block >>>(pbaTextureWorkTopo,pbaTexSize,xm,ym,xM,yM);
	cudaUnbindTexture(pbaTexGray);
	cudaUnbindTexture(pbaTexOriginal);
	cudaUnbindTexture(pbaTexOpenClosed);

	cudaMemcpyFromSymbol(&num_pts,topo_gc,sizeof(unsigned int),0,cudaMemcpyDeviceToHost);//Get #skel-points we have detected from the device-var from CUDA


	cudaMemset(pbaTextureThreshDT,0,sizeof(unsigned char)*pbaTexSize*pbaTexSize);//Faster to zero result and then fill only 1-values (see kernel)

	cudaBindTexture(0, pbaTexColor, pbaTextureWorkTopo);						//Skel points are in a short2-vector in work topo list
	cudaBindTexture(0, pbaTexColor2, pbaTextureFT);								//Also bind the FT to calculate the distance from
	cudaBindTexture(0, pbaTexLinks, pbaTextureSkeletonFT);

	// For a small number of points, comparing every pixel with all points seems cheaper.
	if (num_pts > 300)
	{
		block = dim3(BLOCKX*BLOCKY);												//Prepare the splatting kernel: this operates on a vector of 2D skel points
		int numpts_b = (num_pts/block.x+1)*block.x;									//Find higher multiple of blocksize than # skel points
		grid  = dim3(numpts_b/block.x);

		kernelSplatInterpolate<<< grid, block >>>(pbaTextureThreshDT, lambda, pbaTexSize, num_pts, xm, ym, xM, yM);
	}
	else
	{
		kernelSplatInterpolateFullScan<<< grid, block >>>(pbaTextureThreshDT, lambda, pbaTexSize, num_pts, xm, ym, xM, yM);
	}

	cudaUnbindTexture(pbaTexColor);
	cudaUnbindTexture(pbaTexColor2);
	cudaUnbindTexture(pbaTexLinks);

	if (result) cudaMemcpy(result, pbaTextureThreshDT, pbaTexSize * pbaTexSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	//Return #skel points processed
	return num_pts;
}

// Perform gap detection algorithm described in Sobiecki et al. (Gap-sensitive segmentation
// and restoration of digital images. In: Proc. CGVC. pp. 136–144. Eurographics (2014))
void computeDetectedGaps(int level, float lambda)
{
	dim3 block = dim3(BLOCKX,BLOCKY);
	dim3 grid  = dim3(pbaTexSize/block.x,pbaTexSize/block.y);

	int xm = 0, ym = 0, xM = sizeX, yM = sizeY;

	// Copy image into pbaTextureWork
	cudaBindTexture(0, pbaTexGray, cudaImageOpenClose);
	kernelSiteParamInitChar<<<grid,block>>>(pbaTextureWork,pbaTexSize);
	cudaUnbindTexture(pbaTexGray);

	// Calculate FT from open-close image, used to determine inflation distances from
	skel2DFTCompute(xm, ym, xM, yM, floodBand, maurerBand, colorBand);

	cudaMemcpy(pbaTextureSkeletonFT, pbaTextureFT, pbaTexSize * pbaTexSize * sizeof(short2), cudaMemcpyDeviceToDevice);



	// Copy image into pbaTextureWork
	cudaBindTexture(0, pbaTexGray, cudaImageCloseOpen);
	kernelSiteParamInitChar<<<grid,block>>>(pbaTextureWork,pbaTexSize);
	cudaUnbindTexture(pbaTexGray);

	// Calculate FT from close-open image, used to determine inflation distances from
	skel2DFTCompute(xm, ym, xM, yM, floodBand, maurerBand, colorBand);

	// Inflate skeleton
	int inflated = skelft2DInflateInterpolate(NULL, lambda, xm, ym, xM, yM);

	if (inflated > 0)
	{
		// Compute reconstructed pixels
		cudaBindTexture(0, pbaTexGray, pbaTextureThreshDT); // inflated result is in pbaTextureThreshDT
		cudaBindTexture(0, pbaTexOriginal, cudaImageThresholded);
		kernelReconstruct<<<grid,block>>>(cudaImageDetectedGaps,pbaTexSize,level);
		cudaUnbindTexture(pbaTexGray);
		cudaUnbindTexture(pbaTexOriginal);
    }
}

void copyDetectedGapsD2H(unsigned char* out)
{
	cudaMemcpy(out, cudaImageDetectedGaps, pbaTexSize * pbaTexSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
}

// Compute the final inflation from the filtered skeleton, this results in the final, filtered hair mask
void computeInflation(unsigned char* result, unsigned char* detectedGaps, unsigned char* skeleton)
{
	cudaMemcpy(pbaTextureThreshDT, detectedGaps, pbaTexSize*pbaTexSize*sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(pbaTextureThreshSkel, skeleton, pbaTexSize*pbaTexSize*sizeof(unsigned char), cudaMemcpyHostToDevice);

	dim3 block = dim3(BLOCKX,BLOCKY);
	dim3 grid  = dim3(pbaTexSize/block.x,pbaTexSize/block.y);

	int xm = 0, ym = 0, xM = sizeX, yM = sizeY;

	// Copy image into pbaTextureWork
	cudaBindTexture(0, pbaTexGray, pbaTextureThreshDT);
	kernelSiteParamInitCharInverse<<<grid,block>>>(pbaTextureWork,pbaTexSize);
	cudaUnbindTexture(pbaTexGray);

	// Calculate FT from detectedGaps image, used to determine inflation distances from
	skel2DFTCompute(xm, ym, xM, yM, floodBand, maurerBand, colorBand);

	skelft2DInflate(NULL, xm, ym, xM, yM);

	// Extend result by 2 pixels to properly overlap areas to be inpainted
	morphDilate(pbaTextureThreshDT, pbaTextureThreshDT, 2.0f);

	cudaMemcpy(result, pbaTextureThreshDT, pbaTexSize * pbaTexSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
}

// Used for debugging, copies a GPU buffer to CPU memory.
void copyImage(int which, void *output)
{
	unsigned char *ch = NULL; short2 *sh = NULL; float *fl = NULL;

	switch (which)
	{
		case 0: ch = cudaImageOriginal; break;
		case 1: ch = cudaImageThresholded; break;
		case 2: ch = cudaImageDetectedGaps; break;
		case 3: ch = pbaTextureThreshDT; break;
		case 4: ch = pbaTextureThreshSkel; break;
	}

	switch (which)
	{
		case 5: fl = pbaTexSiteParam; break;
	}

	if (ch) cudaMemcpy(output, ch, pbaTexSize * pbaTexSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (sh) cudaMemcpy(output, sh, pbaTexSize * pbaTexSize * sizeof(short2), cudaMemcpyDeviceToHost);
	if (fl) cudaMemcpy(output, fl, pbaTexSize * pbaTexSize * sizeof(float), cudaMemcpyDeviceToHost);
}
