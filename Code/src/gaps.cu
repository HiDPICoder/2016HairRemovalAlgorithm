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

__global__ void kernelInvertImage(unsigned char* output, int xM, int yM, int size, cudaTextureObject_t pbaTexGray)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if (tx < xM && ty < yM)
	{
		int id = TOID(tx, ty, size);
		unsigned char val = tex1Dfetch<unsigned char>(pbaTexGray, id);
		output[id] = 0xff - val;
	}
}

void invertImage()
{
	dim3 block = dim3(BLOCKX,BLOCKY);
	dim3 grid  = dim3(pbaTexSize/block.x,pbaTexSize/block.y);

	cudaTextureObject_t texObj = bindTextureObject(cudaImageOriginal, pbaTexSize * pbaTexSize * sizeof(unsigned char));
	kernelInvertImage << < grid, block >> > (cudaImageOriginal, sizeX, sizeY, pbaTexSize, texObj);
	cudaDestroyTextureObject(texObj);
}

void clearDetectedGaps()
{
	cudaMemset(cudaImageDetectedGaps, 0, pbaTexSize*pbaTexSize*sizeof(unsigned char));
}

__global__ void kernelThreshold(unsigned char* output, unsigned char* changed, int level, int xM, int yM, int size, cudaTextureObject_t pbaTexGray)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if (tx < xM && ty < yM)
	{
		int id = TOID(tx, ty, size);
		unsigned char val = tex1Dfetch<unsigned char>(pbaTexGray, id);

		output[id] = val >= level ? 0xff : 0;
		if (val == level) *changed = 1;
	}
}

bool computeThreshold(int level)
{
	dim3 block = dim3(BLOCKX, BLOCKY);
	dim3 grid = dim3(pbaTexSize / block.x, pbaTexSize / block.y);

	cudaMemset(pbaTextureThreshDT, 0, sizeof(unsigned char));

	cudaTextureObject_t texObj = bindTextureObject(cudaImageOriginal, pbaTexSize * pbaTexSize * sizeof(unsigned char));
	kernelThreshold << < grid, block >> > (cudaImageThresholded, pbaTextureThreshDT, level, sizeX, sizeY, pbaTexSize, texObj);
	cudaDestroyTextureObject(texObj);

	unsigned char changed;
	cudaMemcpy(&changed, pbaTextureThreshDT, sizeof(unsigned char), cudaMemcpyDeviceToHost);

	return changed != 0;
}

void morphDilate(unsigned char* out, unsigned char* image, float radius)
{
	dim3 block = dim3(BLOCKX, BLOCKY);
	dim3 grid = dim3(pbaTexSize / block.x, pbaTexSize / block.y);

	int xm = 0, ym = 0, xM = sizeX, yM = sizeY;
	xM += (int)radius; if (xM > pbaTexSize - 1) xM = pbaTexSize - 1;
	yM += (int)radius; if (yM > pbaTexSize - 1) yM = pbaTexSize - 1;

	cudaTextureObject_t texGray = bindTextureObject(image, pbaTexSize * pbaTexSize * sizeof(unsigned char));
	kernelSiteParamInitChar << <grid, block >> > (pbaTextureWork, pbaTexSize, texGray);
	cudaDestroyTextureObject(texGray);

	skel2DFTCompute(xm, ym, xM, yM, floodBand, maurerBand, colorBand);

	cudaTextureObject_t texColor = bindTextureObject(pbaTextureFT, pbaTexSize * pbaTexSize * sizeof(short2));
	kernelThresholdDT << < grid, block >> > (out, pbaTexSize, radius * radius, xm - 1, ym - 1, xM + 1, yM + 1, texColor);
	cudaDestroyTextureObject(texColor);
}

void morphErode(unsigned char* out, unsigned char* image, float radius)
{
	dim3 block = dim3(BLOCKX, BLOCKY);
	dim3 grid = dim3(pbaTexSize / block.x, pbaTexSize / block.y);

	int xm = 0, ym = 0, xM = sizeX, yM = sizeY;
	xM += (int)radius; if (xM > pbaTexSize - 1) xM = pbaTexSize - 1;
	yM += (int)radius; if (yM > pbaTexSize - 1) yM = pbaTexSize - 1;

	cudaTextureObject_t texGray = bindTextureObject(image, pbaTexSize * pbaTexSize * sizeof(unsigned char));
	kernelSiteParamInitCharInverse << <grid, block >> > (pbaTextureWork, pbaTexSize, texGray);
	cudaDestroyTextureObject(texGray);

	skel2DFTCompute(xm, ym, xM, yM, floodBand, maurerBand, colorBand);

	cudaTextureObject_t texColor = bindTextureObject(pbaTextureFT, pbaTexSize * pbaTexSize * sizeof(short2));
	kernelThresholdDTInverse << < grid, block >> > (out, pbaTexSize, radius * radius, xm - 1, ym - 1, xM + 1, yM + 1, texColor);
	cudaDestroyTextureObject(texColor);
}

void computeMorphs(unsigned char* out, float radius)
{
	morphDilate(cudaImageOpenClose, cudaImageThresholded, radius);
	morphErode(cudaImageOpenClose, cudaImageOpenClose, radius * 2.0f);
	morphDilate(cudaImageOpenClose, cudaImageOpenClose, radius);

	morphErode(cudaImageCloseOpen, cudaImageThresholded, radius);
	morphDilate(cudaImageCloseOpen, cudaImageCloseOpen, radius * 2.0f);
	morphErode(cudaImageCloseOpen, cudaImageCloseOpen, radius);

	cudaMemcpy(out, cudaImageOpenClose, pbaTexSize * pbaTexSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
}

// Only selects inflated pixels which were not in the original image, i.e. the gaps we are interested in
__global__ void kernelReconstruct(unsigned char* output, int size, int level, cudaTextureObject_t pbaTexGray, cudaTextureObject_t pbaTexOriginal)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if (tx < size && ty < size) {
		int id = TOID(tx, ty, size);
		unsigned char infl = tex1Dfetch<unsigned char>(pbaTexGray, id); // inflated pixels from skeleton
		unsigned char im = tex1Dfetch<unsigned char>(pbaTexOriginal, id); // original image

		if (im && infl) output[id] = (unsigned char)level;
	}
}

__global__ void kernelSplatInterpolate(unsigned char* output, float lambda, int size, int numpts, short xm, short ym, short xM, short yM, cudaTextureObject_t pbaTexColor, cudaTextureObject_t pbaTexLinks, cudaTextureObject_t pbaTexColor2)
{
	int offs = blockIdx.x * blockDim.x + threadIdx.x;

	if (offs < numpts)
	{
		short2 skel = tex1Dfetch<short2>(pbaTexColor, offs);
		int tx = skel.x, ty = skel.y;
		int id = TOID(tx, ty, size);
		short2 voroidOC = tex1Dfetch<short2>(pbaTexLinks, id);
		short2 voroidCO = tex1Dfetch<short2>(pbaTexColor2, id);

		float dOC2 = (float)(tx - voroidOC.x) * (tx - voroidOC.x) + (float)(ty - voroidOC.y) * (ty - voroidOC.y);
		float dCO2 = (float)(tx - voroidCO.x) * (tx - voroidCO.x) + (float)(ty - voroidCO.y) * (ty - voroidCO.y);
		float d2 = (1.0f - lambda) * dOC2 + lambda * dCO2;
		short d = (short)sqrtf(d2);

		short jmin = max((int)(ty - d), (int)ym);
		short jmax = min((int)(ty + d), (int)yM);

		for (short j = jmin; j <= jmax; ++j)
		{
			short w = (short)sqrtf(fmaxf(0.0f, d2 - (float)(j - ty) * (j - ty)));
			short imin = max((int)(tx - w), (int)xm);
			short imax = min((int)(tx + w), (int)xM);

			int start_id = TOID(imin, j, size);
			for (short i = imin; i <= imax; ++i)
			{
				output[start_id++] = 0xff;
			}
		}
	}
}

__global__ void kernelSplatInterpolateFullScan(unsigned char* output, float lambda, int size, int numpts, short xm, short ym, short xM, short yM, cudaTextureObject_t pbaTexColor, cudaTextureObject_t pbaTexLinks, cudaTextureObject_t pbaTexColor2)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if (tx > xm && ty > ym && tx < xM && ty < yM)
	{
		int id = TOID(tx, ty, size);
		for (int i = 0; i < numpts; i++)
		{
			short2 skel = tex1Dfetch<short2>(pbaTexColor, i);
			int sid = TOID(skel.x, skel.y, size);

			short2 voroidOC = tex1Dfetch<short2>(pbaTexLinks, sid);
			short2 voroidCO = tex1Dfetch<short2>(pbaTexColor2, sid);

			float dOC2 = (float)(skel.x - voroidOC.x) * (skel.x - voroidOC.x) + (float)(skel.y - voroidOC.y) * (skel.y - voroidOC.y);
			float dCO2 = (float)(skel.x - voroidCO.x) * (skel.x - voroidCO.x) + (float)(skel.y - voroidCO.y) * (skel.y - voroidCO.y);
			float dft2 = (1.0f - lambda) * dOC2 + lambda * dCO2;

			float ds2 = (float)(skel.x - tx) * (skel.x - tx) + (float)(skel.y - ty) * (skel.y - ty);

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
	dim3 block = dim3(BLOCKX, BLOCKY);
	dim3 grid = dim3(pbaTexSize / block.x, pbaTexSize / block.y);

	unsigned int num_pts = 0;
	cudaMemcpyToSymbol(topo_gc, &num_pts, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);

	cudaTextureObject_t texGray = bindTextureObject(pbaTextureThreshSkel, pbaTexSize * pbaTexSize * sizeof(unsigned char));
	cudaTextureObject_t texOrig = bindTextureObject(cudaImageThresholded, pbaTexSize * pbaTexSize * sizeof(unsigned char));
	cudaTextureObject_t texOpenClosed = bindTextureObject(cudaImageOpenClose, pbaTexSize * pbaTexSize * sizeof(unsigned char));

	kernelGatherSkelPixelsMasked << < grid, block >> > (pbaTextureWorkTopo, pbaTexSize, xm, ym, xM, yM, texGray, texOrig, texOpenClosed);

	cudaDestroyTextureObject(texGray);
	cudaDestroyTextureObject(texOrig);
	cudaDestroyTextureObject(texOpenClosed);

	cudaMemcpyFromSymbol(&num_pts, topo_gc, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
	cudaMemset(pbaTextureThreshDT, 0, sizeof(unsigned char) * pbaTexSize * pbaTexSize);

	cudaTextureObject_t texColor = bindTextureObject(pbaTextureWorkTopo, pbaTexSize * pbaTexSize * sizeof(short2));
	cudaTextureObject_t texColor2 = bindTextureObject(pbaTextureFT, pbaTexSize * pbaTexSize * sizeof(short2));
	cudaTextureObject_t texLinks = bindTextureObject(pbaTextureSkeletonFT, pbaTexSize * pbaTexSize * sizeof(short2));

	if (num_pts > 300)
	{
		dim3 blockSplat(BLOCKX * BLOCKY);
		dim3 gridSplat((num_pts + blockSplat.x - 1) / blockSplat.x);
		kernelSplatInterpolate << < gridSplat, blockSplat >> > (pbaTextureThreshDT, lambda, pbaTexSize, (int)num_pts, xm, ym, xM, yM, texColor, texLinks, texColor2);
	}
	else
	{
		kernelSplatInterpolateFullScan << < grid, block >> > (pbaTextureThreshDT, lambda, pbaTexSize, (int)num_pts, xm, ym, xM, yM, texColor, texLinks, texColor2);
	}

	cudaDestroyTextureObject(texColor);
	cudaDestroyTextureObject(texColor2);
	cudaDestroyTextureObject(texLinks);

	if (result) cudaMemcpy(result, pbaTextureThreshDT, pbaTexSize * pbaTexSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	return (int)num_pts;
}

// Perform gap detection algorithm described in Sobiecki et al. (Gap-sensitive segmentation
// and restoration of digital images. In: Proc. CGVC. pp. 136–144. Eurographics (2014))
void computeDetectedGaps(int level, float lambda)
{
	dim3 block = dim3(BLOCKX, BLOCKY);
	dim3 grid = dim3(pbaTexSize / block.x, pbaTexSize / block.y);

	int xm = 0, ym = 0, xM = sizeX, yM = sizeY;

	cudaTextureObject_t texGray = bindTextureObject(cudaImageOpenClose, pbaTexSize * pbaTexSize * sizeof(unsigned char));
	kernelSiteParamInitChar << <grid, block >> > (pbaTextureWork, pbaTexSize, texGray);
	cudaDestroyTextureObject(texGray);

	skel2DFTCompute(xm, ym, xM, yM, floodBand, maurerBand, colorBand);
	cudaMemcpy(pbaTextureSkeletonFT, pbaTextureFT, pbaTexSize * pbaTexSize * sizeof(short2), cudaMemcpyDeviceToDevice);

	texGray = bindTextureObject(cudaImageCloseOpen, pbaTexSize * pbaTexSize * sizeof(unsigned char));
	kernelSiteParamInitChar << <grid, block >> > (pbaTextureWork, pbaTexSize, texGray);
	cudaDestroyTextureObject(texGray);

	skel2DFTCompute(xm, ym, xM, yM, floodBand, maurerBand, colorBand);

	int inflated = skelft2DInflateInterpolate(NULL, lambda, xm, ym, xM, yM);

	if (inflated > 0)
	{
		cudaTextureObject_t texRes = bindTextureObject(pbaTextureThreshDT, pbaTexSize * pbaTexSize * sizeof(unsigned char));
		cudaTextureObject_t texOrig = bindTextureObject(cudaImageThresholded, pbaTexSize * pbaTexSize * sizeof(unsigned char));
		kernelReconstruct << <grid, block >> > (cudaImageDetectedGaps, pbaTexSize, level, texRes, texOrig);
		cudaDestroyTextureObject(texRes);
		cudaDestroyTextureObject(texOrig);
	}
}

void copyDetectedGapsD2H(unsigned char* out)
{
	cudaMemcpy(out, cudaImageDetectedGaps, pbaTexSize * pbaTexSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
}

// Compute the final inflation from the filtered skeleton, this results in the final, filtered hair mask
void computeInflation(unsigned char* result, unsigned char* detectedGaps, unsigned char* skeleton)
{
	cudaMemcpy(pbaTextureThreshDT, detectedGaps, pbaTexSize * pbaTexSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(pbaTextureThreshSkel, skeleton, pbaTexSize * pbaTexSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

	dim3 block = dim3(BLOCKX, BLOCKY);
	dim3 grid = dim3(pbaTexSize / block.x, pbaTexSize / block.y);

	int xm = 0, ym = 0, xM = sizeX, yM = sizeY;

	cudaTextureObject_t texGray = bindTextureObject(pbaTextureThreshDT, pbaTexSize * pbaTexSize * sizeof(unsigned char));
	kernelSiteParamInitCharInverse << <grid, block >> > (pbaTextureWork, pbaTexSize, texGray);
	cudaDestroyTextureObject(texGray);

	skel2DFTCompute(xm, ym, xM, yM, floodBand, maurerBand, colorBand);
	skelft2DInflate(NULL, xm, ym, xM, yM);

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
