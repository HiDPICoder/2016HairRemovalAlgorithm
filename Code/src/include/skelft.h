#pragma once

#include <cuda_runtime.h>

//====================================================================================================================
// Kernel Prototypes for Texture Object API
// Each prototype now explicitly includes the cudaTextureObject_t parameter(s).
//====================================================================================================================

// Initialization & Logic
__global__ void kernelSiteParamInit(short2* inputVoro, int size, cudaTextureObject_t pbaTexParam);
__global__ void kernelSiteParamInitChar(short2* inputVoro, int size, cudaTextureObject_t pbaTexGray);
__global__ void kernelSiteParamInitCharInverse(short2* inputVoro, int size, cudaTextureObject_t pbaTexGray);

// PBA Core (The Feature Transform)
__global__ void kernelTranspose(short2 *data, int size, cudaTextureObject_t pbaTexColor);
__global__ void kernelFloodDown(short2 *output, int size, int bandSize, cudaTextureObject_t pbaTexColor);
__global__ void kernelFloodUp(short2 *output, int size, int bandSize, cudaTextureObject_t pbaTexColor);
__global__ void kernelPropagateInterband(short2 *output, int size, int bandSize, cudaTextureObject_t pbaTexColor);
__global__ void kernelUpdateVertical(short2 *output, int size, int band, int bandSize, cudaTextureObject_t pbaTexColor, cudaTextureObject_t pbaTexLinks);
__global__ void kernelProximatePoints(short2 *stack, int size, int bandSize, cudaTextureObject_t pbaTexColor);
__global__ void kernelCreateForwardPointers(short2 *output, int size, int bandSize, cudaTextureObject_t pbaTexLinks);
__global__ void kernelMergeBands(short2 *output, int size, int bandSize, cudaTextureObject_t pbaTexColor, cudaTextureObject_t pbaTexLinks);
__global__ void kernelDoubleToSingleList(short2 *output, int size, cudaTextureObject_t pbaTexColor, cudaTextureObject_t pbaTexLinks);
__global__ void kernelColor(short2 *output, int size, cudaTextureObject_t pbaTexColor);

// Skeleton & Distance Transform
__global__ void kernelThresholdDT(unsigned char* output, int size, float threshold2, short xm, short ym, short xM, short yM, cudaTextureObject_t pbaTexColor);
__global__ void kernelThresholdDTInverse(unsigned char* output, int size, float threshold2, short xm, short ym, short xM, short yM, cudaTextureObject_t pbaTexColor);
__global__ void kernelDT(short* output, int size, float threshold2, short xm, short ym, short xM, short yM, cudaTextureObject_t pbaTexColor);
__global__ void kernelSkel(unsigned char* output, short xm, short ym, short xM, short yM, short size, float threshold, float length, cudaTextureObject_t pbaTexColor, cudaTextureObject_t pbaTexParam);
__global__ void kernelSiteFromSkeleton(short2* outputSites, int size, cudaTextureObject_t pbaTexGray);
__global__ void kernelSkelInterpolate(float* output, int size, cudaTextureObject_t pbaTexColor, cudaTextureObject_t pbaTexColor2);

// Topology & Reconstruction
__global__ void kernelGatherSkelPixels(short2* output, int size, short xm, short ym, short xM, short yM, cudaTextureObject_t pbaTexGray);
__global__ void kernelGatherSkelPixelsMasked(short2* output, int size, short xm, short ym, short xM, short yM, cudaTextureObject_t pbaTexGray, cudaTextureObject_t pbaTexOriginal, cudaTextureObject_t pbaTexOpenClosed);
__global__ void kernelTopology(unsigned char* output, short2* output_set, short xm, short ym, short xM, short yM, short size, int maxpts, cudaTextureObject_t pbaTexGray);
__global__ void kernelSplat(unsigned char* output, int size, int numpts, short xm, short ym, short xM, short yM, cudaTextureObject_t pbaTexColor, cudaTextureObject_t pbaTexLinks);
__global__ void kernelReconstruct(unsigned char* output, int size, int level, cudaTextureObject_t pbaTexGray, cudaTextureObject_t pbaTexOriginal);

// Interpolated Splatting (Gap Detection)
__global__ void kernelSplatInterpolate(unsigned char* output, float lambda, int size, int numpts, short xm, short ym, short xM, short yM, cudaTextureObject_t pbaTexColor, cudaTextureObject_t pbaTexLinks, cudaTextureObject_t pbaTexColor2);
__global__ void kernelSplatInterpolateFullScan(unsigned char* output, float lambda, int size, int numpts, short xm, short ym, short xM, short yM, cudaTextureObject_t pbaTexColor, cudaTextureObject_t pbaTexLinks, cudaTextureObject_t pbaTexColor2);

// Fill Operations
__global__ void kernelFill(unsigned char* output, int size, unsigned char bg, unsigned char fg, short xm, short ym, short xM, short yM, bool ne, cudaTextureObject_t pbaTexGray);
__global__ void kernelFillHoles(unsigned char* output, int size, unsigned char bg, unsigned char fg, unsigned char fill_fg, cudaTextureObject_t pbaTexGray);