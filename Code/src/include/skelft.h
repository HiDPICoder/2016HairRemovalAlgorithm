#pragma once
#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

// Add all your function prototypes here
int skelft2DSize(int nx, int ny);
float skelft2DMakeBoundary(unsigned char* input, int xm, int ym, int xM, int yM, float* param, int size, short iso = 1, bool thr_upper = true);
void skelft2DInitialization(int textureSize);
void skelft2DDeinitialization();
void skelft2DParams(int phase1Band, int phase2Band, int phase3Band);
void skelft2DFT(short* output, float* siteParam, short xm, short ym, short xM, short yM, int size);
void skelft2DSkeleton(unsigned char* output, float length, float threshold, short xm, short ym, short xM, short yM);
void skelft2DDT(short* output, float threshold, short xm, short ym, short xM, short yM);
void skelft2DTopology(unsigned char* topo, int* npts, short2* points, short xm, short ym, short xM, short yM);
void skel2DSkeletonDT(float* outputSkelDT, short xm, short ym, short xM, short yM);
int skelft2DFill(unsigned char* outputFill, short seedx, short seedy, short xm, short ym, short xM, short yM, unsigned char foreground);
int skelft2DFillHoles(unsigned char* outputFill, short x, short y, unsigned char fill_val);

#ifdef __cplusplus
}
#endif