#include <device_functions.h>
#include "include/skelft.h"
#include <stdio.h>

// Parameters for CUDA kernel executions
#define BLOCKX		16
#define BLOCKY		16
#define BLOCKSIZE	64
#define TILE_DIM	32
#define BLOCK_ROWS	16
#define MARKER      -32768

#define TOID(x, y, size) ((y) * (size) + (x))

/****** Topology Constants (Must be declared before kernels) *******/
#define X 0xff
#define topo_NPATTERNS  4

__constant__ const unsigned char topo_patterns[][9] = {
	{0,0,0, 0,X,0, 0,X,0},
	{0,0,0, 0,X,0, 0,0,X},
	{0,0,0, 0,X,0, 0,X,X},
	{0,0,0, 0,X,0, X,X,0}
};

__constant__ const unsigned char topo_rot[][9] = {
	{0,1,2,3,4,5,6,7,8},
	{2,5,8,1,4,7,0,3,6},
	{8,7,6,5,4,3,2,1,0},
	{6,3,0,7,4,1,8,5,2}
};

/****** Global Variables *******/
short2* pbaTextureWork;
short2* pbaTextureFT;
short* pbaTextureDT;
short2* pbaTextureWorkTopo;
short2* pbaTextureSkeletonFT;

unsigned char* pbaTextureThreshDT;
unsigned char* pbaTextureThreshSkel;
unsigned char* pbaTextureTopo;

float* pbaTexSiteParam;
int pbaTexSize;
int floodBand = 4, maurerBand = 4, colorBand = 4;

__device__ unsigned int topo_gc = 0;
__device__ bool fill_gc;

/********* Texture Object Helper ********/

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

/********* Kernels ********/

__device__ float interpoint(int x1, int y1, int x2, int y2, int x0)
{
	float xM = (float)(x1 + x2) / 2.0f;
	float yM = (float)(y1 + y2) / 2.0f;
	float nx = (float)(x2 - x1);
	float ny = (float)(y2 - y1);
	return yM + nx * (xM - (float)x0) / ny;
}

__global__ void kernelTranspose(short2* data, int size, cudaTextureObject_t texColor)
{
	__shared__ short2 block1[TILE_DIM][TILE_DIM + 1];
	__shared__ short2 block2[TILE_DIM][TILE_DIM + 1];

	int blockIdx_y = blockIdx.x;
	int blockIdx_x = blockIdx.x + blockIdx.y;
	if (blockIdx_x >= gridDim.x) return;

	int x = __mul24(blockIdx_x, TILE_DIM) + threadIdx.x;
	int y = __mul24(blockIdx_y, TILE_DIM) + threadIdx.y;
	int id1 = __mul24(y, size) + x;

	int x2 = __mul24(blockIdx_y, TILE_DIM) + threadIdx.x;
	int y2 = __mul24(blockIdx_x, TILE_DIM) + threadIdx.y;
	int id2 = __mul24(y2, size) + x2;

	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
		int idx = __mul24(i, size);
		block1[threadIdx.y + i][threadIdx.x] = tex1Dfetch<short2>(texColor, id1 + idx);
		block2[threadIdx.y + i][threadIdx.x] = tex1Dfetch<short2>(texColor, id2 + idx);
	}
	__syncthreads();

	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
		int idx = __mul24(i, size);
		short2 pixel = block1[threadIdx.x][threadIdx.y + i];
		data[id2 + idx] = make_short2(pixel.y, pixel.x);
		pixel = block2[threadIdx.x][threadIdx.y + i];
		data[id1 + idx] = make_short2(pixel.y, pixel.x);
	}
}

__global__ void kernelSiteParamInit(short2* inputVoro, int size, cudaTextureObject_t texParam) {
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	if (tx < size && ty < size) {
		int i = TOID(tx, ty, size);
		float param = tex1Dfetch<float>(texParam, i);
		short2& v = inputVoro[i];
		v.x = v.y = MARKER;
		if (param) { v.x = (short)tx; v.y = (short)ty; }
	}
}

__global__ void kernelSiteParamInitChar(short2* inputVoro, int size, cudaTextureObject_t texGray) {
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	if (tx < size && ty < size) {
		int i = TOID(tx, ty, size);
		unsigned char param = tex1Dfetch<unsigned char>(texGray, i);
		short2& v = inputVoro[i];
		v.x = v.y = MARKER;
		if (param) { v.x = (short)tx; v.y = (short)ty; }
	}
}

__global__ void kernelSiteParamInitCharInverse(short2* inputVoro, int size, cudaTextureObject_t texGray) {
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	if (tx < size && ty < size) {
		int i = TOID(tx, ty, size);
		unsigned char param = tex1Dfetch<unsigned char>(texGray, i);
		short2& v = inputVoro[i];
		v.x = v.y = MARKER;
		if (!param) { v.x = (short)tx; v.y = (short)ty; }
	}
}

__global__ void kernelFloodDown(short2* output, int size, int bandSize, cudaTextureObject_t texColor) {
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * bandSize;
	int id = TOID(tx, ty, size);
	short2 pixel1 = make_short2(MARKER, MARKER);
	for (int i = 0; i < bandSize; ++i, id += size) {
		short2 pixel2 = tex1Dfetch<short2>(texColor, id);
		if (pixel2.x != MARKER) pixel1 = pixel2;
		output[id] = pixel1;
	}
}

__global__ void kernelFloodUp(short2* output, int size, int bandSize, cudaTextureObject_t texColor) {
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = (blockIdx.y + 1) * bandSize - 1;
	int id = TOID(tx, ty, size);
	short2 pixel1 = make_short2(MARKER, MARKER);
	int dist1, dist2;
	for (int i = 0; i < bandSize; i++, id -= size) {
		dist1 = abs(pixel1.y - (ty - i));
		short2 pixel2 = tex1Dfetch<short2>(texColor, id);
		dist2 = abs(pixel2.y - (ty - i));
		if (dist2 < dist1) pixel1 = pixel2;
		output[id] = pixel1;
	}
}

__global__ void kernelPropagateInterband(short2* output, int size, int bandSize, cudaTextureObject_t texColor) {
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int inc = __mul24(bandSize, size);
	int ny, nid, nDist;
	int ty = __mul24(blockIdx.y, bandSize);
	int topId = TOID(tx, ty, size);
	int bottomId = TOID(tx, ty + bandSize - 1, size);
	short2 pixel = tex1Dfetch<short2>(texColor, topId);
	int myDist = abs(pixel.y - ty);
	for (nid = bottomId - inc; nid >= 0; nid -= inc) {
		pixel = tex1Dfetch<short2>(texColor, nid);
		if (pixel.x != MARKER) {
			nDist = abs(pixel.y - ty);
			if (nDist < myDist) output[topId] = pixel;
			break;
		}
	}
	ty = ty + bandSize - 1;
	pixel = tex1Dfetch<short2>(texColor, bottomId);
	myDist = abs(pixel.y - ty);
	for (ny = ty + 1, nid = topId + inc; ny < size; ny += bandSize, nid += inc) {
		pixel = tex1Dfetch<short2>(texColor, nid);
		if (pixel.x != MARKER) {
			nDist = abs(pixel.y - ty);
			if (nDist < myDist) output[bottomId] = pixel;
			break;
		}
	}
}

__global__ void kernelUpdateVertical(short2* output, int size, int band, int bandSize, cudaTextureObject_t texColor, cudaTextureObject_t texLinks) {
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * bandSize;
	short2 top = tex1Dfetch<short2>(texLinks, TOID(tx, ty, size));
	short2 bottom = tex1Dfetch<short2>(texLinks, TOID(tx, ty + bandSize - 1, size));
	int id = TOID(tx, ty, size);
	for (int i = 0; i < bandSize; i++, id += size) {
		short2 pixel = tex1Dfetch<short2>(texColor, id);
		int myDist = abs(pixel.y - (ty + i));
		int dist = abs(top.y - (ty + i));
		if (dist < myDist) { myDist = dist; pixel = top; }
		dist = abs(bottom.y - (ty + i));
		if (dist < myDist) pixel = bottom;
		output[id] = pixel;
	}
}

__global__ void kernelProximatePoints(short2* stack, int size, int bandSize, cudaTextureObject_t texColor) {
	int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int ty = __mul24(blockIdx.y, bandSize);
	int id = TOID(tx, ty, size);
	int lasty = -1;
	short2 last1 = make_short2(MARKER, MARKER), last2 = make_short2(MARKER, MARKER), current;
	float i1, i2;
	for (int i = 0; i < bandSize; i++, id += size) {
		current = tex1Dfetch<short2>(texColor, id);
		if (current.x != MARKER) {
			while (last2.y >= 0) {
				i1 = interpoint(last1.x, last2.y, last2.x, lasty, tx);
				i2 = interpoint(last2.x, lasty, current.x, current.y, tx);
				if (i1 < i2) break;
				lasty = last2.y; last2 = last1;
				if (last1.y >= 0) last1 = stack[TOID(tx, last1.y, size)];
				else last1 = make_short2(MARKER, MARKER);
			}
			last1 = last2; last2 = make_short2(current.x, (short)lasty); lasty = current.y;
			stack[id] = last2;
		}
	}
	if (lasty != ty + bandSize - 1)
		stack[TOID(tx, ty + bandSize - 1, size)] = make_short2(MARKER, (short)lasty);
}

__global__ void kernelCreateForwardPointers(short2* output, int size, int bandSize, cudaTextureObject_t texLinks) {
	int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int ty = __mul24(blockIdx.y + 1, bandSize) - 1;
	int id = TOID(tx, ty, size);
	int lasty = -1, nexty;
	short2 current = tex1Dfetch<short2>(texLinks, id);
	if (current.x == MARKER) nexty = current.y;
	else nexty = ty;
	for (int i = 0; i < bandSize; i++, id -= size) {
		if (ty - i == nexty) {
			current = make_short2((short)lasty, tex1Dfetch<short2>(texLinks, id).y);
			output[id] = current;
			lasty = nexty;
			nexty = current.y;
		}
	}
	if (lasty != ty - bandSize + 1) output[id + size] = make_short2((short)lasty, MARKER);
}

__global__ void kernelMergeBands(short2* output, int size, int bandSize, cudaTextureObject_t texColor, cudaTextureObject_t texLinks) {
	int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int band1 = blockIdx.y * 2;
	int band2 = band1 + 1;
	int firsty, lasty;
	short2 last1 = make_short2(MARKER, MARKER), last2, current;

	lasty = __mul24(band2, bandSize) - 1;
	last2 = make_short2(tex1Dfetch<short2>(texColor, TOID(tx, lasty, size)).x, tex1Dfetch<short2>(texLinks, TOID(tx, lasty, size)).y);
	if (last2.x == MARKER) {
		lasty = last2.y;
		if (lasty >= 0) last2 = make_short2(tex1Dfetch<short2>(texColor, TOID(tx, lasty, size)).x, tex1Dfetch<short2>(texLinks, TOID(tx, lasty, size)).y);
		else last2 = make_short2(MARKER, MARKER);
	}
	if (last2.y >= 0) last1 = make_short2(tex1Dfetch<short2>(texColor, TOID(tx, last2.y, size)).x, tex1Dfetch<short2>(texLinks, TOID(tx, last2.y, size)).y);

	firsty = __mul24(band2, bandSize);
	current = make_short2(tex1Dfetch<short2>(texLinks, TOID(tx, firsty, size)).x, tex1Dfetch<short2>(texColor, TOID(tx, firsty, size)).x);
	if (current.y == MARKER) {
		firsty = current.x;
		if (firsty >= 0) current = make_short2(tex1Dfetch<short2>(texLinks, TOID(tx, firsty, size)).x, tex1Dfetch<short2>(texColor, TOID(tx, firsty, size)).x);
		else current = make_short2(MARKER, MARKER);
	}

	float i1, i2; int top = 0;
	while (top < 2 && current.y >= 0) {
		while (last2.y >= 0) {
			i1 = interpoint(last1.x, last2.y, last2.x, lasty, tx);
			i2 = interpoint(last2.x, lasty, current.y, firsty, tx);
			if (i1 < i2) break;
			lasty = last2.y; last2 = last1; --top;
			if (last1.y >= 0) last1 = make_short2(tex1Dfetch<short2>(texColor, TOID(tx, last1.y, size)).x, output[TOID(tx, last1.y, size)].y);
			else last1 = make_short2(MARKER, MARKER);
		}
		output[TOID(tx, firsty, size)] = make_short2(current.x, (short)lasty);
		if (lasty >= 0) output[TOID(tx, lasty, size)] = make_short2((short)firsty, last2.y);
		last1 = last2; last2 = make_short2(current.y, (short)lasty); lasty = firsty;
		firsty = current.x; top = max(1, top + 1);
		if (firsty >= 0) current = make_short2(tex1Dfetch<short2>(texLinks, TOID(tx, firsty, size)).x, tex1Dfetch<short2>(texColor, TOID(tx, firsty, size)).x);
		else current = make_short2(MARKER, MARKER);
	}

	firsty = __mul24(band1, bandSize); lasty = __mul24(band2, bandSize);
	current = tex1Dfetch<short2>(texLinks, TOID(tx, firsty, size));
	if (current.y == MARKER && current.x < 0) {
		short2 h = tex1Dfetch<short2>(texLinks, TOID(tx, lasty, size));
		current.x = (h.y == MARKER) ? h.x : (short)lasty;
		output[TOID(tx, firsty, size)] = current;
	}
	firsty = __mul24(band1, bandSize) + bandSize - 1; lasty = __mul24(band2, bandSize) + bandSize - 1;
	current = tex1Dfetch<short2>(texLinks, TOID(tx, lasty, size));
	if (current.x == MARKER && current.y < 0) {
		short2 t = tex1Dfetch<short2>(texLinks, TOID(tx, firsty, size));
		current.y = (t.x == MARKER) ? t.y : (short)firsty;
		output[TOID(tx, lasty, size)] = current;
	}
}

__global__ void kernelDoubleToSingleList(short2* output, int size, cudaTextureObject_t texColor, cudaTextureObject_t texLinks) {
	int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int ty = blockIdx.y;
	int id = TOID(tx, ty, size);
	output[id] = make_short2(tex1Dfetch<short2>(texColor, id).x, tex1Dfetch<short2>(texLinks, id).y);
}

__global__ void kernelColor(short2* output, int size, cudaTextureObject_t texColor) {
	__shared__ short2 s_last1[BLOCKSIZE], s_last2[BLOCKSIZE];
	__shared__ int s_lasty[BLOCKSIZE];
	int col = threadIdx.x, tid = threadIdx.y, tx = __mul24(blockIdx.x, blockDim.x) + col;
	int dx, dy, lasty; unsigned int best, dist;
	short2 last1 = make_short2(MARKER, MARKER), last2 = make_short2(MARKER, MARKER);
	if (tid == blockDim.y - 1) {
		lasty = size - 1; last2 = tex1Dfetch<short2>(texColor, __mul24(lasty, size) + tx);
		if (last2.x == MARKER) { lasty = last2.y; last2 = tex1Dfetch<short2>(texColor, __mul24(lasty, size) + tx); }
		if (last2.y >= 0) last1 = tex1Dfetch<short2>(texColor, __mul24(last2.y, size) + tx);
		s_last1[col] = last1; s_last2[col] = last2; s_lasty[col] = lasty;
	}
	__syncthreads();
	for (int ty = size - 1 - tid; ty >= 0; ty -= blockDim.y) {
		last1 = s_last1[col]; last2 = s_last2[col]; lasty = s_lasty[col];
		dx = last2.x - tx; dy = lasty - ty; best = (unsigned int)(dx * dx + dy * dy);
		while (last2.y >= 0) {
			dx = last1.x - tx; dy = last2.y - ty; dist = (unsigned int)(dx * dx + dy * dy);
			if (dist > best) break;
			best = dist; lasty = last2.y; last2 = last1;
			if (last2.y >= 0) last1 = tex1Dfetch<short2>(texColor, __mul24(last2.y, size) + tx);
			else last1 = make_short2(MARKER, MARKER);
		}
		__syncthreads();
		output[TOID(tx, ty, size)] = make_short2(last2.x, (short)lasty);
		if (tid == blockDim.y - 1) { s_last1[col] = last1; s_last2[col] = last2; s_lasty[col] = lasty; }
		__syncthreads();
	}
}

__global__ void kernelThresholdDT(unsigned char* output, int size, float threshold2, short xm, short ym, short xM, short yM, cudaTextureObject_t texColor) {
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	if (tx > xm && ty > ym && tx < xM && ty < yM) {
		int id = TOID(tx, ty, size);
		short2 voroid = tex1Dfetch<short2>(texColor, id);
		float d2 = (float)(tx - voroid.x) * (tx - voroid.x) + (float)(ty - voroid.y) * (ty - voroid.y);
		output[id] = (d2 <= threshold2) ? 0xff : 0;
	}
}

__global__ void kernelThresholdDTInverse(unsigned char* output, int size, float threshold2, short xm, short ym, short xM, short yM, cudaTextureObject_t texColor) {
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	if (tx > xm && ty > ym && tx < xM && ty < yM) {
		int id = TOID(tx, ty, size);
		short2 voroid = tex1Dfetch<short2>(texColor, id);
		float d2 = (float)(tx - voroid.x) * (tx - voroid.x) + (float)(ty - voroid.y) * (ty - voroid.y);
		output[id] = (d2 > threshold2) ? 0xff : 0;
	}
}

__global__ void kernelDT(short* output, int size, float threshold2, short xm, short ym, short xM, short yM, cudaTextureObject_t texColor) {
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	if (tx > xm && ty > ym && tx < xM && ty < yM) {
		int id = TOID(tx, ty, size);
		short2 voroid = tex1Dfetch<short2>(texColor, id);
		float d2 = (float)(tx - voroid.x) * (tx - voroid.x) + (float)(ty - voroid.y) * (ty - voroid.y);
		output[id] = (short)sqrtf(d2);
	}
}

__global__ void kernelSkel(unsigned char* output, short xm, short ym, short xM, short yM, short size, float threshold, float length, cudaTextureObject_t texColor, cudaTextureObject_t texParam) {
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	if (tx > xm && ty > ym && tx < xM && ty < yM) {
		int id = TOID(tx, ty, size);
		short2 voroid = tex1Dfetch<short2>(texColor, id);
		float imp = tex1Dfetch<float>(texParam, TOID(voroid.x, voroid.y, size));
		short2 voroid_r = tex1Dfetch<short2>(texColor, id + 1);
		float imp_r = tex1Dfetch<float>(texParam, TOID(voroid_r.x, voroid_r.y, size));
		short2 voroid_u = tex1Dfetch<short2>(texColor, id + size);
		float imp_u = tex1Dfetch<float>(texParam, TOID(voroid_u.x, voroid_u.y, size));
		float Imp = fmaxf(fabsf(imp_r - imp), fabsf(imp_u - imp));
		Imp = fminf(Imp, fabsf(length - Imp));
		if (Imp >= threshold) output[id] = 0xff;
	}
}

__global__ void kernelTopology(unsigned char* output, short2* output_set, short xm, short ym, short xM, short yM, short size, int maxpts, cudaTextureObject_t texGray) {
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned char t[9];
	if (tx > xm && ty > ym && tx < xM - 1 && ty < yM - 1) {
		int id = TOID(tx, ty, size);
		unsigned char p = tex1Dfetch<unsigned char>(texGray, id);
		if (p) {
			unsigned char idx = 0;
			for (int j = ty - 1; j <= ty + 1; ++j) {
				int fetch_id = TOID(tx - 1, j, size);
				for (int i = 0; i <= 2; ++i, ++fetch_id, ++idx) t[idx] = tex1Dfetch<unsigned char>(texGray, fetch_id);
			}
			for (unsigned char r = 0; r < 4; ++r) {
				const unsigned char* rr = topo_rot[r];
				for (unsigned char p_idx = 0; p_idx < topo_NPATTERNS; ++p_idx) {
					const unsigned char* pat = topo_patterns[p_idx];
					unsigned char j = (p_idx == 0) ? 0 : 7;
					for (; j < 9; ++j) if (pat[j] != t[rr[j]]) break;
					if (j < 6) break;
					if (j == 9) {
						int crt_gc = atomicInc(&topo_gc, maxpts);
						output_set[crt_gc] = make_short2((short)tx, (short)ty);
						output[id] = 0xff; return;
					}
				}
			}
		}
	}
}

__global__ void kernelGatherSkelPixels(short2* output, int size, short xm, short ym, short xM, short yM, cudaTextureObject_t texGray) {
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	if (tx > xm && ty > ym && tx < xM && ty < yM) {
		int id = TOID(tx, ty, size);
		if (tex1Dfetch<unsigned char>(texGray, id)) {
			int crt_gc = atomicInc(&topo_gc, size * size);
			output[crt_gc] = make_short2((short)tx, (short)ty);
		}
	}
}

__global__ void kernelSplat(unsigned char* output, int size, int numpts, short xm, short ym, short xM, short yM, cudaTextureObject_t texSkelPts, cudaTextureObject_t texFT) {
	int offs = blockIdx.x * blockDim.x + threadIdx.x;
	if (offs < numpts) {
		short2 skel = tex1Dfetch<short2>(texSkelPts, offs);
		int tx = skel.x, ty = skel.y;
		short2 voroid = tex1Dfetch<short2>(texFT, TOID(tx, ty, size));
		float d2 = (float)(tx - voroid.x) * (tx - voroid.x) + (float)(ty - voroid.y) * (ty - voroid.y);
		short d = (short)sqrtf(d2);
		short jmin = max((int)(ty - d), (int)ym), jmax = min((int)(ty + d), (int)yM);
		for (short j = jmin; j <= jmax; ++j) {
			short w = (short)sqrtf(max(0.0f, d2 - (float)(j - ty) * (j - ty)));
			short imin = max((int)(tx - w), (int)xm), imax = min((int)(tx + w), (int)xM);
			int start_id = TOID(imin, j, size);
			for (short i = imin; i <= imax; ++i) output[start_id++] = 0xff;
		}
	}
}

__global__ void kernelGatherSkelPixelsMasked(short2* output, int size, short xm, short ym, short xM, short yM, cudaTextureObject_t texGray, cudaTextureObject_t texOriginal, cudaTextureObject_t texOpenClosed) {
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	if (tx > xm && ty > ym && tx < xM && ty < yM) {
		int id = TOID(tx, ty, size);
		if ((tex1Dfetch<unsigned char>(texGray, id) && !tex1Dfetch<unsigned char>(texOpenClosed, id)) && tex1Dfetch<unsigned char>(texOriginal, id)) {
			int crt_gc = atomicInc(&topo_gc, size * size);
			output[crt_gc] = make_short2((short)tx, (short)ty);
		}
	}
}

__global__ void kernelSiteFromSkeleton(short2* outputSites, int size, cudaTextureObject_t texGray) {
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	if (tx < size && ty < size) {
		int i = TOID(tx, ty, size);
		short2& v = outputSites[i]; v.x = v.y = MARKER;
		if (tex1Dfetch<unsigned char>(texGray, i)) { v.x = (short)tx; v.y = (short)ty; }
	}
}

__global__ void kernelSkelInterpolate(float* output, int size, cudaTextureObject_t texColor, cudaTextureObject_t texColor2) {
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	if (tx < size && ty < size) {
		int id = TOID(tx, ty, size);
		short2 vid = tex1Dfetch<short2>(texColor, id);
		float T = sqrtf((tx - vid.x) * (tx - vid.x) + (ty - vid.y) * (ty - vid.y));
		short2 vid2 = tex1Dfetch<short2>(texColor2, id);
		float D = sqrtf((tx - vid2.x) * (tx - vid2.x) + (ty - vid2.y) * (ty - vid2.y));
		output[id] = ((D) ? fminf(T / 2 / D, 0.5f) : 0.5f) + 0.5f * ((T) ? fmaxf(1 - D / T, 0.0f) : 0);
	}
}

__global__ void kernelFill(unsigned char* output, int size, unsigned char bg, unsigned char fg, short xm, short ym, short xM, short yM, bool ne, cudaTextureObject_t texGray) {
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	if (tx > xm && ty > ym && tx < xM && ty < yM) {
		int id0 = TOID(tx, ty, size);
		if (tex1Dfetch<unsigned char>(texGray, id0) == fg) {
			bool fill = false; int id = id0;
			if (ne) {
				for (short x = tx + 1; x < xM; ++x) { if (tex1Dfetch<unsigned char>(texGray, ++id) != bg) break; output[id] = fg; fill = true; }
				id = id0; for (short y = ty - 1; y > ym; --y) { if (tex1Dfetch<unsigned char>(texGray, id -= size) != bg) break; output[id] = fg; fill = true; }
			}
			else {
				for (short x = tx - 1; x > xm; --x) { if (tex1Dfetch<unsigned char>(texGray, --id) != bg) break; output[id] = fg; fill = true; }
				id = id0; for (short y = ty + 1; y < yM; ++y) { if (tex1Dfetch<unsigned char>(texGray, id += size) != bg) break; output[id] = fg; fill = true; }
			}
			if (fill) fill_gc = true;
		}
	}
}

__global__ void kernelFillHoles(unsigned char* output, int size, unsigned char bg, unsigned char fg, unsigned char fill_fg, cudaTextureObject_t texGray) {
	int tx = blockIdx.x * blockDim.x + threadIdx.x, ty = blockIdx.y * blockDim.y + threadIdx.y;
	if (tx >= 0 && ty >= 0 && tx < size && ty < size) {
		int id = TOID(tx, ty, size); unsigned char val = tex1Dfetch<unsigned char>(texGray, id);
		if (val == fill_fg) output[id] = bg; else if (val == bg) output[id] = fg;
	}
}

/********* Host Functions ********/

void skelft2DInitialization(int maxTexSize) {
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

void skelft2DDeinitialization() {
	cudaFree(pbaTextureWork); cudaFree(pbaTextureFT); cudaFree(pbaTextureDT);
	cudaFree(pbaTextureWorkTopo); cudaFree(pbaTextureSkeletonFT);
	cudaFree(pbaTextureThreshDT); cudaFree(pbaTextureThreshSkel); cudaFree(pbaTextureTopo);
	cudaFree(pbaTexSiteParam);
}

void skelft2DParams(int floodBand_, int maurerBand_, int colorBand_) {
	floodBand = floodBand_; maurerBand = maurerBand_; colorBand = colorBand_;
}

void skelft2DInitializeInput(float* sites, int size) {
	pbaTexSize = size;
	size_t bytes = pbaTexSize * pbaTexSize * sizeof(float);
	cudaMemcpy(pbaTexSiteParam, sites, bytes, cudaMemcpyHostToDevice);
	cudaTextureObject_t texObj = bindTextureObject(pbaTexSiteParam, bytes);
	dim3 block(BLOCKX, BLOCKY), grid(pbaTexSize / block.x, pbaTexSize / block.y);
	kernelSiteParamInit << <grid, block >> > (pbaTextureWork, pbaTexSize, texObj);
	cudaDestroyTextureObject(texObj);
}

void pba2DTranspose(short2* texture) {
	dim3 block(TILE_DIM, BLOCK_ROWS), grid(pbaTexSize / TILE_DIM, pbaTexSize / TILE_DIM);
	cudaTextureObject_t texObj = bindTextureObject(texture, pbaTexSize * pbaTexSize * sizeof(short2));
	kernelTranspose << <grid, block >> > (texture, pbaTexSize, texObj);
	cudaDestroyTextureObject(texObj);
}

void pba2DPhase1(int m1, short xm, short ym, short xM, short yM) {
	dim3 block(BLOCKSIZE); dim3 grid(pbaTexSize / block.x, m1);
	size_t sz = pbaTexSize * pbaTexSize * sizeof(short2);
	cudaTextureObject_t texWork = bindTextureObject(pbaTextureWork, sz);
	cudaTextureObject_t texFT = bindTextureObject(pbaTextureFT, sz);
	kernelFloodDown << <grid, block >> > (pbaTextureFT, pbaTexSize, pbaTexSize / m1, texWork);
	kernelFloodUp << <grid, block >> > (pbaTextureFT, pbaTexSize, pbaTexSize / m1, texFT);
	kernelPropagateInterband << <grid, block >> > (pbaTextureWork, pbaTexSize, pbaTexSize / m1, texFT);
	kernelUpdateVertical << <grid, block >> > (pbaTextureFT, pbaTexSize, m1, pbaTexSize / m1, texFT, texWork);
	cudaDestroyTextureObject(texWork); cudaDestroyTextureObject(texFT);
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
		kernelMergeBands << <grid, block >> > (pbaTextureWork, pbaTexSize, pbaTexSize / noBand, texColor, texLinks);
	}
	grid = dim3(pbaTexSize / block.x, pbaTexSize);
	kernelDoubleToSingleList << <grid, block >> > (pbaTextureWork, pbaTexSize, texColor, texLinks);
	cudaDestroyTextureObject(texColor); cudaDestroyTextureObject(texLinks);
}

void pba2DPhase3(int m3) {
	dim3 block(BLOCKSIZE / m3, m3), grid(pbaTexSize / block.x);
	cudaTextureObject_t texColor = bindTextureObject(pbaTextureWork, pbaTexSize * pbaTexSize * sizeof(short2));
	kernelColor << <grid, block >> > (pbaTextureFT, pbaTexSize, texColor);
	cudaDestroyTextureObject(texColor);
}

void skel2DFTCompute(short xm, short ym, short xM, short yM, int floodBand, int maurerBand, int colorBand) {
	pba2DPhase1(floodBand, xm, ym, xM, yM);
	pba2DTranspose(pbaTextureFT);
	pba2DPhase2(maurerBand);
	pba2DPhase3(colorBand);
	pba2DTranspose(pbaTextureFT);
}

void skelft2DFT(short* output, float* siteParam, short xm, short ym, short xM, short yM, int size) {
	skelft2DInitializeInput(siteParam, size);
	skel2DFTCompute(xm, ym, xM, yM, floodBand, maurerBand, colorBand);
	if (output) cudaMemcpy(output, pbaTextureFT, size * size * sizeof(short2), cudaMemcpyDeviceToHost);
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
	cudaDestroyTextureObject(texColor); cudaDestroyTextureObject(texParam);
	if (outputSkel) cudaMemcpy(outputSkel, pbaTextureThreshSkel, pbaTexSize * pbaTexSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
}

int skelft2DInflate(unsigned char* result, short xm, short ym, short xM, short yM) {
	dim3 block(BLOCKX, BLOCKY), grid(pbaTexSize / block.x, pbaTexSize / block.y);
	unsigned int num_pts = 0; cudaMemcpyToSymbol(topo_gc, &num_pts, sizeof(unsigned int));
	cudaTextureObject_t texGray = bindTextureObject(pbaTextureThreshSkel, pbaTexSize * pbaTexSize * sizeof(unsigned char));
	kernelGatherSkelPixels << <grid, block >> > (pbaTextureWorkTopo, pbaTexSize, xm, ym, xM, yM, texGray);
	cudaDestroyTextureObject(texGray);
	cudaMemcpyFromSymbol(&num_pts, topo_gc, sizeof(unsigned int));
	cudaMemset(pbaTextureThreshDT, 0, pbaTexSize * pbaTexSize * sizeof(unsigned char));
	block = dim3(BLOCKX * BLOCKY); grid = dim3((num_pts + block.x - 1) / block.x);
	cudaTextureObject_t texSkelPts = bindTextureObject(pbaTextureWorkTopo, pbaTexSize * pbaTexSize * sizeof(short2));
	cudaTextureObject_t texFT = bindTextureObject(pbaTextureFT, pbaTexSize * pbaTexSize * sizeof(short2));
	kernelSplat << <grid, block >> > (pbaTextureThreshDT, pbaTexSize, num_pts, xm, ym, xM, yM, texSkelPts, texFT);
	cudaDestroyTextureObject(texSkelPts); cudaDestroyTextureObject(texFT);
	if (result) cudaMemcpy(result, pbaTextureThreshDT, pbaTexSize * pbaTexSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	return num_pts;
}

void skelft2DTopology(unsigned char* outputTopo, int* npts, short2* outputPoints, short xm, short ym, short xM, short yM) {
	int maxpts = (npts) ? *npts : pbaTexSize * pbaTexSize;
	dim3 block(BLOCKX, BLOCKY), grid(pbaTexSize / block.x, pbaTexSize / block.y);
	cudaTextureObject_t texObj = bindTextureObject(pbaTextureThreshSkel, pbaTexSize * pbaTexSize * sizeof(unsigned char));
	cudaMemset(pbaTextureTopo, 0, sizeof(unsigned char) * pbaTexSize * pbaTexSize);
	unsigned int num_pts = 0; cudaMemcpyToSymbol(topo_gc, &num_pts, sizeof(unsigned int));
	kernelTopology << <grid, block >> > (pbaTextureTopo, pbaTextureWorkTopo, xm, ym, xM, yM, (short)pbaTexSize, maxpts + 1, texObj);
	cudaDestroyTextureObject(texObj);
	if (outputPoints && maxpts) {
		cudaMemcpyFromSymbol(&num_pts, topo_gc, sizeof(unsigned int));
		if (npts && num_pts) cudaMemcpy(outputPoints, pbaTextureWorkTopo, min(num_pts, (unsigned int)maxpts) * sizeof(short2), cudaMemcpyDeviceToHost);
		if (npts) *npts = (int)num_pts;
	}
	if (outputTopo) cudaMemcpy(outputTopo, pbaTextureTopo, pbaTexSize * pbaTexSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
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
	cudaDestroyTextureObject(texColor); cudaDestroyTextureObject(texColor2);
	if (outputSkelDT) cudaMemcpy(outputSkelDT, pbaTextureWork, pbaTexSize * pbaTexSize * sizeof(float), cudaMemcpyDeviceToHost);
}

int skelft2DFill(unsigned char* outputFill, short sx, short sy, short xm, short ym, short xM, short yM, unsigned char fill_value) {
	dim3 block(BLOCKX, BLOCKY), grid(pbaTexSize / block.x, pbaTexSize / block.y);
	unsigned char background;
	cudaMemcpy(&background, pbaTextureThreshDT + (sy * pbaTexSize + sx), sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaMemset(pbaTextureThreshDT + (sy * pbaTexSize + sx), fill_value, sizeof(unsigned char));
	int iter = 0; bool xy = true;
	for (;; ++iter, xy = !xy) {
		bool filled = false; cudaMemcpyToSymbol(fill_gc, &filled, sizeof(bool));
		cudaTextureObject_t texGray = bindTextureObject(pbaTextureThreshDT, pbaTexSize * pbaTexSize * sizeof(unsigned char));
		kernelFill << <grid, block >> > (pbaTextureThreshDT, pbaTexSize, background, fill_value, xm, ym, xM, yM, xy, texGray);
		cudaDestroyTextureObject(texGray); cudaMemcpyFromSymbol(&filled, fill_gc, sizeof(bool));
		if (!filled) break;
	}
	if (outputFill) cudaMemcpy(outputFill, pbaTextureThreshDT, pbaTexSize * pbaTexSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	return iter;
}

int skelft2DFillHoles(unsigned char* outputFill, short sx, short sy, unsigned char foreground) {
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