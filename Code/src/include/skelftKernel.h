//CUDA-based Skeletonization, Distance Transforms, Feature Transforms, Erosion and Dilation, and Flood Filling Toolkit
//
//(c) Alexandru Telea, Univ. of Groningen, 2011
//
//This code is adapted from "Parallel Banding Algorithm to compute exact distance transforms with the GPU", 
//T. Cao, K. Tang, A. Mohamed, T. Tan, Proc. ACM Symo. on Interactive 3D Graphics and Games (I3D), 2010, 83-90
//====================================================================================================================

// MARKER is used to mark blank pixels in the texture. Any uncolored pixels will have x = MARKER. 
// Input texture should have x = MARKER for all pixels other than sites
#define MARKER      -32768


#define TOID(x, y, size)    (__mul24((y), (size)) + (x))



// Transpose: Use the 'texObj' passed as the 3rd argument
__global__ void kernelTranspose(short2* data, int size, cudaTextureObject_t texObj)
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
        // FIX: Use 'texObj' instead of 'pbaTexColor'
        block1[threadIdx.y + i][threadIdx.x] = tex1Dfetch<short2>(texObj, id1 + idx);
        block2[threadIdx.y + i][threadIdx.x] = tex1Dfetch<short2>(texObj, id2 + idx);
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

// FloodDown: Use the 'texObj' parameter
__global__ void kernelFloodDown(short2* output, int size, int bandSize, cudaTextureObject_t texObj)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * bandSize;
    int id = TOID(tx, ty, size);

    short2 pixel1 = make_short2(MARKER, MARKER);
    for (int i = 0; i < bandSize; ++i, id += size) {
        // FIX: Use 'texObj' parameter
        short2 pixel2 = tex1Dfetch<short2>(texObj, id);
        if (pixel2.x != MARKER) pixel1 = pixel2;
        output[id] = pixel1;
    }
}


__global__ void kernelFloodUp(short2* output, int size, int bandSize, cudaTextureObject_t texObj)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = (blockIdx.y + 1) * bandSize - 1;
    int id = TOID(tx, ty, size);

    short2 pixel1 = make_short2(MARKER, MARKER);
    int dist1, dist2;

    for (int i = 0; i < bandSize; i++, id -= size)
    {
        dist1 = abs(pixel1.y - (ty - i));

        // FIX: Use 'texObj' instead of 'pbaTexColor'
        short2 pixel2 = tex1Dfetch<short2>(texObj, id);
        dist2 = abs(pixel2.y - (ty - i));

        if (dist2 < dist1) pixel1 = pixel2;

        output[id] = pixel1;
    }
}

__global__ void kernelPropagateInterband(short2* output, int size, int bandSize, cudaTextureObject_t texObj)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int inc = __mul24(bandSize, size);
    int ny, nid, nDist;

    // Top row, look backward
    int ty = __mul24(blockIdx.y, bandSize);
    int topId = TOID(tx, ty, size);
    int bottomId = TOID(tx, ty + bandSize - 1, size);

    // FIX: Use 'texObj' instead of 'pbaTexColor'
    short2 pixel = tex1Dfetch<short2>(texObj, topId);
    int myDist = abs(pixel.y - ty);

    for (nid = bottomId - inc; nid >= 0; nid -= inc)
    {
        pixel = tex1Dfetch<short2>(texObj, nid);
        if (pixel.x != MARKER)
        {
            nDist = abs(pixel.y - ty);
            if (nDist < myDist) output[topId] = pixel;
            break;
        }
    }

    // Last row, look downward
    ty = ty + bandSize - 1;
    pixel = tex1Dfetch<short2>(texObj, bottomId);
    myDist = abs(pixel.y - ty);

    for (ny = ty + 1, nid = topId + inc; ny < size; ny += bandSize, nid += inc)
    {
        pixel = tex1Dfetch<short2>(texObj, nid);

        if (pixel.x != MARKER)
        {
            nDist = abs(pixel.y - ty);
            if (nDist < myDist) output[bottomId] = pixel;
            break;
        }
    }
}

// UpdateVertical: Uses TWO objects passed as arguments
__global__ void kernelUpdateVertical(short2* output, int size, int band, int bandSize,
    cudaTextureObject_t texColor, cudaTextureObject_t texLinks)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * bandSize;

    // FIX: Use 'texLinks' parameter
    short2 top = tex1Dfetch<short2>(texLinks, TOID(tx, ty, size));
    short2 bottom = tex1Dfetch<short2>(texLinks, TOID(tx, ty + bandSize - 1, size));

    int id = TOID(tx, ty, size);
    for (int i = 0; i < bandSize; i++, id += size) {
        // FIX: Use 'texColor' parameter
        short2 pixel = tex1Dfetch<short2>(texColor, id);
        int myDist = abs(pixel.y - (ty + i));

        int dist = abs(top.y - (ty + i));
        if (dist < myDist) { myDist = dist; pixel = top; }

        dist = abs(bottom.y - (ty + i));
        if (dist < myDist) pixel = bottom;

        output[id] = pixel;
    }
}

// Input: y1 < y2
__device__ float interpoint(int x1, int y1, int x2, int y2, int x0) 
{
    float xM = float(x1 + x2) / 2.0f; 
    float yM = float(y1 + y2) / 2.0f; 
    float nx = x2 - x1; 
    float ny = y2 - y1; 

    return yM + nx * (xM - x0) / ny; 
}

__global__ void kernelProximatePoints(short2* stack, int size, int bandSize, cudaTextureObject_t texObj)
{
    int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    int ty = __mul24(blockIdx.y, bandSize);
    int id = TOID(tx, ty, size);
    int lasty = -1;
    short2 last1 = make_short2(MARKER, MARKER);
    short2 last2 = make_short2(MARKER, MARKER);
    short2 current;
    float i1, i2;

    for (int i = 0; i < bandSize; i++, id += size) {
        // FIX: Use 'texObj' instead of 'pbaTexColor'
        current = tex1Dfetch<short2>(texObj, id);

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

__global__ void kernelCreateForwardPointers(short2* output, int size, int bandSize, cudaTextureObject_t texLinks)
{
    int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    int ty = __mul24(blockIdx.y + 1, bandSize) - 1;
    int id = TOID(tx, ty, size);
    int lasty = -1, nexty;
    short2 current;

    // FIX: Use 'texLinks' instead of 'pbaTexLinks'
    current = tex1Dfetch<short2>(texLinks, id);

    if (current.x == MARKER)
        nexty = current.y;
    else
        nexty = ty;

    for (int i = 0; i < bandSize; i++, id -= size) {
        if (ty - i == nexty) {
            current = make_short2((short)lasty, tex1Dfetch<short2>(texLinks, id).y);
            output[id] = current;

            lasty = nexty;
            nexty = current.y;
        }
    }

    if (lasty != ty - bandSize + 1)
        output[id + size] = make_short2((short)lasty, MARKER);
}
__global__ void kernelMergeBands(short2* output, int size, int bandSize,
    cudaTextureObject_t texColor, cudaTextureObject_t texLinks)
{
    int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    int band1 = blockIdx.y * 2;
    int band2 = band1 + 1;
    int firsty, lasty;

    // DECLARE LOCAL VARIABLES AT THE TOP
    short2 last1 = make_short2(MARKER, MARKER);
    short2 last2 = make_short2(MARKER, MARKER);
    short2 current = make_short2(MARKER, MARKER);

    // Get the two last items of the first list
    lasty = __mul24(band2, bandSize) - 1;
    // Fetch using passed texture objects instead of global identifiers
    last2 = make_short2(tex1Dfetch<short2>(texColor, TOID(tx, lasty, size)).x,
        tex1Dfetch<short2>(texLinks, TOID(tx, lasty, size)).y);

    if (last2.x == MARKER) {
        lasty = last2.y;
        if (lasty >= 0)
            last2 = make_short2(tex1Dfetch<short2>(texColor, TOID(tx, lasty, size)).x,
                tex1Dfetch<short2>(texLinks, TOID(tx, lasty, size)).y);
        else
            last2 = make_short2(MARKER, MARKER);
    }

    if (last2.y >= 0) {
        last1 = make_short2(tex1Dfetch<short2>(texColor, TOID(tx, last2.y, size)).x,
            tex1Dfetch<short2>(texLinks, TOID(tx, last2.y, size)).y);
    }

    // Get the first item of the second band
    firsty = __mul24(band2, bandSize);
    current = make_short2(tex1Dfetch<short2>(texLinks, TOID(tx, firsty, size)).x,
        tex1Dfetch<short2>(texColor, TOID(tx, firsty, size)).x);

    if (current.y == MARKER) {
        firsty = current.x;
        if (firsty >= 0)
            current = make_short2(tex1Dfetch<short2>(texLinks, TOID(tx, firsty, size)).x,
                tex1Dfetch<short2>(texColor, TOID(tx, firsty, size)).x);
        else
            current = make_short2(MARKER, MARKER);
    }

    float i1, i2;
    int top = 0;

    while (top < 2 && current.y >= 0) {
        while (last2.y >= 0) {
            i1 = interpoint(last1.x, last2.y, last2.x, lasty, tx);
            i2 = interpoint(last2.x, lasty, current.y, firsty, tx);

            if (i1 < i2) break;

            lasty = last2.y;
            last2 = last1;
            --top;

            if (last1.y >= 0) {
                // Read from global memory 'output' and texture 'texColor'
                last1 = make_short2(tex1Dfetch<short2>(texColor, TOID(tx, last1.y, size)).x,
                    output[TOID(tx, last1.y, size)].y);
            }
            else {
                last1 = make_short2(MARKER, MARKER);
            }
        }

        output[TOID(tx, firsty, size)] = make_short2(current.x, (short)lasty);

        if (lasty >= 0)
            output[TOID(tx, lasty, size)] = make_short2((short)firsty, last2.y);

        last1 = last2;
        last2 = make_short2(current.y, (short)lasty);
        lasty = firsty;
        firsty = current.x;

        top = max(1, top + 1);

        if (firsty >= 0)
            current = make_short2(tex1Dfetch<short2>(texLinks, TOID(tx, firsty, size)).x,
                tex1Dfetch<short2>(texColor, TOID(tx, firsty, size)).x);
        else
            current = make_short2(MARKER, MARKER);
    }

    // Update head and tail pointers
    firsty = __mul24(band1, bandSize);
    lasty = __mul24(band2, bandSize);
    current = tex1Dfetch<short2>(texLinks, TOID(tx, firsty, size));

    if (current.y == MARKER && current.x < 0) {
        short2 head_ptr = tex1Dfetch<short2>(texLinks, TOID(tx, lasty, size));
        current.x = (head_ptr.y == MARKER) ? head_ptr.x : (short)lasty;
        output[TOID(tx, firsty, size)] = current;
    }

    firsty = __mul24(band1, bandSize) + bandSize - 1;
    lasty = __mul24(band2, bandSize) + bandSize - 1;
    current = tex1Dfetch<short2>(texLinks, TOID(tx, lasty, size));

    if (current.x == MARKER && current.y < 0) {
        short2 tail_ptr = tex1Dfetch<short2>(texLinks, TOID(tx, firsty, size));
        current.y = (tail_ptr.x == MARKER) ? tail_ptr.y : (short)firsty;
        output[TOID(tx, lasty, size)] = current;
    }
}

__global__ void kernelDoubleToSingleList(short2* output, int size,
    cudaTextureObject_t pbaTexColor, cudaTextureObject_t pbaTexLinks)
{
    int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    int ty = blockIdx.y;
    int id = TOID(tx, ty, size);

    output[id] = make_short2(tex1Dfetch<short2>(pbaTexColor, id).x, tex1Dfetch<short2>(pbaTexLinks, id).y);
}

__global__ void kernelColor(short2* output, int size, cudaTextureObject_t pbaTexColor)
{
    __shared__ short2 s_last1[BLOCKSIZE], s_last2[BLOCKSIZE];
    __shared__ int s_lasty[BLOCKSIZE];

    int col = threadIdx.x;
    int tid = threadIdx.y;
    int tx = __mul24(blockIdx.x, blockDim.x) + col;
    int dx, dy, lasty;
    unsigned int best, dist;
    short2 last1, last2;

    if (tid == blockDim.y - 1)
    {
        lasty = size - 1;
        last2 = tex1Dfetch<short2>(pbaTexColor, __mul24(lasty, size) + tx);

        if (last2.x == MARKER) {
            lasty = last2.y;
            last2 = tex1Dfetch<short2>(pbaTexColor, __mul24(lasty, size) + tx);
        }

        if (last2.y >= 0)
            last1 = tex1Dfetch<short2>(pbaTexColor, __mul24(last2.y, size) + tx);
        else
            last1 = make_short2(MARKER, MARKER);

        s_last1[col] = last1; s_last2[col] = last2; s_lasty[col] = lasty;
    }

    __syncthreads();

    for (int ty = size - 1 - tid; ty >= 0; ty -= blockDim.y)
    {
        last1 = s_last1[col]; last2 = s_last2[col]; lasty = s_lasty[col];

        dx = last2.x - tx; dy = lasty - ty;
        best = (unsigned int)(dx * dx + dy * dy);

        while (last2.y >= 0) {
            dx = last1.x - tx; dy = last2.y - ty;
            dist = (unsigned int)(dx * dx + dy * dy);

            if (dist > best) break;

            best = dist; lasty = last2.y; last2 = last1;

            if (last2.y >= 0)
                last1 = tex1Dfetch<short2>(pbaTexColor, __mul24(last2.y, size) + tx);
        }

        __syncthreads();

        output[TOID(tx, ty, size)] = make_short2(last2.x, (short)lasty);

        if (tid == blockDim.y - 1) {
            s_last1[col] = last1; s_last2[col] = last2; s_lasty[col] = lasty;
        }

        __syncthreads();
    }
}