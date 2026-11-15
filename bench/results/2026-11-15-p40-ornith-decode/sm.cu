// sm: what does a small kernel cost on this card - the floor per launch and per block.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("ERR %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

// the shape of k_add_rms_q: 2048 floats in, 3 arrays, a 256-wide tree, 2 KB out
__global__ void __launch_bounds__(256) small(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ o,
                                            int n) {
    __shared__ float sh[256];
    const float* ar = a + (size_t) blockIdx.x * n;
    const float* br = b + (size_t) blockIdx.x * n;
    float* orr = o + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += 256) {
        const float v = ar[i] + br[i];
        orr[i] = v;
        s += v * v;
    }
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int k = 128; k > 0; k >>= 1) {
        if ((int) threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0 && sh[0] == 1234.f) o[0] = sh[0];
}

__global__ void tiny(float* o) { if (threadIdx.x == 1024) o[0] = 1.f; }

static float ms(cudaEvent_t a, cudaEvent_t b) { float v = 0; cudaEventElapsedTime(&v, a, b); return v; }

int main() {
    cudaDeviceProp pr;
    CK(cudaGetDeviceProperties(&pr, 0));
    printf("%s SMs %d\n", pr.name, pr.multiProcessorCount);
    const int n = 2048;
    float *a, *b, *o;
    CK(cudaMalloc(&a, (size_t) 64 * n * 4));
    CK(cudaMalloc(&b, (size_t) 64 * n * 4));
    CK(cudaMalloc(&o, (size_t) 64 * n * 4));
    CK(cudaMemset(a, 1, (size_t) 64 * n * 4));
    CK(cudaMemset(b, 1, (size_t) 64 * n * 4));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    const int iters = 2000;

    // 1. an empty kernel: the launch floor
    CK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) tiny<<<1, 32>>>(o);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    printf("empty 1 block 32 thr        %7.2f us/launch\n", ms(e0, e1) / iters * 1000);

    // 2. the norm shape at 1..30 blocks, 20 launches a token (the router's norm)
    for (int blocks : {1, 2, 4, 8, 16, 20}) {
        CK(cudaEventRecord(e0));
        for (int i = 0; i < iters; ++i) small<<<blocks, 256>>>(a, b, o, n);
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        const double us = ms(e0, e1) / iters * 1000;
        printf("norm %2d blocks (1 row)      %7.2f us   %6.1f GB/s\n", blocks, us, (double) blocks * n * 12 / (us * 1e3));
    }
    // 3. 20 in a row, as a layer's 20 layers of one small kernel
    CK(cudaEventRecord(e0));
    for (int r = 0; r < 100; ++r)
        for (int i = 0; i < 20; ++i) small<<<1, 256>>>(a, b, o, n);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    printf("20 x norm 1 block           %7.2f us total, %.2f us each\n", ms(e0, e1) / 100 * 1000, ms(e0, e1) / 2000 * 1000);
    return 0;
}
