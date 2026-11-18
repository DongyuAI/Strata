// mb2: is the router GEMV slow, or is the pipeline around it?  Standalone, sm_61.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("ERR %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

// ---- the engine's k_gemv_f32 (q36_engine.cu), verbatim shape: warp per row, float4 per lane
__global__ void k_gemv_f32(const float* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int n_in, int n_out) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o = blockIdx.x * 8 + warp;
    if (o >= n_out) return;
    const float4* wr = (const float4*) (W + (size_t) o * n_in);
    const float4* xr = (const float4*) (x + (size_t) blockIdx.y * n_in);
    float acc = 0.f;
    for (int i = lane; i < n_in / 4; i += 32) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) y[(size_t) blockIdx.y * n_out + o] = acc;
}
// ---- the same product with RB rows per warp (RB independent rows in flight) and CO columns per thread
template <int RB, int CO>
__global__ void __launch_bounds__(256) k_gemv_f32_rb(const float* __restrict__ W, const float* __restrict__ x, float* __restrict__ y,
                                                     int n_in, int n_out) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o0 = (blockIdx.x * 8 + warp) * RB;
    const int i0 = lane * CO, i1 = n_in / 4;
    float acc[RB][CO];
#pragma unroll
    for (int r = 0; r < RB; ++r)
#pragma unroll
        for (int c = 0; c < CO; ++c) acc[r][c] = 0.f;
    float xv[CO];
    for (int i = i0; i < i1; i += 32 * CO) {
#pragma unroll
        for (int c = 0; c < CO; ++c) xv[c] = ((const float4*) x)[i + c].x + ((const float4*) x)[i + c].y +
                                             ((const float4*) x)[i + c].z + ((const float4*) x)[i + c].w;
        (void) xv;
        break;   // unused variant
    }
    (void) acc;
}

// a plain read: each thread streams float4 with a grid-stride loop, GR independent accumulators
template <int GR>
__global__ void k_stream(const float4* __restrict__ p, size_t n4, float* out) {
    float a[GR];
#pragma unroll
    for (int r = 0; r < GR; ++r) a[r] = 0.f;
    const size_t stride = (size_t) gridDim.x * blockDim.x * GR;
    for (size_t i = (size_t) blockIdx.x * blockDim.x * GR + threadIdx.x; i + stride <= n4; i += stride) {
#pragma unroll
        for (int r = 0; r < GR; ++r) {
            const float4 v = p[i + (size_t) r * blockDim.x];
            a[r] += v.x + v.y + v.z + v.w;
        }
    }
    float s = 0.f;
#pragma unroll
    for (int r = 0; r < GR; ++r) s += a[r];
    if (s == 1234.5678f) out[0] = s;
}

// dp4a throughput: how many int8 MAC-4 per cycle does the card give?
__global__ void k_dp4a(const int* __restrict__ p, int n, int* out) {
    int a0 = p[threadIdx.x], a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3;
    int s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
    const int b = p[threadIdx.x + 32];
    for (int i = 0; i < n; ++i) {
        s0 = __dp4a(a0, b, s0); s1 = __dp4a(a1, b, s1); s2 = __dp4a(a2, b, s2); s3 = __dp4a(a3, b, s3);
        s4 = __dp4a(a0, b, s4); s5 = __dp4a(a1, b, s5); s6 = __dp4a(a2, b, s6); s7 = __dp4a(a3, b, s7);
    }
    if (s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 == 7) out[0] = 1;
}

static float ms(cudaEvent_t a, cudaEvent_t b) { float v = 0; cudaEventElapsedTime(&v, a, b); return v; }

int main() {
    cudaDeviceProp pr;
    CK(cudaGetDeviceProperties(&pr, 0));
    printf("%s  SMs %d  clock %.2f GHz  mem %.0f MHz  bus %d\n", pr.name, pr.multiProcessorCount, pr.clockRate * 1e-6,
           pr.memoryClockRate * 1e-3, pr.memoryBusWidth);
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));

    // ---------- 1. router shape: 257 x 2048 f32 ----------
    {
        const int n_out = 257, n_in = 2048;
        float *W, *x, *y;
        CK(cudaMalloc(&W, (size_t) n_out * n_in * 4));
        CK(cudaMalloc(&x, (size_t) n_in * 4));
        CK(cudaMalloc(&y, (size_t) n_out * 4));
        CK(cudaMemset(W, 1, (size_t) n_out * n_in * 4));
        CK(cudaMemset(x, 1, (size_t) n_in * 4));
        const double gb = (double) n_out * n_in * 4 / 1e9;
        for (int rep = 0; rep < 3; ++rep) {
            CK(cudaEventRecord(e0));
            for (int i = 0; i < 200; ++i) k_gemv_f32<<<dim3((n_out + 7) / 8, 1), 256>>>(W, x, y, n_in, n_out);
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            const float t = ms(e0, e1) / 200;
            printf("router k_gemv_f32  33 blocks  %8.1f us  %7.1f GB/s\n", t * 1000, gb / (t * 1e-3));
        }
        // the same product as 4 independent K splits (4x the blocks, a sum kernel after)
        for (int rep = 0; rep < 2; ++rep) {
            CK(cudaEventRecord(e0));
            for (int i = 0; i < 200; ++i) {
                for (int s = 0; s < 4; ++s)
                    k_gemv_f32<<<dim3((n_out + 7) / 8, 1), 256>>>(W + (size_t) s * 512, x + (size_t) s * 512, y, 512, n_out);
            }
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            const float t = ms(e0, e1) / 200;
            printf("router 4xK-split   132 blocks %8.1f us  %7.1f GB/s (4 launches)\n", t * 1000, gb / (t * 1e-3));
        }
    }
    // ---------- 2. read bandwidth against block count ----------
    {
        const size_t n4 = (size_t) 32 * 1024 * 1024 / 16;   // 32 MiB
        float4* p;
        float* o;
        CK(cudaMalloc(&p, n4 * 16));
        CK(cudaMalloc(&o, 4));
        CK(cudaMemset(p, 1, n4 * 16));
        const double gb = (double) n4 * 16 / 1e9;
        for (int blocks : {30, 60, 120, 240, 480, 960, 1920}) {
            CK(cudaEventRecord(e0));
            for (int i = 0; i < 20; ++i) k_stream<4><<<blocks, 256>>>(p, n4, o);
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            const float t = ms(e0, e1) / 20;
            printf("stream blocks %5d (%6d thr)  %8.1f us  %7.1f GB/s\n", blocks, blocks * 256, t * 1000, gb / (t * 1e-3));
        }
    }
    // ---------- 3. dp4a throughput ----------
    {
        int* p;
        int* o;
        CK(cudaMalloc(&p, 4096));
        CK(cudaMalloc(&o, 4));
        CK(cudaMemset(p, 1, 4096));
        const int n = 200000;
        for (int blocks : {240, 960}) {
            CK(cudaEventRecord(e0));
            k_dp4a<<<blocks, 256>>>(p, n, o);
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            const double t = ms(e0, e1);
            const double dp4a = (double) blocks * 256 * n * 8;
            printf("dp4a blocks %5d  %8.2f ms  %6.1f G dp4a/s = %6.1f T int8-MAC/s\n", blocks, t, dp4a / t / 1e9,
                   dp4a * 4 / t / 1e9);
        }
    }
    return 0;
}
