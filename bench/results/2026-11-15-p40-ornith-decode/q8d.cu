// q8d: is the packed Q8_0 GEMV memory-bound or issue-bound?  Same reads, the dp4a work removed.
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("ERR %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

struct Q81Block { int8_t qs[32]; half2 ds; };
constexpr int THREADS = 256;

inline std::size_t smem_bytes(int n_in) { return std::size_t(n_in) + std::size_t(n_in / 32) * sizeof(float); }

// the production shape: 1 row a warp, CH chunks of 16 bytes a lane a step
template<int CH, bool DP>
__global__ void __launch_bounds__(THREADS)
qk(const int8_t* __restrict__ qs, const half* __restrict__ dpl, const Q81Block* __restrict__ x, float* __restrict__ y,
   int n_in, int n_out) {
    extern __shared__ int4 sm[];
    int8_t* sx = reinterpret_cast<int8_t*>(sm);
    float* sd = reinterpret_cast<float*>(sx + std::size_t(n_in));
    const int nb = n_in / 32;
    if (DP)
        for (int b = int(threadIdx.x); b < nb; b += THREADS) {
            const int* src = reinterpret_cast<const int*>(x[b].qs);
            int* dst = reinterpret_cast<int*>(sx + std::size_t(b) * 32);
#pragma unroll
            for (int i = 0; i < 8; ++i) dst[i] = src[i];
            sd[b] = __low2float(x[b].ds);
        }
    __syncthreads();
    const int lane = int(threadIdx.x) & 31, rloc = int(threadIdx.x) >> 5;
    const int nch = n_in / 16;
    const int groups = (n_out + 7) / 8;
    for (int g = int(blockIdx.x); g < groups; g += int(gridDim.x)) {
        const int row = g * 8 + rloc;
        const bool ok = row < n_out;
        const int4* w = reinterpret_cast<const int4*>(qs + std::size_t(ok ? row : 0) * n_in);
        const half* d = dpl + std::size_t(ok ? row : 0) * nb;
        float acc = 0.f;
        if (DP) {
            const int4* sxv = reinterpret_cast<const int4*>(sx);
            for (int c = lane; c < nch; c += 32 * CH) {
                int4 wv[CH];
#pragma unroll
                for (int q = 0; q < CH; ++q) wv[q] = __ldg(w + c + q * 32);
#pragma unroll
                for (int q = 0; q < CH; ++q) {
                    const int4 xv = sxv[c + q * 32];
                    int s = __dp4a(wv[q].x, xv.x, 0);
                    s = __dp4a(wv[q].y, xv.y, s);
                    s = __dp4a(wv[q].z, xv.z, s);
                    s = __dp4a(wv[q].w, xv.w, s);
                    acc += __half2float(d[(c + q * 32) >> 1]) * sd[(c + q * 32) >> 1] * float(s);
                }
            }
        } else {   // the same weight reads, no activation, no dp4a: the pure read this shape can do
            for (int c = lane; c < nch; c += 32 * CH) {
                int4 wv[CH];
#pragma unroll
                for (int q = 0; q < CH; ++q) wv[q] = __ldg(w + c + q * 32);
#pragma unroll
                for (int q = 0; q < CH; ++q) acc += float(wv[q].x + wv[q].y + wv[q].z + wv[q].w);
            }
        }
        if (ok) {
            float v = acc;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) y[row] = v;
        }
    }
}

static float ms(cudaEvent_t a, cudaEvent_t b) { float v = 0; cudaEventElapsedTime(&v, a, b); return v; }

int main(int argc, char** argv) {
    const int n_out = argc > 1 ? atoi(argv[1]) : 8192;
    const int n_in = argc > 2 ? atoi(argv[2]) : 2048;
    cudaDeviceProp pr;
    CK(cudaGetDeviceProperties(&pr, 0));
    printf("%s SMs %d  %d x %d (%.2f MB)\n", pr.name, pr.multiProcessorCount, n_out, n_in, (double) n_out * n_in / 1e6);
    int8_t* qs; half* d; Q81Block* x; float* y;
    CK(cudaMalloc(&qs, (size_t) n_out * n_in));
    CK(cudaMalloc(&d, (size_t) n_out * (n_in / 32) * 2));
    CK(cudaMalloc(&x, (size_t) (n_in / 32) * sizeof(Q81Block)));
    CK(cudaMalloc(&y, (size_t) n_out * 4));
    CK(cudaMemset(qs, 1, (size_t) n_out * n_in));
    CK(cudaMemset(d, 0x3c, (size_t) n_out * (n_in / 32) * 2));
    CK(cudaMemset(x, 1, (size_t) (n_in / 32) * sizeof(Q81Block)));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    const double gb = (double) n_out * n_in / 1e9;
    const int groups = (n_out + 7) / 8;
    for (int bps : {4, 8, 16}) {
        const int grid = groups < pr.multiProcessorCount * bps ? groups : pr.multiProcessorCount * bps;
        for (int dp = 0; dp < 2; ++dp)
            for (int ch : {1, 2, 4}) {
                for (int rep = 0; rep < 2; ++rep) {
                    CK(cudaEventRecord(e0));
                    for (int i = 0; i < 200; ++i) {
                        if (dp && ch == 1) qk<1, true><<<grid, THREADS, smem_bytes(n_in)>>>(qs, d, x, y, n_in, n_out);
                        else if (dp && ch == 2) qk<2, true><<<grid, THREADS, smem_bytes(n_in)>>>(qs, d, x, y, n_in, n_out);
                        else if (dp) qk<4, true><<<grid, THREADS, smem_bytes(n_in)>>>(qs, d, x, y, n_in, n_out);
                        else if (ch == 1) qk<1, false><<<grid, THREADS, 0>>>(qs, d, x, y, n_in, n_out);
                        else if (ch == 2) qk<2, false><<<grid, THREADS, 0>>>(qs, d, x, y, n_in, n_out);
                        else qk<4, false><<<grid, THREADS, 0>>>(qs, d, x, y, n_in, n_out);
                    }
                    CK(cudaEventRecord(e1));
                    CK(cudaEventSynchronize(e1));
                    const float t = ms(e0, e1) / 200;
                    if (rep == 1)
                        printf("%-9s ch %d bps %2d grid %5d  %8.1f us  %7.1f GB/s\n", dp ? "gemv" : "read-only", ch, bps,
                               grid, t * 1000, gb / (t * 1e-3));
                }
            }
    }
    return 0;
}
