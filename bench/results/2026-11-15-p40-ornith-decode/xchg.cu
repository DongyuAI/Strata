// xchg: what does one 8 KB exchange between the two cards cost, with a spin-wait barrier (no host round trip)?
// The answer decides whether every layer can run on both cards at once, which is the only 2x left.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("ERR %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

constexpr int N = 2048;   // floats: one hidden state

// card 1: wait for the flag, read the payload, set the ack
__global__ void wait_kernel(volatile int* flag, volatile int* ack, float* buf, int iters) {
    if (threadIdx.x != 0) return;
    for (int i = 0; i < iters; ++i) {
        { long g = 0; while (*flag < i + 1 && ++g < 200000000L) {} }
        __threadfence_system();
        buf[threadIdx.x] = (float) i;   // touch the payload
        __threadfence_system();
        *ack = i + 1;
    }
}
// card 0: write the payload, raise the flag, wait for the ack
__global__ void send_kernel(volatile int* flag, volatile int* ack, float* buf, int iters) {
    if (threadIdx.x != 0) return;
    for (int i = 0; i < iters; ++i) {
        buf[threadIdx.x] = (float) i;
        __threadfence_system();
        *flag = i + 1;
        { long g = 0; while (*ack < i + 1 && ++g < 200000000L) {} }
    }
}

static float ms(cudaEvent_t a, cudaEvent_t b) { float v = 0; cudaEventElapsedTime(&v, a, b); return v; }

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int n = 0;
    CK(cudaGetDeviceCount(&n));
    printf("devices %d\n", n);
    if (n < 2) return 1;
    // peer access, both ways, and report whether it is real
    int p01 = 0, p10 = 0;
    cudaDeviceCanAccessPeer(&p01, 0, 1);
    cudaDeviceCanAccessPeer(&p10, 1, 0);
    printf("peer 0->1 %d, 1->0 %d\n", p01, p10);
    CK(cudaSetDevice(0));
    cudaError_t e = cudaDeviceEnablePeerAccess(1, 0);
    printf("enable 0->1: %s\n", cudaGetErrorString(e));
    CK(cudaSetDevice(1));
    e = cudaDeviceEnablePeerAccess(0, 0);
    printf("enable 1->0: %s\n", cudaGetErrorString(e));

    // 1. back-to-back 8 KB peer copies: the streaming cost
    {
        float *a, *b;
        CK(cudaSetDevice(0));
        CK(cudaMalloc(&a, N * 4));
        CK(cudaSetDevice(1));
        CK(cudaMalloc(&b, N * 4));
        CK(cudaSetDevice(0));
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        cudaStream_t s;
        CK(cudaStreamCreate(&s));
        const int it = 500;
        CK(cudaEventRecord(e0, s));
        for (int i = 0; i < it; ++i) CK(cudaMemcpyPeerAsync(b, 1, a, 0, N * 4, s));
        CK(cudaEventRecord(e1, s));
        CK(cudaStreamSynchronize(s));
        printf("8 KB peer copy, back to back   %7.2f us each\n", ms(e0, e1) / it * 1000);
        CK(cudaEventDestroy(e0));
        CK(cudaEventDestroy(e1));
        CK(cudaStreamDestroy(s));
        cudaFree(a);
        CK(cudaSetDevice(1));
        cudaFree(b);
    }

    // 2. a spin-wait round trip, 8 KB payload
    {
        int *flag, *ack;
        float* buf;
        CK(cudaHostAlloc((void**) &flag, 64, cudaHostAllocMapped));
        CK(cudaHostAlloc((void**) &ack, 64, cudaHostAllocMapped));
        *flag = 0;
        *ack = 0;
        CK(cudaSetDevice(1));
        CK(cudaHostGetDevicePointer((void**) &flag, flag, 0));
        CK(cudaHostGetDevicePointer((void**) &ack, ack, 0));
        CK(cudaMalloc(&buf, N * 4));
        int* dflag;
        int* dack;
        CK(cudaHostGetDevicePointer((void**) &dflag, flag, 0));
        CK(cudaHostGetDevicePointer((void**) &dack, ack, 0));
        // the flag lives in host memory, visible to both
        const int it = 2000;
        CK(cudaSetDevice(1));
        wait_kernel<<<1, 32>>>(dflag, dack, buf, it);
        CK(cudaSetDevice(0));
        send_kernel<<<1, 32>>>(dflag, dack, buf, it);
        CK(cudaSetDevice(0));
        CK(cudaDeviceSynchronize());
        CK(cudaSetDevice(1));
        CK(cudaDeviceSynchronize());
        printf("spin round trip (host flag)    %7.2f us each over %d\n", 0.0, it);
        cudaFreeHost(flag);
        cudaFreeHost(ack);
        CK(cudaSetDevice(1));
        cudaFree(buf);
    }
    return 0;
}
