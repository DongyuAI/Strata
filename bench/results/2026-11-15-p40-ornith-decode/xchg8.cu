// The exchange a tensor-parallel layer needs, at the shape it would really run: one card pushes its half of the
// residual (1024 floats) into mapped host memory, flags it, and spins on the peer's flag before reading the peer's
// half.  xchg7 measured this with four launches an iteration from one host thread, where the cudaSetDevice per launch
// (134 us) dominated everything.  Here each card has its own host thread and its own stream, the whole loop is
// enqueued ahead of the GPU, and the wait is a device-side spin: this is the cost the row split would pay a layer.
//
// The timed loop is its own barrier (card 0 cannot reach iteration i+1 before card 1 has flagged i), so the two cards
// stay in lockstep and the sequence numbers can be compared with !=.  The warm-up does not spin, or one card would
// run ahead of the other and both would wait on a number that is already gone.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>
#include <thread>
#include <atomic>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::printf("FAIL %s: %s\n", #x, cudaGetErrorString(e)); std::fflush(stdout); std::exit(1); } } while (0)

struct Slot {
    volatile unsigned flag;
    volatile unsigned pad[7];
    float data[1024];   // one card's half of a 2048-wide residual
};

// ARM 1: the real exchange.  Push my half, fence, flag, spin on the peer's flag, read the peer's half.
__global__ void k_xchg(Slot* mine, Slot* peer, unsigned seq) {
    float* d = (float*) mine->data;
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) d[i] = (float) (seq + i);
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicExch((unsigned*) &mine->flag, seq);
        unsigned long long spins = 0;
        while (*(volatile unsigned*) &peer->flag != seq)
            if (++spins > 200000000ull) { mine->pad[1] = 0xDEADu; break; }
    }
    __syncthreads();
    __threadfence_system();
    const float* p = (const float*) peer->data;
    float acc = 0.f;
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) acc += p[i];
    if (threadIdx.x == 0) mine->pad[0] = __float_as_uint(acc);
}

// ARM 2: control.  The same push and flag, no spin and no read of the peer: what two cards cost with the launch and
// the write alone, so the spin and the payload read can be separated from them.
__global__ void k_push(Slot* mine, unsigned seq) {
    float* d = (float*) mine->data;
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) d[i] = (float) (seq + i);
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) atomicExch((unsigned*) &mine->flag, seq);
}

// ARM 3: the spin, no payload read: the flag round trip on its own.
__global__ void k_spin(Slot* mine, Slot* peer, unsigned seq) {
    if (threadIdx.x == 0) {
        atomicExch((unsigned*) &mine->flag, seq);
        unsigned long long spins = 0;
        while (*(volatile unsigned*) &peer->flag != seq)
            if (++spins > 200000000ull) { mine->pad[1] = 0xDEADu; break; }
    }
    __syncthreads();
    if (threadIdx.x == 0) mine->pad[0] = 0;
}

// ARM 4: read the peer's payload with no wait: the PCIe read cost alone.
__global__ void k_read(Slot* mine, Slot* peer) {
    const float* p = (const float*) peer->data;
    float acc = 0.f;
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) acc += p[i];
    if (threadIdx.x == 0) mine->pad[0] = __float_as_uint(acc);
}

struct Ctx {
    int dev = 0;
    cudaStream_t st = nullptr;
    Slot *mine = nullptr, *peer = nullptr;
    int arm = 0, N = 0;
    std::atomic<int>* ready = nullptr;
    double us = 0;
};

static void run(Ctx& c) {
    CK(cudaSetDevice(c.dev));
    for (int i = 1; i <= 10; ++i)   // warm the context and the mapped window; no spin, so no lockstep needed
        k_push<<<1, 256, 0, c.st>>>(c.mine + (i & 1), (unsigned) i);
    CK(cudaStreamSynchronize(c.st));
    c.ready->fetch_add(1);
    while (c.ready->load() < 2) { }

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < c.N; ++i) {
        Slot* s = c.mine + (i & 1);
        Slot* p = c.peer + (i & 1);
        const unsigned seq = 1u + (unsigned) i;
        if (c.arm == 1) k_xchg<<<1, 256, 0, c.st>>>(s, p, seq);
        else if (c.arm == 2) k_push<<<1, 256, 0, c.st>>>(s, seq);
        else if (c.arm == 3) k_spin<<<1, 256, 0, c.st>>>(s, p, seq);
        else k_read<<<1, 256, 0, c.st>>>(s, p);
    }
    auto t1 = std::chrono::steady_clock::now();
    CK(cudaStreamSynchronize(c.st));
    auto t2 = std::chrono::steady_clock::now();
    c.us = std::chrono::duration<double, std::micro>(t2 - t0).count() / c.N;
    std::printf("  dev%d enqueue %.1f us/iter\n", c.dev,
                std::chrono::duration<double, std::micro>(t1 - t0).count() / c.N);
    std::fflush(stdout);
}

int main(int argc, char** argv) {
    int n = 0;
    CK(cudaGetDeviceCount(&n));
    std::printf("devices %d\n", n);
    std::fflush(stdout);
    if (n < 2) return 1;
    const int N = argc > 1 ? std::atoi(argv[1]) : 200;

    Slot* h = nullptr;
    CK(cudaHostAlloc((void**) &h, 4 * sizeof(Slot), cudaHostAllocMapped | cudaHostAllocPortable));
    for (int i = 0; i < 4; ++i) { h[i].flag = 0; h[i].pad[0] = h[i].pad[1] = 0; }
    Slot* d[2] = {nullptr, nullptr};
    CK(cudaSetDevice(0)); CK(cudaHostGetDevicePointer((void**) &d[0], h, 0));
    CK(cudaSetDevice(1)); CK(cudaHostGetDevicePointer((void**) &d[1], h, 0));
    // card 0 owns slots 0,1; card 1 owns slots 2,3
    Slot* base[2] = {d[0], d[0] + 2};
    Slot* peer[2] = {d[0] + 2, d[0]};

    const char* names[4] = {"exchange (push+flag+spin+read 4 KB)", "push+flag only (control)",
                            "flag round trip only", "peer read only (4 KB)"};
    for (int arm = 1; arm <= 4; ++arm) {
        for (int i = 0; i < 4; ++i) { h[i].flag = 0; h[i].pad[1] = 0; }
        std::atomic<int> ready{0};
        Ctx c0, c1;
        c0.dev = 0; c1.dev = 1;
        CK(cudaSetDevice(0)); CK(cudaStreamCreate(&c0.st));
        CK(cudaSetDevice(1)); CK(cudaStreamCreate(&c1.st));
        c0.mine = base[0]; c0.peer = peer[0];
        c1.mine = base[1]; c1.peer = peer[1];
        c0.arm = c1.arm = arm;
        c0.N = c1.N = N;
        c0.ready = c1.ready = &ready;
        std::thread t0(run, std::ref(c0)), t1(run, std::ref(c1));
        t0.join(); t1.join();
        std::printf("%-36s %.2f us/iter  (dev0 %.2f, dev1 %.2f)%s\n", names[arm - 1],
                    (c0.us + c1.us) / 2, c0.us, c1.us,
                    h[0].pad[1] == 0xDEADu || h[2].pad[1] == 0xDEADu ? "  SPIN TIMEOUT" : "");
        std::fflush(stdout);
        CK(cudaSetDevice(0)); CK(cudaStreamDestroy(c0.st));
        CK(cudaSetDevice(1)); CK(cudaStreamDestroy(c1.st));
    }
    return 0;
}
