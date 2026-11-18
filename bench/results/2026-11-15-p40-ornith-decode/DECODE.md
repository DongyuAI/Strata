# 2026-11-15 — 2× Tesla P40, Ornith-1.5-35B-A3B UD-IQ4_XS, decode at 64K

Where a decode token's time goes on two Pascal cards, one change that came out of it (+9.3 % decode), and the
ceilings measured along the way. Every number below was measured on this machine; the static reasoning is marked as
such. Engine 0.1.58 built for sm_61 from this tree, one engine start per arm, 32 greedy tokens per arm.

## Machine and configuration

- 2× Tesla P40 24 GB (sm_61, 30 SMs, 1,531 MHz, memory 3,615 MHz), WDDM, power limit 125 W, driver 551.78.
- Intel i7-12700K, 96 GiB RAM, Windows 11 Pro, CUDA 12.4.
- Model `Ornith-1.5-35B-A3B-Abliterated-MTPv2-UD-IQ4_XS.gguf` (18,684,044,960 bytes): gate/up experts IQ3_S,
  down experts IQ4_XS, attention/GDN/shared-expert dense weights Q8_0, router F32, output head Q6_K.
- Geometry: n_embd 2048, 16 q heads, 2 KV heads, head_dim 256, 256 experts of 8 used, expert n_ff 512, 40 layers
  (10 attention, 30 GDN) plus an MTP block, vocab 248,320.
- `--max-context 65536`, layer split 20/20 (`STRATA_Q36_SPLIT=1 STRATA_Q36_SPLIT_LAYERS=20`), card-local decode
  graphs (`STRATA_Q36_SPLIT_GRAPHS=1`), int8 KV (`kv8_on`), packed Pascal Q8_0 decode GEMV (`STRATA_Q8_SM60=1`),
  `STRATA_Q36_DECODE_TILE=1`, `STRATA_Q36_GDN_TILED=1`, `STRATA_Q36_PRE_I8=1`, `STRATA_Q36_CHUNK=1024`, MTP off.
- Prompt: 2,000 repetitions of one 30-token English sentence = 60,165 tokens, `max_tokens=32`, temperature 0.
  The same prompt every arm, same `prompt_sha256`.

## Method

A small probe script outside this tree (it holds this machine's paths) starts the engine through the control CLI with a
fixed environment, waits for `/health`, sends one `/v1/chat/completions` request with `temperature 0`,
`presence_penalty 0`, `repetition_penalty 1.0`, `max_tokens 32` and reads the engine's own `timings`. A cold start
plus the 60,165-token prefill is about 115 s of wall clock; the 32 decode tokens are 0.7 s of it. Repeat runs of one
arm spread by about 2 % on this harness, so a difference smaller than that is not called a gain, and the 0.2 tok/s
between arms c and d is noise.

## Result

| Arm | Change | decode tok/s | prefill tok/s |
| --- | --- | ---: | ---: |
| a | starting point (same harness, same prompt, before the change below) | 42.8 | 1,690 |
| b | IQ4_XS down product, 16 lanes a row, 4 rows a half-warp | **46.8 (+9.3 %)** | 1,691.6 |
| c | + packed Q8_0 GEMV 2 chunks a step (`q8_sm60.cuh`) | 46.6 (no gain) | 1,690.9 |
| d | + attention carveout at maximum, and K/V in separate buffers | 46.1 (no gain, reverted) | 1,685.8 |

Arm b is the one change that reached the token rate. It is bitwise equal to the old kernel: `native_expert_bench`
on layer 0 and layer 3 of the real file reports `bitwise equal`, hash `9bcde71ce8465b09`.

## The change

`native_down_multi_kernel` (mode 8) ran the down product as one warp a row. At n_ff 512 an IQ4_XS row is 16 dots,
so 16 of every warp's 32 lanes idled and a warp issued one 16-byte load a row. The new branch puts 16 lanes on a
row and 4 rows in a half-warp, all four loads issued before any reduction. The 16-lane butterfly it uses is the
32-lane one with zeros in the upper half, which is why the sums are unchanged.

`native_expert_bench`, 8 groups of 1 token, 200 iterations, layer 0:

| phase | before | after |
| --- | ---: | ---: |
| gate/up (IQ3_S) | 57.6 µs, 125 GB/s | 56.6 µs, 127 GB/s |
| down (IQ4_XS) | 74.1 µs, 60 GB/s | 34.7 µs, **133 GB/s** |
| whole expert step | 135.8 µs | **92.4 µs** |

## Ceilings measured on this card

- **A pure linear read** of 32 MiB with 4 independent accumulators: 232 GB/s at 30 blocks, 267 at 240, 305 at 480.
  This is the most any access pattern gave.
- **The same read in the shape of a row-major GEMV** (8,192 rows of 2,048 bytes, one row a warp): 225–231 GB/s at
  every grid from 120 to 480 blocks (`q8d.cu`, `read-only` rows).
- **The production packed Q8_0 GEMV** on the same shape: 202 GB/s, 214 with two chunks a step — **94 % of that
  shape's read-only ceiling**, so this kernel has no headroom left.
- **A kernel launch**: an empty kernel costs 7.41 µs; a 2048-wide RMS norm costs 6.26–6.59 µs at any block count
  from 1 to 20 (`sm.cu`). The floor is the launch, not the work. Graphs take most of it away: with the same
  configuration and card-local decode graphs off the same token read 41.6 tok/s against 46.8 with them (that arm
  also had the head timing on, which adds one synchronisation a token, so it is an upper bound on the gain).
- **Between the cards**: `cudaDeviceCanAccessPeer` is 0 in both directions and `cudaDeviceEnablePeerAccess` fails
  with *peer access is not supported between these two devices*. An 8 KB `cudaMemcpyPeerAsync` back to back costs
  **40.16 µs** (`xchg.cu`).

## Where the time goes

Per-token traffic, from the file's tensor types and the configuration above: weights 2.06 GB (both cards' halves of
the dense weights and the 8 active experts), KV 0.69 GB (10 attention layers × 60,165 cells × 1,152 B), output head
0.42 GB. **3.17 GB in 21.4 ms = 148 GB/s.**

The per-layer phase marks (`STRATA_Q36_DECODE_PROFILE`) sum to 13.0 ms on card 1 and 11.7 ms on card 0:

| phase | card 1 (ms) | card 0 (ms) |
| --- | ---: | ---: |
| projections (qkv / q / k / v + gate) | 3.29 | 2.93 |
| attention core (the 5 attention layers) | 2.82 | 2.75 |
| GDN core (the 15 GDN layers) | 1.58 | 1.42 |
| routed experts | 2.37 | 2.12 |
| router (norm + product + top-k + grouping) | 1.40 | 1.18 |
| shared expert | 1.06 | 0.86 |
| combine | 0.51 | 0.42 |

The router's 1.18–1.40 ms is mostly launch floor, not bandwidth: the product itself is 2.1 MB and measures 9.5 µs
in isolation (`mb2.cu`), while the 2048-wide norm and the single-block top-k that bracket it each cost about 6.4 µs
of floor, and the three are a serial chain twenty times over.

## What this says about 120 tok/s

The target is 120 tok/s single stream. Nothing below is a statement about what the card can do; it is a statement
about what this configuration moves and where the next bytes come from.

At the traffic measured above, 3.17 GB in 21.4 ms is 148 GB/s, against a 225–231 GB/s read ceiling for that shape.
The 0.8 ms of the 21.4 that is not bytes is launch floor. So the bytes are the whole cost, and the work left is to
move fewer of them and to stop paying for both cards in series:

- **KV at q4_0.** 0.69 GB of the 3.17 is the KV read at 60K cells. Halving its width to q4_0 takes 0.35 GB out
  without touching the model file — the engine currently keeps f32 and q8 only. This is the next change.
- **The output head on both cards.** 0.42 GB of Q6_K, 2.8 ms on one card. Both cards reading half of it makes it
  1.4 ms, and the reduction is 248,320 values.
- **Both cards on every layer.** Today the layer split makes a token the sum of the two cards' times. Running each
  layer's rows on both cards makes it the maximum instead. This is the largest single item and it is an engine change,
  not a setting. What it costs is measured below.
- **Fewer weight bytes.** 2.06 GB a token for 8 of 256 experts: 2-bit experts and lower-precision dense weights are
  the remaining lever, and those do change the quantization.

The state above is what is committed here. It is a checkpoint, not the end of the work.

## What the two cards cost each other

Two numbers decide whether running every layer's rows on both cards is worth building, and both are now measured.

**Each card's own GPU time.** `STRATA_Q36_CARD_TIMING` puts events around each card's forward, inside the split, so
the handoff is outside them: card 0 9.47 ms, card 1 11.62 ms for 20 layers each. The sum, 21.09 ms, is the measured
token interval 21.30 ms to within 0.21 ms. **The two cards do not overlap at all today**; a token is their sum. (The
per-layer phase marks above sum to 13.0 and 11.7 ms because each card's marks include the time it spends waiting for
the other, which is why they are not the number to project from.)

**The exchange a row split needs.** One card pushes its half of the residual — 1024 floats — into mapped host memory,
fences, flags it, spins on the peer's flag, and reads the peer's half. Both cards run their own stream on their own
host thread with the whole loop enqueued ahead, so the host is out of the loop and the wait is a device-side spin
(`xchg8.cu`). Over 200 iterations, averaged over the two cards:

| step | µs a layer |
| --- | ---: |
| push 4 KB + flag only (control: launch and write, no wait, no read) | 16.2 |
| flag round trip only (no payload) | 16.7 |
| peer read only, 4 KB (no wait) | 17.4 |
| **the whole exchange** | **23.8** |

The launch floor is 7.4 µs of that, so a 4 KB write to host memory costs about 9 µs and a 4 KB read about 10 µs, and
the flag round trip fits inside the payload movement. 23.8 µs a layer is 80 exchanges a token (one after attention,
one after the experts) at 1.9 ms, against 9.47 ms of card 0 work that a row split would halve. That is the trade the
row split makes, and it is affordable. There is no peer access between these cards in either direction, so mapped host
memory is the only path.

Note that the 40.16 µs figure quoted in `xchg.cu` is `cudaMemcpyPeerAsync`, which falls back to host memory when there
is no peer access and is not the path an exchange would take; 23.8 µs is the path it would.

## Files

- `q8d.cu` — the packed Q8_0 GEMV against the same reads with the dot removed, at several grids. `nvcc -O3
  -arch=sm_61 -o q8d.exe q8d.cu`.
- `mb2.cu` — peak read bandwidth against block count, the router-shaped GEMV on its own, and the dp4a rate.
- `sm.cu` — the launch floor: an empty kernel, and the 2048-wide norm at 1 to 20 blocks.
- `xchg.cu` — peer access between the two cards, and the cost of an 8 KB exchange.
- `xchg8.cu` — the exchange a row split would run: two host threads, two streams, the loop enqueued ahead, the wait a
  device-side spin on a mapped flag. Prints the exchange, the flag alone, the payload alone and the launch control.
- `native_expert_bench` (`src/kernels/native_expert_bench.cpp`, `--target native_expert_bench`) — the expert step on
  real GGUF rows, mode against mode, bitwise and timed. The geometry now comes from the tensors, so it runs on a
  model that is not Flash-Next.
