# 2026-11-08 — 2× Tesla P40, Qwen3.8-Flash-Next IQ3_S, a 127K prompt

A single cold 127,042-token prompt read on a two-card layer split, used to find where a long prompt's time goes on
compute capability 6.1 and which switches shorten it. Engine 0.1.58, Windows 11, one request per arm, each arm on a
freshly started engine.

## Machine and configuration

- 2× Tesla P40 24 GB (sm_61, 30 SMs, 1,531 MHz), driver 551.78, PCIe 3.0: card 0 on x16, card 1 on x4.
- Intel i7-12700K (12 cores / 20 threads), 96 GiB RAM, Windows 11 Pro.
- Strata 0.2.12, engine 0.1.58 (the released CUDA 12 component), `--layer-split auto` → CUDA0 layers 0-24,
  CUDA1 layers 25-47; 9,155 + 7,543 expert-cache slots (16.50 + 15.17 GiB); 561 MiB VRAM free at load.
- `--pack iq3_s --native …IQ3_S-00001-of-00002.gguf --ple-gguf …00002 --expert-cache auto --prefill auto --spec 4
  --mtp F:\LLMs\mtp\rt --max-context 131072 --kv int8 --kv-resident 32768 --vision --pool-workers 10
  --vram-reserve-mib 512`, `"drafting": "dflash2"`, `STRATA_DENSE_F32=1` in `env`.
- The prompt is one repeated 30-token English sentence, `max_tokens=4`, temperature 0.

## Method

`long128.py` (in this folder) writes a run config, starts the engine with `strata.py runtime start --profile NAME`,
waits for `/health`, sends one `/v1/chat/completions` request, reads the engine's own `timings.prompt_per_second`,
and stops the engine. `STRATA_PREFILL_TIMING=1` is set for every arm, so each run also prints the per-phase GPU
timeline.

One run per arm. Four arms that differ only in settings that turned out to do nothing (k4, k5, k6, k7) read
609.2 / 610.6 / 608.2 / 609.4 tok/s and 191,623 / 191,057 / 191,851 / 191,532 ms of GPU timeline, so the
run-to-run spread at this size is about ±0.4 %. A difference smaller than that is not called a gain here.

## Results

| Arm | Change (on top of the row above) | Prompt tok/s | Wall | GPU timeline |
| --- | --- | ---: | ---: | ---: |
| k2 | `STRATA_DENSE_F32=1` only | 529.3 | 240.4 s | 219,651 ms |
| k3 | `STRATA_SELECT_SIMT=1` | 584.4 (+10.4 %) | 217.8 s | 199,856 ms |
| k4 | `STRATA_PATTN_SM60=1` | **609.2 (+15.1 % over k2)** | **208.9 s** | 191,623 ms |
| k5 | k4 + `--prefill-pipe 1` | 610.6 (no gain) | 208.5 s | 191,057 ms |
| k6 | k4 + `--prefill-pipe 3328` | 608.2 (no gain) | 209.3 s | 191,851 ms |
| k7 | k4 + `STRATA_ARENA_PIN_GIB=auto` | 609.4 (no gain) | 208.9 s | 191,532 ms |
| k8 | k4 + `STRATA_ARENA_PIN_GIB=24` | engine refused to start | — | — |

k8's log: `expert cache auto: 0.00 GiB free … no VRAM is left for the expert cache` and
`layer split, CUDA1 free memory: out of memory`. Page-locked host memory the GPU maps is charged to its WDDM shared
segment, so 24 GiB of pinned arena leaves the cards without VRAM for the cache (the case `src/core/pinned.cu`
describes). `auto` left the pinning exactly as it was (8 slices, 7 GiB, the same line as k2).

## Where the 127K read goes

`STRATA_PREFILL_TIMING=1`, the 127,037-token part of the prompt (the last 4,157 tokens are a separate line):

| Phase | k2 (529.3) | k4 (609.2) | what it is |
| --- | ---: | ---: | --- |
| gdn | 31,412 | 31,249 | the GDN projections of all 48 layers |
| gemm gate/up | 22,152 | 22,100 | the routed experts' gate/up, per 16-expert MMQ group |
| embed+steps | 16,636 | 19,781 | from the end of one chunk to the first layer's hc read of the next: the embedding, the hand-off upload and the wait for the other stage |
| hc read (4 phases) | 21,147 | 21,115 | the hyper-connection down/up/inject/mix |
| qsa attn | 23,427 | **14,463** | QSA attention over the selected cells |
| gdn out proj | 13,266 | 13,234 | |
| gemm down | 13,052 | 12,967 | the routed experts' down |
| qsa proj | 12,455 | 12,159 | |
| qsa select | 29,173 | **7,522** | block scores + top-k, the only phase that grows with position |
| router+shared | 7,605 | 7,605 | |
| gdn recurrence | 6,033 | 5,996 | |
| hc write | 5,166 | 5,054 | |
| wait copy | 4,864 | 4,802 | expert uploads |
| dequant | 3,866 | 4,012 | |
| ple | 1,678 | 1,684 | the PLE n-gram table |
| kv stage | 533 | 520 | KV streaming from RAM |

`qsa select` is the only phase whose cost rises with the position: per 8,192-token chunk it went 189 → 3,513 ms from
the first chunk to the fifteenth in k2, i.e. it is quadratic in the prompt length. At 128K it was 13.3 % of the read
before the switch and 3.9 % after; a 256K read would pay about four times the k2 figure.

## What the two switches do

- `STRATA_SELECT_SIMT=1` replaces the warp block scorer with the FP32 tiled one on the pre-sm_80 CUDA path
  (`src/kernels/cuda/qsa_select.cu`). Its summation order differs from the warp scorer's, so the scores are not
  bitwise the same and a block at the top-k cut-off can differ. 29,173 → 7,522 ms on this prompt.
- `STRATA_PATTN_SM60=1` runs the prompt path's QSA attention through the sm_6x score/value kernels
  (`src/kernels/cuda/qsa_pattn.cu`). It is the decode kernels' arithmetic in the same order, bit for bit.
  23,427 → 14,463 ms.

## What did not help

- `--prefill-pipe` (`1` and `3328`): no change to `embed+steps` (19,614 / 19,923 ms against 19,781 ms) and no change
  to the total. The 19.8 s is not removed by re-chunking the hand-off on this pair.
- `STRATA_ARENA_PIN_GIB=auto`: no change at all; `=24` does not start.

## Not measured here

- `STRATA_MMQ_WS=2` (weight-stationary expert products, extended to cc 6.1 in `src/prefill/moe_ws.cu`): the kernels
  are in the released binary for sm_61 but the run-time gate refused anything but cc 6.0, and the extension needs a
  build. No number is claimed.
- Decode: these arms read prompts; decode throughput was not the subject and is not reported.
- Quality: `STRATA_SELECT_SIMT=1` is not bitwise the warp scorer, so its effect on answers was not measured here.
  `STRATA_PATTN_SM60=1` is bit for bit.
