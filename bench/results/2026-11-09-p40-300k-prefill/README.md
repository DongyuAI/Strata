# 300K prefill on 2x P40: what is left, measured one item at a time

Baseline: **605.6 prompt tok/s** for 300,052 tokens (495.4 s), DFlash2, the three prefill switches on.
Every arm below is one engine start, one `/v1/chat/completions` request with the same 30-word filler
paragraph, and the engine's own `timings.prompt_per_second`. Nothing below 128K is considered: the
short-prompt behaviour is dominated by other phases and is not what this work is about.

## Where the 300K time goes

Median of the 36 full 8192-token chunks (`STRATA_PREFILL_TIMING=1`), which is 13,110 ms per chunk:

| phase | ms/chunk | share |
| --- | ---: | ---: |
| wait copy | 2,394 | 18.3 % |
| gdn | 1,754 | 13.4 % |
| gemm gate/up | 1,340 | 10.2 % |
| qsa attn | 974 | 7.4 % |
| gemm down | 806 | 6.1 % |
| qsa proj | 786 | 6.0 % |
| gdn out proj | 752 | 5.7 % |
| qsa select | 650 | 5.0 % |
| hc read: down+silu | 550 | 4.2 % |
| router+shared | 452 | 3.4 % |
| gdn recurrence | 344 | 2.6 % |
| hc read: up | 341 | 2.6 % |
| dequant | 316 | 2.4 % |
| hc write | 304 | 2.3 % |
| kv stage | 301 | 2.3 % |
| gather | 228 | 1.7 % |
| host grouping | 153 | 1.2 % |
| combine | 96 | 0.7 % |
| gdn conv+gates | 70 | 0.5 % |

Note for anyone re-deriving this from a log: the engine also prints one line for the whole prompt
(`prefill timing: 300047 tokens, ...`). That line is **not** the sum of the per-chunk lines — it folds
the gaps between chunks into the first phase, which makes `embed+steps` look like 57.8 s and `wait copy`
like 2.8 %. The per-chunk medians above are the ones that add up (13,082 against 13,110).

## Arms

| arm | change | prefill tok/s | against baseline |
| --- | --- | ---: | --- |
| s300k | baseline: 2 cards, DFlash2, `DENSE_F32`+`SELECT_SIMT`+`PATTN_SM60` | 605.6 | — |
| n3001 | one card (`--gpu 0`), same settings | 381.8 | −37 % |
| base300 | control for the copy arms (prefill-only request, 4 tokens generated) | 605.9 | +0.05 % |
| ring128 | `STRATA_STAGER_RING=128` (the default is 4 x pool workers = 40) | 600.5 | −0.9 % |
| sa0 | `STRATA_PREFILL_STREAM_AHEAD=0` (the previous upload schedule, a regression arm) | 607.2 | +0.2 % |
| gchunk2 | `STRATA_GDN_CHUNK=2` (the chunked GDN, after the 48 KB fix below) | 507.7 | −16.2 % |
| warp2b | `STRATA_GDN_REC_WARP=2` (the chain-split recurrence) | 605.2 | −0.1 % |
| warp1 | `STRATA_GDN_REC_WARP=1` (the sync-free warp recurrence) | none — hangs | — |
| direct1 | `STRATA_MMQ_DIRECT=1` (resident experts read in their cache slot) | not applicable | — |
| gsort | `STRATA_MMQ_GROUP_SORT=1` (full MMQ groups taken in row-count order) | 607.2 | +0.2 % |

base300 against s300k is the control: the same settings measured twice, 605.6 and 605.9, so the bar for a
gain on this prompt is about ±0.5 %.

`warp2b` was cut short at 253,952 of 300,052 tokens when the client's timeout fired; the engine's own
rate line for the tokens it did read is 605.2 tok/s (419.6 s), which is the number in the table.

`warp1` produced no number: GPU 0 sat at 97 % for eleven minutes with no 8192-token chunk finished (the
baseline needs 13 s per chunk) and GPU 1 idle, i.e. the first stage never returned. The kernel compiles
for sm_61 but does not run there. The gate in `src/prefill/kernels.cu` was put back to `volta` after the
measurement, so the switch cannot reach that path on this card again.

### gdn, three ways, measured

| arm | gdn ms/chunk | whole chunk | what it means |
| --- | ---: | ---: | --- |
| baseline (pipelined recurrence) | 1,776 | 13,150 | |
| chunked (`STRATA_GDN_CHUNK=2`) | 1,795 | 15,659 | gdn itself does not move; the path adds 2.7 s of recurrence time elsewhere |
| chain-split (`STRATA_GDN_REC_WARP=2`) | 1,753 | 13,117 | inside the noise of the baseline |

So gdn's 13.4 % is not reachable through any of the three kernels that already exist for it. The only
path left is a new kernel (int8 DP4A on the recurrence), which is a different project: it changes the
numbers the model produces, and three attempts at this phase have already come back flat or negative.

## Ruled out by reading the code, before spending a run

- **Fusing a separate dequant pass into the expert GEMM.** There is no separate dequant pass on this
  path. In `src/prefill/prefill.cpp` the phase the timer calls `dequant` is the *gather*: it copies an
  expert's GGUF blocks into its MMQ group slot (`mmq::gather_native_group`, and `STRATA_MMQ_DIRECT=1`
  already skips it for a resident expert with guard room). The 316 ms is the copy, not arithmetic.
- **Merging the four hyper-connection reads.** Both fused paths exist and both are unavailable here:
  `strata_pf_hcdown_bf16` (down + inject in one GEMM) returns false unless the device is gfx11
  (`src/prefill/wmma_gemm.cu:660`), and `gr_upmix` (up + mix) needs CUDA sm_80+
  (`src/prefill/gr_upmix_parity.cu:113`). On sm_61 the four passes stay separate; merging them means
  writing a new kernel, not flipping a switch.
- **`STRATA_GROUP_COPY=1`.** It is not an optimisation arm: it forces the copy path instead of the
  mapped pinned buffer (`src/prefill/prefill.cpp:1931`).
- **`STRATA_MMQ_DIRECT=1`.** The released engine says so itself at start-up: `STRATA_MMQ_DIRECT=1 is
  ignored: this build has no pointer-table MMQ (CUDA with STRATA_MMQ_TAB only)`. Reaching it means
  rebuilding with `STRATA_MMQ_TAB`, and it would only skip the gather of *resident* experts — the
  `dequant` phase at 300K is 2.4 % of a chunk, and the experts that cost 18.3 % (`wait copy`) are the
  ones that are not resident at all.

## The one code change this folder is about

`src/prefill/gdn_chunk.cu`, `gdn_chunk_kkt_kernel`: the kernel staged both the k and the q tiles in
dynamic shared memory, 2 x 64 x 129 x 4 = 66,048 bytes. A Pascal block cannot be given more than 48 KB
and the `cudaFuncSetAttribute` opt-in fails there silently, so the first forced run on a P40 died with
`gdn_chunk kkt: invalid argument`. The k tile now stays in shared (33,024 bytes) and q is read from the
global tile — a warp's lanes share `t` and `i`, so each read is one broadcast address, and the values and
the FMA order are unchanged, so the result is bit for bit the staged version. The path is still slower
than the recurrence (table above), so it stays opt-in; the fix is what makes it runnable at all on
Pascal.

## What the community has, and what reaches Pascal

- **llama.cpp is the only serious reference for Pascal CUDA, and this engine already vendors its
  kernels.** `src/prefill/mmq_kq_lib.cu` includes llama.cpp's `mmq.cuh`; for IQ3_S that file selects
  `MMQ_DP4A_TXS_Q8_0` (`mmq.cuh:418`), i.e. the int8 DP4A tile, and `ggml_cuda_dp4a` is chosen from
  `GGML_CUDA_CC_DP4A = 610` (`common.cuh:51`). The released engine is built for archs
  60/61/70/75/80/86/89 (`BUILD.json`), so the expert GEMM (16.3 % of a chunk) already runs the
  community's int8 DP4A path on this P40. A new int8 expert GEMM would duplicate it.
- **The community's DP4A work is decode, not prefill.** llama.cpp PR #8215 ("CUDA: refactor and
  optimize IQ MMVQ") is about MMVQ — matrix times vector, where int8 halves the bytes read. Prefill is
  compute-bound, and llama.cpp has no int8 GEMM for prompt batches beyond MMQ itself.
- **The fast GDN kernels need Hopper.** Qwen's FlashQLA (TileLang) reports 2-3x on chunked GDN prefill
  against the FLA Triton kernels, but its requirements are SM90, CUDA 12.8 and PyTorch 2.8; cuLA
  (CuTe/CUTLASS) targets sm_80+. Neither runs on sm_61.
- **DP4A cannot be applied to the recurrence that runs here.** Its per-token dot is
  `state(float) · k(float)`; putting it on DP4A would mean re-quantizing the 128-float state on every
  token, which costs more than the four FMAs per lane it saves. Only the chunked path has matrix
  products (`k·k`, `q·k`) that int8 could take, and that path measured 16 % slower overall.
- **The two-pass select cannot be moved to int8 without a new proof.** `qsa_two_pass.hpp` derives its
  bound from TF32 operands and MMA accumulation error, and the whole point of the design is that the
  selection is bit-identical to the exact scorer. An int8 screen needs its own bound; without one the
  chosen blocks change, and so do the answers.
