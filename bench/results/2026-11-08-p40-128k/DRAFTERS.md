# MTP against DFlash2, 128K to 300K

A second drafter (the small model that guesses the next words so the big one can check several at once)
was added to this machine on 2026-11-09: the MTP layer, from the original Qwen BF16 checkpoint. Until then
the engine had none of it, and `drafting` was absent from the run config, which means `auto` — and `auto`
picks DFlash2 on this model. So every number this folder reported before 2026-11-09 as "MTP" was DFlash2.

This file answers one question: **which of the two is faster here at long context, and does the answer
change with the length of the prompt?** Only 128K and above is considered; below that the prompt path is
dominated by other phases and is not what this machine is used for.

## The two arms

| | MTP | DFlash2 |
| --- | --- | --- |
| what it is | the model's own draft layer, one extra block from the Qwen checkpoint | a separate 1.2 GB draft model (`incoai/Qwen3.8-Flash-Next-DFlash2`) |
| run config | `"drafting": "mtp"`, `--mtp F:\LLMs\mtp\rt` | `"drafting": "dflash2"` |
| draft window | `--spec 4` | `--spec 6 --draft-conf 0.35` (the pair `strata_control/drafting.py` writes) |
| weights | `rt/experts.bin` 708 MB + `rt/dense.bin` 116 MB | `qwen3.8-flash-next-dflash2.q4_0.pack` 522 MB |
| VRAM when loaded | 836 MiB (`experts 675, dense 111`) | 890 MiB |

Both windows are the product defaults for their drafter, so this compares the two shipped configurations,
not two hand-tuned ones. The MTP weights were fetched from ModelScope (`STRATA_SOURCE=modelscope`,
`tools/mtp_fetch.py fetch`), every tensor checked against the pinned revision's SHA256, then
`tools/mtp_pack.py --experts q2_0` and `tools/mtp_rt.py` built `rt/`.

## Method

One engine start per cell, cold, then one `/v1/chat/completions` request: the same 30-word filler
paragraph repeated to the tier length, `temperature 0`, `max_tokens 256`. Nothing is reused between cells.

- engine 0.1.58 (stock component), CUDA 12, 2x Tesla P40 (sm_61), Windows 11
- `--max-context 524288 --kv int8 --kv-resident 32768 --rope-scaling yarn --rope-scale 2`
- `--layer-split auto` (CUDA1 runs layers 25-47), `--expert-cache auto`, `--prefill auto` (8192-token chunks)
- env in every cell: `STRATA_DENSE_F32=1 STRATA_SELECT_SIMT=1 STRATA_PATTN_SM60=1 STRATA_PREFILL_TIMING=1`
- the numbers are the engine's own `timings.prompt_per_second` and `timings.predicted_per_second`;
  acceptance is the last `drafts accepted X of Y` in that start's log

Scripts: `draft_sweep.py` (every tier), `draft_ab.py` (the first pairs). Raw records: `draft_sweep.json`,
`draft_ab.json`, and the engine logs beside them.

## Decode

| prompt tokens | MTP decode | MTP accepted | DFlash2 decode | DFlash2 accepted | faster |
| ---: | ---: | ---: | ---: | ---: | --- |
| 128,032 | **32.7** | 122/164 (74.4 %) | 31.9 | 135/212 (63.7 %) | MTP +2.5 % |
| 192,052 | 30.3 | 127/189 (67.6 %) | **32.4** | 138/197 (70.1 %) | DFlash2 +6.9 % |
| 200,032 | **31.8** | 127/179 (71.0 %) | 31.3 | 139/218 (63.8 %) | MTP +1.6 % |
| 256,042 | 29.7 | — (66.3 %) | **30.1** | 140/243 (57.6 %) | DFlash2 +1.3 % |
| 300,052 | 29.6 | — (65.4 %) | **30.4** | 134/209 (64.1 %) | DFlash2 +2.7 % |

## Prefill

| prompt tokens | MTP prefill | DFlash2 prefill | MTP against DFlash2 |
| ---: | ---: | ---: | ---: |
| 128,032 | 596.4 | 605.1 | −1.4 % |
| 192,052 | 600.9 | 611.2 | −1.7 % |
| 200,032 | 602.8 | 606.5 | −0.6 % |
| 256,042 | 602.3 | 604.5 | −0.4 % |
| 300,052 | 602.7 | 605.6 | −0.5 % |

A drafter cannot change the prompt path, so this column is really "what the MTP costs in VRAM": its own
draft layer holds 836 MiB and its K/V grows with the context, which takes expert-cache slots away from
the main model. At 128K and 192K that is 1.4-1.7 %, and it shrinks as the prompt grows and the fixed
836 MiB matters less.

## What the numbers say

- **Short and long disagree.** At 128K the MTP is ahead on decode (32.7 against 31.9, acceptance 74.4 %
  against 63.7 %); from 192K on DFlash2 is ahead on four of the five tiers.
- **The margin is small except at 192K.** +1.3 % at 256K and +2.7 % at 300K are barely outside the
  ±0.5 % spread this machine shows on repeat runs; 192K's +6.9 % is the one clear gap.
- **The MTP's acceptance stays higher** (65-74 % against 58-70 %) but its window is narrower (4 tokens
  against 6), so it wins the ratio and loses the throughput at long context.
- **Neither is worth switching for on speed alone at 300K.** The MTP costs 1.7 % of prefill and 2.7 % of
  decode there. The one thing that argues for DFlash2 beyond speed: the MTP layer is another 0.9 GB of
  VRAM competing with the expert cache, which is the scarce resource at this context length.
