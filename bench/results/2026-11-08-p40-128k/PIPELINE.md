# Where a decode window goes, and `--pipeline-windows 2` on the two-card split

This file measures two things on the 2×P40 machine: what one decode window spends its time on at a
127K prompt, and whether the one existing switch that overlaps the two cards is worth turning on.
The profile says the answer is yes, so the run config in `F:\Strata\install\strata-iq3_s.json` now
carries `"drafting": "mtp"` and `--pipeline-windows 2`.

## The profile

One engine start, one 127,042-token prompt, 96 greedily chosen tokens, `"drafting": "dflash2"`,
`STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1`. DFlash2 was the config in place at the time; the
profile is about where a window's time goes, which the drafter does not change.

Decode on that run: 31.95 ms on CUDA0 plus 34.25 ms on CUDA1 per window, 80.78 ms per window,
2.40 tokens per window, 29.7 tok/s.

| what | ms/window | share of GPU time |
| --- | ---: | ---: |
| expert matmuls (`VRAM hits`, GDN and QSA layers) | 13.98 | 21.1 % |
| hyper-connections (norm + down + up, both streams) | 12.01 | 18.1 % |
| k/v/index/out projections | 12.51 | 18.9 % |
| router, plan and shared-expert math | 6.11 | 9.2 % |
| waits and copies inside the stages (`waitA/B`, `waitCPU`, `copy+combine`) | 6.12 | 9.2 % |
| GDN conv / a-b / z / recurrence | 5.12 | 7.7 % |
| LM head (CUDA1 only) | 3.41 | 5.2 % |
| QSA scores and top-k | 2.97 | 4.5 % |
| KV index, norm-rope, append, resolve | 2.52 | 3.8 % |
| attention itself | 1.43 | 2.2 % |
| **sum of both cards' GPU stages** | **66.2** | 100 % |

The rest of the 80.78 ms is host-side: `draft` 6.62, `commit/emit` 1.36, `adapt join` 3.10,
`adapt wait` 0.27.

- **The two cards run one after another.** 66.2 of 80.78 ms is GPU work and the stages never overlap,
  so that is the whole opportunity.
- **Attention is not the problem.** 1.43 ms, 2.2 % of GPU time, including 2.97 ms of block scoring and
  top-k beside it. No attention kernel can matter at this size.
- **Expert misses are not the problem either.** The cache answered 57,309 of 58,636 lookups (97.7 %);
  one lookup in the whole request went over PCIe. The 13.98 ms is the expert *math*, not fetching it.
- **The expert math and the hyper-connections are the largest blocks** (21 % and 18 %), with the
  projections close behind at 19 %. All three are bandwidth-bound matrix products spread over 48 GDN
  and 12 QSA layers; a smaller tile or a wider read would have to attack them.

## The A/B

Two arms, one engine start each, same 127,042-token prompt and 96 greedy tokens, `"drafting": "mtp"`
in both. The only difference is `--pipeline-windows 2`. Two rounds, with the arm order reversed in the
second, because the second arm of a pair runs hotter (the P40s drop about 60 MHz past 89 °C).

| round | order | MTP baseline | MTP + pipeline 2 | difference |
| --- | --- | ---: | ---: | ---: |
| 1 | baseline first | 30.3 tok/s (34/42 °C) | **34.3** (40/44 °C) | **+13.2 %** |
| 2 | pipeline first | 30.9 (40/44 °C) | **33.2** (38/43 °C) | **+7.4 %** |

Prefill is unchanged in all four runs, 614.8–617.2 tok/s.

The two arms differ by more than the spread inside an arm: baseline read 30.3 and 30.9 (2.0 %), the
pipeline 33.2 and 34.3 (3.3 %), and both rounds put the pipeline ahead. Against DFlash2's 29.7 tok/s
on the same prompt, MTP + pipeline is 15.5 % faster.

## Why it helps

The pipeline lets CUDA0 start the next window while CUDA1 still verifies this one. It does not make a
window cheaper; it makes fewer of them and puts the host steps beside a running GPU:

| per round-1 arm | MTP baseline | MTP + pipeline 2 |
| --- | ---: | ---: |
| windows | 47 | 44 |
| tokens per window | 2.04 | 2.18 |
| ms per window | 67.48 | 63.66 |
| `draft` + `commit/emit` in the timing line | 6.6 + 4.5 ms | 0 + 0 ms |

In the pipeline arm those steps happen on the other card inside the overlap, so the window's own
timing line no longer charges them; the `strata pipeline` lines count them instead.

The guess is cheap when it is wrong. In round 1, 34 windows ran speculatively, 7 of them landed on the
path and 27 were rolled back; a speculative window took 42.70 ms against 67.59 ms for a fresh one, so
a rollback costs less than the wait it removed. Round 2 read 36 speculative, 9 on the path, 27 rolled
back.

## What it costs

- 278 MiB of CUDA0 is kept out of the expert cache: 6,551 slots become 6,441, and the hit rate moves
  from 97.4 % to 97.3 % (round 1) — inside the run-to-run variation.
- The log at load reports 691 / 565 MiB of VRAM free against 609 / 485 without the pipeline, so the
  card that gains the second verifier is not the one left shorter.
- The draft layer stays on the last card, so its chain shares that card with a window. The engine's log
  names the alternative: `STRATA_MTP_STAGE0=1` puts it on the first card, and that placement lost to the
  serial chain.

## Limits

- One prompt shape, 96 tokens, 127K. The 8K–300K grid in `DRAFTERS.md` was measured with the MTP
  without the pipeline, so its MTP column now understates this machine's best decode.
- Acceptance depends on content; the speculative/rollback split will move with it, and so will the gain.
- DFlash2 cannot use the pipeline: it drafts from the finished window's taps on every stage. The
  pipeline needs `--mtp`, `--serve`, `--spec 2` or more, and a layer split on exactly two cards.
- 192K and above were not re-run.

## Reproduce

`F:\tmp-strata-bench\decode_prof.py` is the profile; `pipeline_ab.py` runs the two arms in the order
above and `pipeline_ab2.py` with the order reversed. Both reuse `long128.py` for the config, the
start, the request and the stop.
