# HunyuanVideo 1.0 on the PyTorch-native neuron backend vs the Trainium AoT backend (trn2.3xlarge)

**Model:** `hunyuanvideo-community/HunyuanVideo` at revision `e8c2aaa66fe3742a32c11a6766aecbf07c56e773` (diffusers layout; transformer in bf16)  
**Backends:** neuron (TorchNeuron: torchrun, one process per NeuronCore, per-block `torch.compile`) and the Trainium AoT backend (NxD), the same `difflet` CLI on the same host  
**Device:** trn2.3xlarge: 1 device, 4 NeuronCores (LNC=2, 24 GiB HBM per core), 12 vCPU, 124 GB RAM, one EBS root volume  
**Workload:** 320x512x61 (16 latent frames, 10,240 image tokens), 50 steps, guidance 6.0 (embedded: HunyuanVideo is guidance-distilled, one DiT forward per step), seed 42, prompt "a cinematic shot of a red fox running through a snowy forest", `HF_HUB_OFFLINE=1`, pinned revision  
**Campaign:** 2026-10-25 00:18 – 05:30 UTC, branch `feat/hunyuan-neuron`. Native cells at `b38f3bc` (the tp4 default and `--device-vae` cells) and `d6ffb9a` (every later cell, AoT included; that commit only flips the neuron decode default, see "Decode placement"); clean tree in every run (`dirty=0` in each `run.log`)  
**Raw evidence:** kept out of git, under `/home/ubuntu/bench-2026-10-25/` on the benchmark host (`B` below): per run `B/hunyuan_video/<cell>/<run>/` holds `run.log` (the CLI output and `/usr/bin/time -v`), `summary.json` (harness record: command, HEAD, meminfo, probe timeline, hashes, mp4 check), `probe/` (per-process events), `nmon.jsonl` (neuron-monitor, 1 s), `out.mp4`, `mid.png` (frame 30) and `work/latents.pt`. `B/hunyuan_video/summary.jsonl` has one line per run. The harness and probe are in `B/scripts/` and `B/probe/`.

Every number below comes from a run made for this report and names its run directory, unless it is marked *reused*. Inferences are labelled.

## TP4: native vs AoT

Native is the neuron backend at its default placement after `d6ffb9a` (host Llama/CLIP encode on rank 0, decode in a one-core vae launch). AoT is the Trainium backend (`--backend trainium`, the NxD venv) with its own default placement: CLIP in a one-core stage, Llama TP4 on the device in a stage of its own, then the DiT and a compiled bf16 VAE decoder (replicated on the four ranks) in the generate stage. Both at tp4. "Compile cold" is `difflet compile` on a fresh cache root.

| metric | native (`tp4_devvae`) | AoT (`aot_tp4`) | native vs AoT | native, host decode (`tp4_default`, the default before `d6ffb9a`) |
|---|---:|---:|---:|---:|
| DiT step median / p10 / p90 | **808.3** / 806.7 / 809.9 ms | **813.9** / 813.4 / 814.5 ms (runs 814.0, 813.8) | −0.7% | 808.7 / 806.9 / 810.2 ms |
| compile cold | **2877.2 s** (DiT 177 s, eager VAE 1752 s, vae launch exit 927 s) | **4347.7 s** (CLIP 41 s, Llama 165 s, DiT build 445 s, VAE build 3684 s; pre-sharded DiT and CLIP weights reused, see below) | −33.8% | 185.2 s (DiT only) |
| e2e warm | **239.1 s** (241.3, 233.7, 242.1) | **132.9 s** (133.0, 132.9; gen0 135.8) | +79.9% | 539.8 s |
| e2e cold, page cache dropped | **570.4 s** | **617.3 s** (Llama 129.0 s, DiT 371.4 s weight init) | −7.6% | 881.1 s |
| peak HBM per DiT core: alloc / neuron-monitor | 11.67 / 12.98 GiB (vae launch: 11.53 / 20.89 GiB on its core) | — / 15.52 GiB (DiT and VAE in one process; the AoT stages are not torchrun ranks, so the probe records no `max_memory_allocated`) | −16% (neuron-monitor) | 11.67 / 12.98 GiB |
| output (mid frame opened) | fox running through snow, `68adae23…` | fox running through snow, `201e5ef7…` (other composition: *inference*, torch 2.9 and 2.14 draw different seed-42 noise, as `ltx_2_neuron.md` measured) | | same latents as native |
| logs | `B/hunyuan_video/tp4_devvae/` | `B/hunyuan_video/aot_tp4/` | | `B/hunyuan_video/tp4_default/` |

What the table says:

- **The DiT is at parity: native's step is 0.7% faster** (808.3 vs 813.9 ms; the two-core attention is already in HunyuanVideo's AoT CLI, `_VIRTUAL_CORE_SIZE = 2`). The earlier AoT figure, 850.6 ms at 20 steps on neuronx-cc 2.25 (`benchmark/trn2/hunyuan_video.md`, 2026-06-30), is a different workload and toolchain and is not used here.
- **Warm e2e: AoT is 106 s faster (132.9 vs 239.1 s), and almost all of it is the decode.** AoT decodes with a compiled bf16 VAE inside the generate stage in 17.1 s (+ 7.2 s weight init). Native's eager fp32 vae launch costs about 118 s end to end (launch 5.3, imports 2.3, load 10.7, decode 73.9, frames to mp4 21.9, exit 3.7). Up to the end of the denoise the two are close: 123.4 s into native's run (launch 2.6 + 21.0, load and warm-up 57.0, host encode 2.3, denoise 40.4) against 113.1 s into AoT's (CLIP and Llama stages 11.8 + 16.2 plus gaps, generate stage start-to-load 5.1, load 36.2, denoise 41.0); up to 10 s of the difference is native's launch jitter (below).
- **Cold e2e: native is 7.6% faster** (570.4 vs 617.3 s): both read the checkpoint cold. Native loads the DiT and the host Llama/CLIP in one 360.6 s load phase; AoT reads Llama in its own stage (weights 129.0 s) and then the pre-sharded DiT (weight init 371.4 s), one after the other, which costs more than its faster decode saves.
- **Compile: native is 33.8% faster, and both are dominated by the VAE.** Native's eager decoder compiles 3,751 per-op kernels in 1752 s and its launch then took 927 s to exit (open issue 2); AoT's compiled decoder took 3582 s in neuronx-cc's priority-HLO compile. **AoT's figure is a lower bound:** it reused the pre-sharded DiT (tp4) and CLIP weights already in `~/.cache/difflet/_shared_weights/` from before this campaign ("Reusing pre-sharded checkpoints" in `aot_tp4/compile/run.log`), so it did not pay the weight sharding that dominated AoT compiles in `neuron_final.md`. Native writes nothing (its weight store reports "no gain": the checkpoint is already bf16).
- **HBM: native uses 2.5 GiB less per DiT core** by neuron-monitor (12.98 vs 15.52 GiB), because its decoder lives in a separate launch; the vae core peaks at 20.89 GiB by neuron-monitor (11.53 GiB allocated; *inference:* the rest is the ~3,750 loaded eager NEFFs).
- **Where native could close the warm gap:** a compiled (traced or per-block `torch.compile`) decoder instead of the eager one (open issue 1). The DiT, the host encode and the launch are already level.

## Method

The method is `benchmark/trn2/neuron_final.md`'s "Method", with these differences:

- **Observer probe.** The campaign probe that report names (`scripts/probe/sitecustomize.py`) is not on this host: not in `/home/ubuntu/campaign`, not in any git worktree, not in `git log --all -- '*sitecustomize*'`. An equivalent observer was written for this report, `B/probe/sitecustomize.py`. It only records timestamps: each wrapper calls the original with the same arguments and returns its result; no argument, tensor or control flow changes. It loads in every process through `PYTHONPATH` (the torchrun ranks inherit it; so do the `neuronx-cc` subprocesses, which only write an empty record). It stamps `HunyuanVideoOrchestrator._scheduler_step` (after each step), `_denoise`, `_decode_latents` (AoT decode), the neuron application's `load` and `encode_prompt`, the AoT application's `load`, the VAE decoder's `load` / `decode`, and the CLI's `_write_video` / `_save_video` / `_decode_latents_host`; at exit each process writes its events and, on a neuron rank, `torch_neuronx.max_memory_allocated()`. From 02:50 UTC (the TeaCache, SP and later cells) it also reads the application's own `phase_seconds` after `load` (read-only), which gives the load split.
- **DiT step** = median / p10 / p90 of rank 0's inter-step deltas of the real loop, step 0 excluded (49 deltas per run). The table value is the median of the per-run medians of the timed runs; p10 / p90 are pooled over the timed runs.
- **Runs per cell:** a cold `difflet compile` on a fresh root `B/roots/<cell>` (also exported as `DIFFLET_COMPILE_CACHE` for native), then for the two tp4 headline cells and AoT a generate with the page cache dropped first (`sync; echo 3 > /proc/sys/vm/drop_caches`, passwordless sudo; meminfo before shows `Cached` 0.2 GiB), then one discarded warm generate (gen0), then n timed warm generates: n=3 for the two tp4 headline cells, n=2 elsewhere. The compile runs see the page cache as the previous run left it (`meminfo_before` in each `summary.json`).
- **e2e** = the process wall of `python -m difflet.cli.main generate` (harness clock; `/usr/bin/time -v` agrees to 0.1 s).
- **HBM** = `torch_neuronx.max_memory_allocated()` per rank (max over ranks), plus neuron-monitor's per-core peak of `memory_used` (sum of its categories), from `nmon.jsonl`.
- **Outputs:** every run's mp4 was decoded and checked (61 frames, 320x512, every frame's pixel std at least 49 grey levels, mean frame-to-frame difference 32 levels: moving content, not a frozen or uniform clip), and one `mid.png` per cell was opened: each shows a red fox running through snow in a forest. Within a cell every run, cold and warm, produced the same latents and mp4 bit for bit (hashes in `summary.json`; AoT keeps no latents file, its mp4s are identical).
- **No exported timeout:** `DIFFLET_NEURON_PG_TIMEOUT_S` is unset in every run (the harness removes it).

## Native tp4, default placement at `b38f3bc` (host decode) and with `--device-vae`

| metric | default at `b38f3bc`: host decode (`tp4_default`) | `--device-vae` (`tp4_devvae`; the default from `d6ffb9a`) |
|---|---:|---:|
| DiT step median / p10 / p90 (ms) | **808.7** / 806.9 / 810.2 (runs 808.8, 808.7, 807.7) | **808.3** / 806.7 / 809.9 (runs 807.6, 808.3, 808.5) |
| e2e warm (s) | 532.9, 543.9, 542.5 (mean **539.8**; gen0 535.8) | 241.3, 233.7, 242.1 (mean **239.1**; gen0 233.7) |
| e2e cold, page cache dropped (s) | **881.1** (load + warm-up 361.8, host decode 417.6) | **570.4** (load + warm-up 360.6, vae launch 7.9 + load 14.4 + decode 76.9 + write 24.2) |
| compile cold, fresh root (s) | **185.2** (generate stage only: 117 NEFFs; load + warm-up 176.0 per the CLI) | **2877.2** (generate stage 177.3; vae warm-up decode 1752.3 with about 3,751 eager NEFFs; then 927 s until the vae launch exited, see open issue 2) |
| load + warm-up per warm generate (s) | 57.2–57.6 | 57.0–57.7 |
| peak HBM: alloc / neuron-monitor per DiT core (GiB) | 11.67 / 12.98 | 11.67 / 12.98; vae core 11.53 / 20.89 |
| output | `c18d9603…` mp4, latents `b62d0891…` | `68adae23…` mp4, same latents `b62d0891…` |
| logs | `B/hunyuan_video/tp4_default/{compile,cold,gen0..gen3}/` | `B/hunyuan_video/tp4_devvae/{compile,cold,gen0..gen3}/` |

The step agrees with the DiT check's timing-only forward at this shape, 805.6 ms at tp4 (*reused*, `docs/verification/hunyuan-video-neuron-evidence.md`, `J/dit_f61_tp4_timing.log`), and with peak HBM there, 11.67 GiB.

### Where the warm e2e goes (gen1, seconds)

Times from the probe events (`B/hunyuan_video/<cell>/gen1/summary.json`; `B/scripts/breakdown.py` prints them).

| part | host decode (`tp4_default/gen1`) | device vae (`tp4_devvae/gen1`) |
|---|---:|---:|
| process wall | 532.9 | 241.3 |
| launch: driver start to rank 0 start (CLI import, torchrun) | 2.7 | 2.6 |
| launch: rank 0 start to load start (imports, runtime, process group) | 11.1 | 21.0 |
| load + warm-up | 57.3 | 57.0 |
| — of which (from `phase_seconds`, TeaCache-cell gen1, same DiT and root as `tp4_devvae`) | DiT checkpoint 4.0, build on meta 1.0, warm-up forward 36.5, host Llama + CLIP load 15.4 | same |
| host encode (Llama + CLIP, rank 0) | 2.3 | 2.3 |
| denoise, 50 steps | 40.5 | 40.4 |
| generate launch exit to decode start | 2.8 | 5.3 (vae torchrun launch) + 2.3 (imports, runtime) |
| VAE load | 2.4 (host) | 10.7 (device) |
| VAE decode | **410.1** (host CPU, fp32, 12 vCPU) | **73.9** (one core, fp32, eager) |
| frames to mp4 | 0.4 | 21.9 (includes the device-to-host copy that completes the decode; *inference*) |
| torchrun / driver exit | 0.0 | 3.7 |

**AoT, for comparison** (`aot_tp4/gen1`, seconds; one process per stage, NxD, no torchrun):

| part | seconds |
|---|---:|
| process wall | 133.0 |
| CLIP stage (one core): process start to exit | 11.8 (weight init 6.9) |
| Llama stage (TP4): process start to exit | 16.2 (weights 10.6) |
| generate stage: process start to exit | 100.0 |
| — start to load | 5.1 |
| — load (DiT weight init 28.4, VAE 7.2) | 36.2 |
| — denoise, 50 steps | 41.0 |
| — decode (compiled bf16 VAE) + mp4 | 17.1 + 0.7 |
| gaps between stages and driver exit | 5.0 |

- **The host decode is 76% of the default warm e2e at `b38f3bc`** (410.8 of 539.8 s, means over gen1–3). The device vae launch cuts the decode-and-write part from 415.7 s (gap 2.8, load 2.4, decode 410.1, write 0.4) to about 117.8 s (launch, imports, load, decode, write and exit), so the warm e2e falls from 539.8 to 239.1 s (−55.7%) and the cold one from 881.1 to 570.4 s (−35.3%). **Hence `d6ffb9a` makes the vae launch the default** (`_DEVICE_VAE_DEFAULT = True` in `difflet/cli/orchestrators/hunyuan_video.py`, pinned by `tests/unit/cli/test_orchestrator_hunyuan_neuron.py::test_device_vae_is_the_default`; `--host-vae` keeps the host decode).
- **The price is a one-time 29-minute eager compile of the decoder** (1752 s; the root ends with 3,868 NEFFs, 117 of them the DiT's as in `tp4_default`, so about 3,751 per-op decoder NEFFs; *inference:* each distinct tile shape of diffusers' tiled 3D decode, interior and edge tiles in space and time, compiles its own set of op kernels). `compile` now pays it; a `generate` on a root without it would pay it in its decode.
- **Rank 0's start-to-load time is bimodal, 11.1 or 20–21 s,** in both cells (11.05, 19.79, 20.68 s for `tp4_default` gen1–3; 20.99, 11.27, 20.04 s for `tp4_devvae`). It is the launch jitter `neuron_final.md` reports as open issue 2 and accounts for most of the run-to-run e2e spread (±5 s).
- **The warm-up forward costs 36 s of every launch** even with every NEFF cached; the denoise itself is 40 s. *Inference:* most of it is loading the 60 block NEFFs onto the cores (the first decode of a fresh vae process shows the same effect, below).

## Decode placement and VAE dtype

`NeuronHunyuanVideoVaeDecoderApplication` (the CLI's decoder: diffusers' `AutoencoderKLHunyuanVideo`, tiled 256 px / 16 frames, eager on one core) decoding the same latents (`J/cli/tp2_host.work/latents.pt`, 1x16x16x40x64) on the device in fp32 and bf16, against the fp32 host decode of those latents (`J/vae_cpu_fp32.pt`). Script: `J/vae_dtype.py` (reused from the earlier attempt) and `B/scripts/vae_repeat.py` (same decoder, every decode saved), run through `B/scripts/vaebench.py`.

| decode | first decode in a fresh process, warm cache (s) | later decodes, same process (s) | cold compile (s) | peak HBM alloc / neuron-monitor (GiB) | vs fp32 host decode | log |
|---|---:|---:|---:|---:|---|---|
| host CPU, fp32 | 417.7 (*reused*, `J/vae_cpu_fp32.log`); 410.1–412.4 in the CLI here | — | — | — | reference | `tp4_default/gen*` |
| host CPU, bf16 | 185.9 (one decode) | — | — | — | rel-L1 3.8e-3, mean abs 0.31 grey levels, PSNR 55.9 dB | `B/hunyuan_video/vae/cpu_bf16/run.log` |
| device, fp32 (the default) | 95.0, 95.9 | 41.0, 40.7, 40.8 | 1752.3 (in `tp4_devvae/compile`) | 11.53 / 20.89 | rel-L1 **3.5e-7**, max abs 8.6e-6, PSNR 136.7 dB | `B/hunyuan_video/vae/neuron_fp32/`, `neuron_fp32_repeat/` |
| device, bf16 | 51.8 | 18.3, 18.8 | 2258.2 (first decode on a fresh root) | — / 12.44 | rel-L1 **0.226**, mean abs 18.2 grey levels, PSNR **20.8 dB**; visible colour banding and tints | `B/hunyuan_video/vae/neuron_bf16_cold/`, `neuron_bf16_warm/` |

- **fp32 on the device matches the fp32 host decode to 3.5e-7** and is identical between the first and later decodes of a process (`torch.equal`).
- **bf16 is 2.2x faster per decode but fails on accuracy:** 20.8 dB against the fp32 decode, with banding and purple/blue tints visible in the frames (`B/hunyuan_video/vae/bf16_f30.png` vs `fp32cpu_f30.png`). **The default stays fp32 (amendment A6 unchanged).** The error is the device's bf16 path, not bf16 itself: the same decoder in bf16 on the host CPU is at 55.9 dB (rel-L1 3.8e-3) against the fp32 decode. AoT's compiled bf16 decoder also gives clean frames (`aot_tp4/cold/mid.png`; not compared numerically, its latents differ). Not debugged further here (open issue 3): the CLI always builds the decoder in fp32, so no CLI path reaches bf16.
- **Most of a fresh process's first decode is not compute:** 95 s first vs 41 s later in fp32, 52 vs 18 s in bf16. *Inference:* the extra 54 s / 34 s is loading the ~3,750 eager NEFFs on first use. A decoder that stayed resident (as in serving) would decode in 41 s.
- **Comparing mp4s is misleading for this.** Writing the device fp32 decode and the host fp32 decode through the CLI's own `_write_video` (`B/scripts/write_mp4.py`): before encoding, 809 of 29,982,720 8-bit values differ, by 1 level; after x264 the two mp4s differ by 2.06 levels on average (37.8 dB, max 91) (`B/hunyuan_video/vae/mp4_encode_check/`). The host- and device-decoded mp4s of the tp4 cells differ by the same amount (1.83 levels, 38.1 dB) for the same reason. Compare decoded tensors, not mp4s.

## Extra cells (n=2)

| cell | DiT step median / p10 / p90 (ms) | e2e warm (s) | other | log |
|---|---:|---:|---|---|
| tp4 `--sp`, `--host-vae` (compared with `tp4_default`, host decode) | 809.3 / 808.0 / 811.4 (runs 809.2, 809.3) | 544.9, 546.6 (mean 545.8; gen0 545.2) | compile cold 177.5 s; HBM 11.61 / 13.08 GiB | `B/hunyuan_video/tp4_sp_hostvae/` |
| tp4 `--teacache-cadence 2` (root of `tp4_devvae`: the cache key does not include TeaCache, so no compile) | 30 full steps per run: 806.8–815.6 (median 809.2); 20 skipped: 1.3–2.6 | 228.6, 227.8 (mean **228.2**; gen0 219.1) | 20 of 50 steps skipped (`[teacache] stats: full_steps 30, skipped_steps 20`); denoise 24.3 s vs 40.4 s | `B/hunyuan_video/tp4_teacache2/` |
| tp4 adaptive TeaCache | not run | | no HunyuanVideo TeaCache calibration exists on this host (searched the repo, `~/.cache`, every worktree and every JSON with `poly_coef` under `/home/ubuntu`) | |

- **SP gives nothing at tp4 here:** +0.07% per step (809.3 vs 808.7 ms); 0.06 GiB less allocated HBM (11.61 vs 11.67 GiB) but 0.10 GiB more by neuron-monitor (13.08 vs 12.98 GiB). *Inference:* at 10,240 tokens the norm and modulation regions SP shards are small next to the attention and MLP.
- **TeaCache cadence 2 takes 40% off the denoise (40.4 → 24.3 s) but only 4.6% off the warm e2e (239.1 → 228.2 s)**, because the denoise is 17% of the e2e. Its output stays close: final latents vs the run without TeaCache have rel-L1 1.51e-2 and cosine 0.99984, and the mid frame shows the same fox and scene (`tp4_teacache2/gen0/mid.png`). The skipped steps take 1.3–2.6 ms each (host-side residual reuse).

## Parity (reused from this campaign's verification stages)

| check | result | log (`J = /home/ubuntu/.claude/jobs/28e8164e/tmp/hv1_device`) |
|---|---|---|
| end to end, TP2 compile, full depth, 5x320x512, 3 steps, seed 42: our application vs diffusers' fp32 `HunyuanVideoPipeline` latents, from the same bf16 embeddings | **PASS**: rel-L1 7.601e-2 = **1.003x** the bf16 diffusers control (7.575e-2), gate 1.15x; cos 0.996888 | `J/e2e_tp2.log` (*reused*) |
| the same with our rank-0 host encoder | **PASS**: encoder outputs equal diffusers' `encode_prompt` exactly (rel-L1 0, mask equal, 11 valid tokens); latents 1.003x | `J/e2e_tp2_encode.log` (*reused*) |
| end to end, TP2xCP2 gather_kv | **PASS**: 1.021x | `J/e2e_tp2cp2_gather.log` (*reused*) |
| DiT per layout | truncated depth: all 8 layouts PASS (0.948x–1.060x); full depth: compiled layouts at 1.188x–1.191x on the t=900 / 20-token oracle (FAIL as gated), 0.838x–1.054x on the other timesteps | `docs/verification/hunyuan-video-neuron-evidence.md` (*reused*) |

## Open issues

1. **The vae launch's cold compile is 29 minutes** (3,751 eager per-op NEFFs) and a fresh vae process's first decode takes 95 s against 41 s for later decodes in the same process (*inference:* the difference is loading those NEFFs). Options, not tried: compile the decoder's tile function per block as the DiT is (fewer, larger NEFFs), or keep the decoder resident across requests (serving).
2. **After the compile-mode warm-up decode, the vae launch took 927 s to exit** (the rank's last Python `atexit` handler ran at 1949.3 s; torchrun returned at 2876.6 s; `B/hunyuan_video/tp4_devvae/compile/summary.json`). Generate-mode vae launches exit 3–4 s after the mp4 is written. *Inference:* process teardown after a compile of 3,751 NEFFs (runtime unload or compile-cache flush); not investigated. It adds 15 minutes to the cold compile above.
3. **The eager bf16 decoder on a NeuronCore is inaccurate** (20.8 dB against the fp32 decode, visible banding) while the same decoder in bf16 on the host is at 55.9 dB. Not reachable from the CLI (the vae launch builds the decoder in fp32), so it blocks only a bf16 default, which would decode 2.2x faster. Suspected cause, not verified: one of the eager ops (the decoder's GroupNorm or causal Conv3d) losing precision in bf16 on the device; bisecting it op by op against the host bf16 decode is the next step.
4. **The 36 s warm-up forward of every generate launch,** with all NEFFs cached, is about as long as the 50-step denoise.
5. **Full-depth DiT parity on the t=900 / 20-token oracle** stays at 1.19x the control for every compiled layout (`docs/verification/hunyuan-video-neuron-evidence.md`); a gating decision is pending with the plan owner.

## Reproduction

```bash
# native: the native venv's bin/ on PATH (neuronx-cc is spawned by name), this worktree on PYTHONPATH
R=<fresh root>
DIFFLET_COMPILE_CACHE=$R HF_HUB_OFFLINE=1 python -m difflet.cli.main compile --backend neuron \
  --model-id hunyuanvideo-community/HunyuanVideo --revision e8c2aaa66fe3742a32c11a6766aecbf07c56e773 \
  --height 320 --width 512 --num-frames 61 --tp-degree 4 [--sp] [--host-vae|--device-vae] --cache-dir $R
DIFFLET_COMPILE_CACHE=$R HF_HUB_OFFLINE=1 python -m difflet.cli.main generate --backend neuron \
  --model-id hunyuanvideo-community/HunyuanVideo --revision e8c2aaa66fe3742a32c11a6766aecbf07c56e773 \
  --height 320 --width 512 --num-frames 61 --tp-degree 4 [--sp] [--host-vae|--device-vae] \
  [--teacache-cadence 2] --steps 50 --guidance-scale 6.0 --seed 42 \
  --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.mp4 --cache-dir $R
# AoT: the same with the NxD venv (/home/ubuntu/venvs/difflet-neuron) and --backend trainium
#   (no DIFFLET_COMPILE_CACHE); pass --steps 50 (the AoT HunyuanVideo CLI defaults to 4)
```

Harness: `B/scripts/hvbench.py cell --cell <cell> --venv native|aot --runs compile,cold,gen0,gen1,gen2[,gen3] -- <flags>` (it adds the shape, revision, `--cache-dir B/roots/<cell>`, and for a generate the steps, guidance, seed, prompt, output and `--work-dir <run>/work --keep-work-dir`); `B/scripts/breakdown.py` (breakdowns, cell statistics); `B/scripts/vaebench.py` with `J/vae_dtype.py` or `B/scripts/vae_repeat.py` (VAE); `B/scripts/cmp_mp4.py`, `write_mp4.py`.

## Toolchain

- Native: `/home/ubuntu/venvs/native-312`, Python 3.12.3, torch 2.14.0, torch-neuronx 2.14.3.0.1, neuronx-cc 2.28.7631.0, nki 0.7.0, diffusers 0.38.0.
- AoT: `/home/ubuntu/venvs/difflet-neuron`, torch 2.9.1, torch-neuronx 2.9.0.2.15.32035, neuronx-cc 2.26.6360.0, neuronx-distributed 0.19.28492, neuronx-distributed-inference 0.10.18399, diffusers 0.38.0, transformers 4.57.6.
- Host runtime: aws-neuronx-runtime-lib 2.35.30.0, collectives 2.35.43.0, dkms 2.31.7.0, tools 2.33.38.0.
