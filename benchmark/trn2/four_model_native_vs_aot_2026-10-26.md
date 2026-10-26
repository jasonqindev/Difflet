# Four models, native vs AoT at TP4 on trn2.3xlarge, 2026-10-26

**Status:** phase 1 of the plan in `docs/superpowers/plans/2026-10-26-four-model-native-vs-aot-metrics.md`. This version has the AoT cells for **Qwen-Image** and **LTX-2**, measured on the latest `origin/main`. Every native column is **pending (phase 2)**. HunyuanVideo 1.0 and Wan2.1 come later in phase 2. PSNR / SSIM against the 50-step CPU fp32 references come after the device runs.

**Device:** trn2.3xlarge: 1 device, 4 NeuronCores (LNC=2, 24 GiB HBM per core), 12 vCPU, 124.8 GiB RAM, one EBS root volume. The host is the same one used for `native_vs_aot_tp4_2026-10-25.md`.

**Raw evidence:** kept out of git, under `/home/ubuntu/bench-2026-10-26-4m/` on the benchmark host (`B` below).
- Each run has its own directory `B/<model>/<cell>/<run>/`, which holds:
  - `run.log`: the CLI output and `/usr/bin/time -v`, plus a header with the date, worktree, HEAD, `difflet.__file__`, the extra env and the exact command;
  - `summary.json`: the harness record (command, env, HEAD and dirty flag, meminfo, checkpoint page-cache residency, the pre-read record, the probe timeline, HLO modules, `difflet_file_precheck` and `difflet_files_seen`, `nmon_peak_gib`, `device_released`);
  - `probe/`: per-process events;
  - `nmon.jsonl`: neuron-monitor, one sample per second;
  - the outputs.
- `B/<model>/summary.jsonl` has one line per run.
- `B/<model>/<cell>/stats_gen1-3.txt` comes from `B/scripts/bench4m_stats.py`.
- `B/<model>/<cell>/hlo_allreduce.jsonl` comes from `B/scripts/hlo_allreduce.py`.

Every number below comes from a run made for this report and names its run directory, unless it is marked *reused*. Inferences are labelled.

## Method

**Code under test**
- **AoT:** `origin/main` at `540e5f086ffa5dae8e957aa94b1b23616091a7bc`, checked out as a separate detached worktree at `/home/ubuntu/aot-main`. The tree was clean in every run (`head: 540e5f0… dirty=0` in each `run.log`).
  - The five bf16 all-reduce fixes are ancestors of that commit: `fc6a580`, `b4bdf35` (Qwen-Image), `4d4cc60` (LTX-2), `7fe5308` (HunyuanVideo) and `1c8a64e` (Wan umT5). Each was checked with `git merge-base --is-ancestor`, exit 0.
  - The NxD venv `/home/ubuntu/venvs/difflet-neuron` was used with `PYTHONPATH=B/probe:/home/ubuntu/aot-main` and cwd `/home/ubuntu/aot-main`.
  - Before every run, the harness imports `difflet` with the run's exact env and refuses to start if `difflet.__file__` is not under `/home/ubuntu/aot-main` (`difflet_file_precheck`). The probe also records `difflet.__file__` in every process (`difflet_files_seen`).
  - In all 14 runs below, both fields read `/home/ubuntu/aot-main/difflet/__init__.py`. That rules out the venv's editable install, which points at another worktree.
- **Native (phase 2, not run yet):** branch `feat/hunyuan-neuron` with the native venv `/home/ubuntu/venvs/native-312`.
  - The branch HEAD when this was written is `05614a2`, which includes F12, the LTX-2 text-encoder index fix that the LTX-2 native cell needs.
  - The native runs will record their own HEAD.
- The two backends run on different commits, and this cannot be avoided: `difflet/backends/neuron` does not exist on main, and the two branches conflict in 21 files.

**Workload**
- Seed 42, prompt "a cinematic shot of a red fox running through a snowy forest", `HF_HUB_OFFLINE=1`, pinned revisions.
- TP4 with `--cp-degree 1`.
- 50 steps, with `--steps 50` and the guidance always passed explicitly. Without them, the AoT CLI runs Qwen-Image and HunyuanVideo at 4 steps, and LTX-2 at 40 steps with guidance 3.5.

**Backend selection:** main's CLI has no `--backend` flag. It chooses the backend from `DIFFLET_BACKEND` or by auto-detection. Every run exports `DIFFLET_BACKEND=trainium`.

**bf16 per component (AoT on main)**
- Every component runs at main's bf16 default. No flag that falls back to fp32 was passed: no `--host-vae`, and `--tp-reduce-dtype` was left at its default.
- Qwen-Image components: text encoder (Qwen2.5-VL, TP4 on the device), DiT (TP4) and VAE (compiled, one core).
- LTX-2 components: the DiT runs at TP4 on the device. The host pipeline holds the Gemma-3 text encoder, the connectors, the video and audio VAEs and the vocoder.
- The HLO check (per-model sections) confirms the DiT all-reduces are bf16.

**Two-core attention**
- **Qwen-Image:** main's CLI sets two-core attention itself. `difflet/cli/orchestrators/qwen_image.py:34` reads `qwen_common.VIRTUAL_CORE_SIZE`, which is 2 (`difflet/common/orchestrators/qwen_image.py:23`). `run_stage` sets it with `setdefault` (`difflet/cli/runner.py:78`). The harness does not export the variable.
- **LTX-2:** main's CLI does not set it: `difflet/cli/orchestrators/ltx_2.py` has no reference to it. Only the serving path, `difflet/serving/models/ltx_2.py`, mentions it, and the serving path is not used here.
  - The cell exports `NEURON_RT_VIRTUAL_CORE_SIZE=2` for the compile and for every generate.
  - The AoT stage key does not include this variable, so the cell uses a fresh root.
  - The DiT HLO module was checked to differ from the single-core `MODULE_20da5719`.

**Runs per cell**
- **Compile cold:**
  - `difflet compile` on a fresh root `R = B/roots/<model>_<cell>`.
  - Every run of the cell sets `DIFFLET_SHARED_WEIGHTS_DIR=$R/_shared_weights`, `NEURON_COMPILE_CACHE_URL=$R/neuron-compile-cache` and `--cache-dir $R`. The compile therefore includes every NEFF compile and all of the weight pre-sharding.
  - Before the compile, the harness drops the page cache and reads the checkpoint twice. `mincore` residency is recorded in `preread_rec`.
- **Cold e2e (the approved definition, plan decision 2):**
  - `sync; echo 3 > /proc/sys/vm/drop_caches` through passwordless sudo, with meminfo `Cached` recorded before the run.
  - A fresh process and a fresh `--work-dir`, all 50 steps.
  - The text encoder, DiT and VAE weights are read from disk inside the timed run, and no stage output is reused.
  - Compiled artifacts may exist: NEFFs, and the AoT pre-shards in the cell's own root.
- **Warm:** one discarded warm generate (gen0), then gen1–gen3 timed, in the order compile, cold, gen0, gen1, gen2, gen3, q1.
- **Quality run (q1):** one more warm generate with lossless output: PNG for Qwen-Image, a frames tensor `out.pt` for LTX-2.
  - It also sets `DIFFLET_DUMP_LATENTS_DIR=<run>` and `DIFFLET_STEP_TIMES_FILE=<run>/steps.jsonl`. Both are supported on main: `difflet/utils/latent_dump.py` and `step_timing.py`, used in `qwen_image/pipeline.py` and `ltx_2/pipeline.py`.
  - The dumped initial latents seed the CPU fp32 references (plan phase 4).
  - q1 is not part of the timing statistics.

**Metrics**
- **DiT step** = the median / p10 / p90 of the deltas between consecutive denoising steps, with step 0 excluded (49 deltas per run).
  - Qwen-Image: from the generate stage process (`QwenImageOrchestrator._scheduler_step` stamps).
  - LTX-2: from the generate process. One stamp per video+audio pair of `_scheduler_step` calls, checked in the probe self-test: 6 calls give 3 stamps.
  - The table value is the median of the per-run medians over gen1–3; p10 / p90 are pooled over gen1–3.
  - q1's `steps.jsonl`, written by main's own step timer, agrees: Qwen-Image 361.0 ms, LTX-2 400.8 ms (medians of 50 rows).
- **e2e** = the process wall of `python -m difflet.cli.main generate` (harness clock).
- **Cold-e2e breakdown** = probe spans in the cold run's `summary.json`.
- **Peak HBM** = neuron-monitor's per-core peak of `memory_used` (the sum of its categories) from `nmon.jsonl`. The AoT stages are not torchrun ranks, so there is no `max_memory_allocated`.
- **Outputs:** SHA-256 of every output and dump. The image or a frame strip was opened and checked against the prompt.

**Supervision**
- One device job at a time. Before each launch, the harness waited until `fuser /dev/neuron0` showed no process. After each run it confirmed the device was released: `device_released: true` in all 14 `summary.json`.
- The jobs ran under `scripts/supervise.sh` (Qwen-Image) and as a `systemd-run --user` unit (LTX-2).
- `DIFFLET_NEURON_PG_TIMEOUT_S` was not exported.

**Disk cleanup before phase 1 (plan decision 6, option A, approved):**

| root deleted | built at | size | freed |
|---|---|---:|---:|
| `bench-2026-10-25/roots/qwen_aot_tp4` | `4f6e1de` | 53G | 56 GB |
| `bench-2026-10-25/roots/wan_aot_tp4` (`--host-vae`) | `8b2408c` | 39G | 41 GB |
| `bench-2026-10-25/roots/wan_aot_tp4_twocore` (`--host-vae`, two-core env) | `8b2408c` | 39G | 41 GB |
| HunyuanVideo `aot_tp4` | `d6ffb9a` | 59G | 16 GB |

- All four roots were built before the bf16 fixes; their manifests have no reduce-dtype field.
- Free space on `/` went from 230.27 to 385.92 GB (`B/prep/delete_prefix_roots.log`).
- **Why the HunyuanVideo root freed only 16 GB:** its 47.95 GB of transformer, VAE and CLIP shards are hardlinked from `~/.cache/difflet/_shared_weights/hunyuanvideo-*` and from `~/.cache/difflet/hunyuan_video_{dit,clip}/…`.
- **What was kept:**
  - `~/.cache/difflet/_shared_weights` was not touched. The deleted Qwen-Image and Wan roots used their own internal `_shared_weights`. The HunyuanVideo entries are still linked from `~/.cache/difflet/hunyuan_video_*`, outside the approved list, so deleting them would free 0 bytes.
  - Nothing native, nothing in the HF cache and nothing else was deleted.
- After phase 1, the two new roots take 53 GB (Qwen-Image) and 38 GB (LTX-2) on disk, and `/` has 269 GB free.

## Qwen-Image

**Model:** `Qwen/Qwen-Image` at revision `75e0b4be04f60ec59a75f475837eced720f823b6`.  
**Workload:** 1024x1024, 50 steps, `--guidance-scale 4.0` (the DiT drops it: `guidance_embeds` is off), seed 42.  
**AoT cell:** `aot_main_tp4`, root `B/roots/qwen_image_aot_main_tp4`. Three stage processes run one after the other:
- the text stage: Qwen2.5-VL at TP4 on the device;
- the generate stage: the DiT at TP4;
- the vae stage: the compiled VAE on one core.

**Native cell:** pending (phase 2). Phase 2 is one hash/quality run on the current branch against the reused 2026-10-25 cell (plan D10, option A).

| metric | AoT on main (`aot_main_tp4`) | native | native vs AoT | AoT before the bf16 fix (2026-10-25, `4f6e1de`, *reused*) |
|---|---|---|---|---|
| DiT step median / p10 / p90 | **360.6** / 360.2 / 361.8 ms (runs 360.6, 360.9, 360.5) | pending (phase 2); reusable: 359.0 / 358.2 / 360.1 ms, *to be confirmed by hash* | (−0.4% if confirmed) | 418.2 ms (now −13.8%) |
| compile cold, checkpoint in the page cache | **1209.9 s** (text stage 128.7, generate stage 660.5, vae stage 401.1) | pending; reusable: 187.0 s, *to be confirmed by hash* | (−84.5%) | 1238.5 s (−2.3%) |
| e2e warm, n=3 (gen0 discarded) | **73.7 s** mean (74.16, 76.06, 70.98; median 74.2; gen0 72.9) | pending; reusable: 82.3 s, *to be confirmed by hash* | (+11.6%) | 75.4 s (−2.2%) |
| e2e cold, page cache dropped | **515.6 s** (breakdown below) | pending; reusable: 502.0 s, *to be confirmed by hash* | (−2.6%) | 504.2 s (+2.3%) |
| peak HBM per core (neuron-monitor) | **10.10 GiB** on all 4 cores in gen0–3 (10.04 in cold and q1) | pending; reusable: 10.81 GiB, *to be confirmed by hash* | (+7.0%) | 10.29 GiB |
| output | PNG `506c09f4…`, `work/latents.pt` `35e26fb7…`, identical in cold, gen0–3 and q1 | reused cell: PNG `b1906d53…`, latents `4da4c89e…` | different noise (torch 2.14 vs 2.9 bf16 draw) | PNG `e24798fe…` |
| logs | `B/qwen_image/aot_main_tp4/{compile,cold,gen0..gen3,q1}/` | | | |

*Native vs AoT* is (native − AoT) / AoT, as in the 2026-10-25 report. The bracketed values use the reused native numbers and stand only if the phase-2 hash check passes.

**The reusable native numbers (*reused*, to be confirmed by hash)**
- They come from `native_vs_aot_tp4_2026-10-25.md`, cell `native_tp4`, measured at `4f6e1de` on this host with the native venv listed under Toolchain.
- The cell has no weight store (bf16 checkpoint, "no gain"), so it has one cold number, under definition A and B alike.
- Phase 2 re-runs one native generate on the current `feat/hunyuan-neuron` HEAD. If its PNG is `b1906d53…` and its latents `4da4c89e…`, the four timings and the HBM figure stand. If not, the full native cell is re-measured.
- Its DiT matches AoT's in two respects: it runs in bf16 (plan §1), and its NEFFs have the two-core attention split (`attention_cte_lnc0` / `lnc1` subgraphs, plan D1).
- It differs in three respects:
  - the VAE decodes in fp32;
  - Qwen2.5-VL encodes on rank 0's host, not on the device;
  - the decode runs in a launch of its own.

  These are placement and dtype differences (plan D2, D8), kept as decided for the device runs.

**Compile (`compile/run.log`)**
- **Text stage, 128.7 s:** build 101.0 s, of which the priority neuronx-cc compile took 58.5 s; sharding 7.2 s.
- **Generate stage, 660.5 s:** DiT build 293.4 s (module load 92.1, HLO 16.0, neuronx-cc 30.2), then pre-sharding 361.6 s (4 files, 38 987 MB).
- **VAE stage, 401.1 s:** neuronx-cc 386.2 s.
- **Pre-read:** the drop left `Cached` at 0.27 GiB and the checkpoint 0% resident. The checkpoint (53.74 GiB, 14 files) was then read twice: 439.1 s, then 20.6 s, 100% resident before the compile.

**Cold e2e breakdown (`cold/summary.json`, seconds from process start; `Cached` 0.26 GiB before)**

| part | start | duration |
|---|---:|---:|
| text stage | 0.25 | **140.2** |
| — text-encoder load (Qwen2.5-VL from disk) | 18.19 | 121.8 |
| generate stage | 144.26 | **353.1** |
| — DiT pre-shard load | 149.83 | 329.1 |
| — denoise, 50 steps | 479.0 | 18.4 |
| vae stage | 498.8 | **15.2** |
| — VAE load | 503.38 | 10.0 |
| — decode | 513.42 | 0.27 |
| gaps between stages, driver exit | | 7.1 |
| **process wall** | | **515.6** |

Against the pre-fix AoT cold run (504.2 s), the text stage is level (140.2 vs 140.7 s). The DiT load is 13.2 s longer (329.1 vs 315.9 s). The DiT load reads the cell's 39 GB of pre-shards from disk, so most of this e2e is EBS read time.

**Where the warm e2e goes (gen2, seconds; the 2026-10-25 pre-fix AoT gen2 in brackets, *reused*)**

| part | AoT on main (`gen2`) |
|---|---:|
| process wall | 76.06 [76.4] |
| text stage | 16.7 [16.4]; encoder load 11.9 |
| generate stage | 41.3 [41.2]; DiT load 18.0, denoise 18.4 [21.1] |
| vae stage | 13.8 [14.4]; VAE load 8.6, decode 0.27 |

- The denoise is 2.7 s shorter, in line with the 57.6 ms per-step drop from the bf16 all-reduce.
- gen2's DiT load is 3 s longer than gen1's and gen3's (15.0 and 14.9 s). The warm spread, 71.0–76.1 s, is mostly load time, not compute: the denoise is 18.2–18.4 s in every run.

**bf16 and the HLO modules (`compile/summary.json` `hlo_modules`; `hlo_allreduce.jsonl`)**
- DiT `MODULE_5421e02c…`: 240 all-reduces, all bf16 (120 on `[1,1024,3072]` and 120 on `[1,4096,3072]`).
- Text encoder `MODULE_9887b749…` and `MODULE_4804a098…`: 56 bf16 all-reduces each.
- VAE `MODULE_c08dfa24…`: none (one core).
- There is no f32 all-reduce in any module.

**Quality run q1**
- `out.png` (`506c09f4…`, lossless PNG) was opened: a red fox walking across snow in front of a snowy forest, on prompt.
- `initial.pt` (`c2d9d8b3…`) and `final.pt` (`a9c1f37e…`, packed latents) are kept for the CPU reference.
- `initial.pt` is byte-identical to `/home/ubuntu/campaign/bf16-reduce/runs/qwen/bf16/warm1/initial.pt`: the same seed-42 noise as the 20-step bf16 campaign.
- `steps.jsonl` has 50 rows, median 361.0 ms.

**Against the post-fix 20-step campaign** (`/home/ubuntu/campaign/bf16-reduce/runs/qwen/summary.json`, *reused*): step 361.9 → 360.6 ms (−0.4%). Its 60.6 s warm and 498.8 s cold e2e were measured at 20 steps, so they are not comparable.

## LTX-2

**Model:** `Lightricks/LTX-2` at revision `47da56e2ad66ce4125a9922b4a8826bf407f9d0a`.  
**Workload:** 480x704x49 (2310 video tokens, 51 audio frames), 50 steps, `--guidance-scale 1.0` (no CFG), seed 42.  
**AoT cell:** `aot_main_tp4_twocore`, root `B/roots/ltx_2_aot_main_tp4_twocore`, with `NEURON_RT_VIRTUAL_CORE_SIZE=2` exported for the compile and every generate (`env_extra` in every `run.log`).
- One process: the host pipeline (Gemma-3 text encoder, connectors, video and audio VAEs, vocoder) and the DiT at TP4 on the device.
- The prompt is encoded on the host.

**Native cell:** pending (phase 2, full cell under definition A, needs F12). There are no reusable native numbers. The 2026-10-18/20 native measurements (`ltx_2_neuron.md`) were taken with an older toolchain, and their raw evidence (`/home/ubuntu/bench-ltx2-neuron/`) is not on this host.

| metric | AoT on main, two-core (`aot_main_tp4_twocore`) | native | native vs AoT |
|---|---|---|---|
| DiT step median / p10 / p90 | **400.1** / 398.1 / 400.9 ms (runs 398.2, 400.3, 400.1) | pending (phase 2) | pending |
| compile cold, checkpoint in the page cache | **1052.2 s** (model build 542.2, pre-sharding 470.5) | pending (phase 2) | pending |
| e2e warm, n=3 (gen0 discarded) | **66.55 s** mean (66.00, 68.00, 65.64; median 66.0; gen0 66.1) | pending (phase 2) | pending |
| e2e cold, page cache dropped | **778.1 s** (breakdown below) | pending (phase 2), A (weight store); B later | pending |
| peak HBM per core (neuron-monitor) | **9.81 GiB** on all 4 cores (cold, gen2, gen3, q1; 9.75 in gen0 and gen1) | pending (phase 2) | pending |
| output | `out.mp4` `5aab9e4e…`, identical in cold, gen0–3; q1 `out.pt` `1a875738…` | pending (phase 2) | |
| logs | `B/ltx_2/aot_main_tp4_twocore/{compile,cold,gen0..gen3,q1}/` | | |

**Compile (`compile/run.log`)**
- **Model build, 542.2 s:** DiT module load 92.4 s, HLO 22.7 s, priority neuronx-cc 80.1 s, and the weight-layout transformation.
- **Pre-sharding, 470.5 s:** 4 files, 38 011 MB.
- **Pre-read:** the drop left `Cached` at 0.27 GiB and the checkpoint 0% resident. The checkpoint (transformer, text_encoder `model-*-of-00011`, connectors, vae, audio_vae, vocoder: 85.71 GiB, 23 files) was then read twice: 701.1 s, then 32.8 s, 100% resident before the compile.

**Cold e2e breakdown (`cold/summary.json`, seconds from process start; `Cached` 0.26 GiB before)**

| part | start | duration |
|---|---:|---:|
| imports and start-up | 0 | 21.1 |
| host pipeline load (Gemma-3, connectors, VAEs, vocoder) | 21.13 | **373.8** |
| DiT pre-shard load | 394.89 | **309.7** |
| prompt encode (host) | 704.59 | 30.7 |
| denoise, 50 steps | 735.32 | 20.0 |
| decode | 755.27 | 15.2 |
| save video | 770.49 | 1.4 |
| process exit | 771.9 | |
| **harness wall** | | **778.1** |

- The cold run is 88% load: 683.5 s of the 778.1 s is host pipeline and DiT load.
- **Warm gen1 for comparison:** imports 4.7, host load 5.0, DiT load 11.6, encode 10.6, denoise 19.9, decode 9.4, save 1.2; process exit 62.5 s, wall 66.0 s.

**Two-core attention: verified two ways**
- The DiT HLO module is `MODULE_e1043f485ceeeb495915+9de06980` (`compile/summary.json` `hlo_modules`), not the single-core `MODULE_20da5719`.
- The NKI kernel configs were compared with the single-core module, which is still in `/var/tmp/neuron-compile-cache` (`B/ltx_2/aot_main_tp4_twocore/hlo_nki_attention_vs_20da5719.txt`, from `B/scripts/hlo_nki_diff.py`):
  - In this module, all 288 `attention_cte` calls (6 distinct kernels × 48) are `attention_cte_lnc0_*` with `"grid": 2`.
  - In `MODULE_20da5719`, all 288 are `attention_cte_*` with `"grid": 1`.
  - The two modules share no kernel config.

**bf16 (`hlo_allreduce.jsonl`)**
- The hidden-state all-reduces are bf16: 192 on `[1,2310,4096]` (video) and 192 on `[1,51,2048]` (audio).
- The f32 all-reduces are only on `[1,N,1]` shapes: 96 on `[1,1024,1]`, 240 on `[1,2310,1]` and 240 on `[1,51,1]`.
- These counts are the same as in the 2026-10-24 single-core bf16 module, which is `MODULE_20da5719` itself (`/home/ubuntu/campaign/bf16-reduce/runs/ltx2/hlo_allreduce.txt`). Its fp32 predecessor, `MODULE_85212230`, had the same five groups, with the hidden-state ones in f32.
- *Inference, not checked:* the `[1,N,1]` reductions are norm statistics.

**Quality run q1**
- `out.pt`: the decoded frames, shape (1,49,3,480,704), bf16 in [0,1], `1a875738…`.
- `final_video.pt` (`9c454431…`, (1,2310,128) bf16) and `final_audio.pt` (`648b6af8…`, (1,51,128) bf16) are kept.
- `initial_video.pt` (`4458e4cf…`) and `initial_audio.pt` (`8e52babb…`) are byte-identical to the 2026-10-24 campaign's dumps: the same seed-42 noise.
- `steps.jsonl` has 50 rows, median 400.8 ms.

**Content: only partly on prompt.**
- I opened the mid frame and a strip of frames 0, 12, 24, 36 and 48 (`q1/strip_0_12_24_36_48.png`).
- Every frame shows a snowy forest with snow-covered trees and snow on the ground.
- A red-orange and dark animal-like shape moves through it, but it is incoherent and cannot be recognised as a fox.
- The 2026-10-24 20-step output (`B/ltx_2/ref_2026-10-24/warm1_mid.png`) is also only loosely on prompt.
- `ltx_2_neuron.md` notes that guidance 1 (no CFG) gives low quality on both backends.
- *Inference:* this is the workload, not a regression. The 50-step CPU fp32 reference from the same noise (phase 4) will settle it.

**Against earlier AoT measurements (*reused*, not like for like)**
- **2026-10-20 two-core AoT, pre-fix (fp32 DiT all-reduce)** (`ltx_2_neuron.md`, 50 steps; raw evidence not on this host):
  - step 419.2 → 400.1 ms (−4.6%);
  - warm e2e 68.0 → 66.55 s (−2.1%);
  - cold e2e 781.3 → 778.1 s (−0.4%).
  - Compile, 576.9 vs 1052.2 s, is not comparable: the 2026-10-19 compile's pre-shard state is not documented, and today's compile includes 470.5 s of pre-sharding.
- **2026-10-24 single-core bf16 campaign** (`/home/ubuntu/campaign/bf16-reduce/runs/ltx2/`, 20 steps):
  - step 433.5 → 400.1 ms (−7.7%). Both use the bf16 all-reduce, so this is the two-core gain.
  - Its 54.8 s warm, 767.5 s cold and 555.9 s compile are not comparable: 20 steps, and that compile reused pre-shards.

## Problems during phase 1 (harness, not Difflet)

1. **The first Qwen-Image compile exited in 0.1 s with rc=2.**
   - Cause: the harness passed `--backend trainium`, which main's CLI does not have.
   - Fix: the job scripts pass `DIFFLET_BACKEND=trainium` instead. Every compile and generate argument list for both models then passed a parse-only check against main's parser.
   - The failed run is kept as `B/qwen_image/aot_main_tp4/compile.failed-backend-flag`. The root had not been created, so it was still fresh for the rerun.
2. **The first Qwen-Image cold run was killed about 2 min in (08:56:42).**
   - Cause: the job had been started under `supervise.sh` from a Monitor task. It died when that Monitor reached its 30-minute limit, even with `setsid`.
   - The job quarantined the run as `cold.partial-1793005519` and reran cold after a fresh page-cache drop. All later runs are clean.
   - The LTX-2 cell ran as a `systemd-run --user` unit instead, which survived a Monitor expiry.

## Next

- **Phase 2 (device):**
  - Qwen-Image native hash/quality run, to confirm the reused native column;
  - LTX-2 native full cell under definition A;
  - HunyuanVideo AoT and Wan2.1 AoT;
  - HunyuanVideo and Wan2.1 native hash/quality runs.
- **Phase 4 (CPU):** the 50-step CPU fp32 references, Qwen-Image and LTX-2 first. The AoT-noise inputs are `B/qwen_image/aot_main_tp4/q1` and `B/ltx_2/aot_main_tp4_twocore/q1`. The LTX-2 reference still needs a lossless frames output (plan D9).

## Toolchain

- **AoT:** `/home/ubuntu/venvs/difflet-neuron`: Python 3.12.3, torch 2.9.1, torch-neuronx 2.9.0.2.15.32035+de43f57c, neuronx-cc 2.26.6360.0+6f180f47.
- **Native (phase 2):** `/home/ubuntu/venvs/native-312`: Python 3.12.3, torch 2.14.0, torch-neuronx 2.14.3.0.1, neuronx-cc 2.28.7631.0 (as listed in `native_vs_aot_tp4_2026-10-25.md`).
- **Host runtime:** aws-neuronx-runtime-lib 2.35.30.0, collectives 2.35.43.0 (as listed in `native_vs_aot_tp4_2026-10-25.md`).
