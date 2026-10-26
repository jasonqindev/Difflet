# Four models, native vs AoT at TP4 on trn2.3xlarge, 2026-10-26

**Status:** phases 1 and 2 of the plan in `docs/superpowers/plans/2026-10-26-four-model-native-vs-aot-metrics.md`. All eight device cells (four models, native and AoT) have numbers:
- **AoT**, all four models, measured on the latest `origin/main`: Qwen-Image and LTX-2 in phase 1, HunyuanVideo 1.0 and Wan2.1 in phase 2.
- **LTX-2 native:** a full cell under cold definition A, measured in phase 2.
- **Qwen-Image, HunyuanVideo and Wan2.1 native:** numbers reused from earlier cells on this host. Phase 2 confirmed them at the current branch HEAD by bit-identical outputs (plan D10, option A).

PSNR / SSIM: **Qwen-Image and LTX-2 have numbers** (phase 4, two 50-step CPU fp32 references each, sections "Phase 4: Qwen-Image CPU fp32 references" and "Phase 4: LTX-2 CPU fp32 references", collected with the tool sanity checks and a look at the references in "Quality vs CPU fp32 reference (part 1)"). HunyuanVideo and Wan2.1 are **pending CPU references**. Cold e2e is definition A throughout. Definition B (native without its weight store, Wan2.1 and LTX-2 only) comes later (plan phase 6).

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

## Summary: four models x two backends (TP4, 50 steps, seed 42)

All values were re-read from the run records (`summary.json`, `stats_gen1-3.txt`, `cell_record.json`) on 2026-10-26 after phase 2.
- **DiT step:** median of the per-run medians over gen1-3, step 0 excluded.
- **Warm e2e:** mean of gen1-3, with gen0 discarded.
- **Cold e2e A:** page cache dropped. Compile-time weight artifacts are allowed: the native weight store and the AoT pre-shards.
- **Compile:** a cold `difflet compile` on a fresh root, with the checkpoint already in the page cache.
- **Peak HBM:** neuron-monitor's per-core peak.
- **PSNR / SSIM:** the device q1 output against a 50-step CPU fp32 reference (diffusers) started from that backend's own initial latents. RGB uint8, SSIM per Wang 2004; latent cosine on the final packed latents. Not a native-vs-AoT comparison: each backend has its own noise and its own reference.

| model | backend | DiT step | warm e2e | cold e2e A | compile | peak HBM per core | PSNR / SSIM | date, commit, evidence |
|---|---|---:|---:|---:|---:|---|---|---|
| HunyuanVideo 1.0, 320x512x61, g6.0 | native | 808.2 ms | 126.7 s | 457.8 s | 274.5 s | 17.95 GiB on the decoder core, 12.98 on the other three | pending CPU references | *reused*: 2026-10-25/26 at `eadddec`, `/home/ubuntu/bench-2026-10-26/hunyuan_video/s4_tp4/`. Confirmed by hash on 2026-10-26 at `4f5d46a`: `B/hunyuan_video/native_tp4_hashcheck/` |
| | AoT | 801.8 ms | 133.0 s | 617.5 s | 4526.7 s | 15.53 GiB | pending CPU references | 2026-10-26 at main `540e5f0`: `B/hunyuan_video/aot_main_tp4/` |
| Qwen-Image, 1024x1024, g4.0 | native | 359.0 ms | 82.3 s | 502.0 s | 187.0 s | 10.81 GiB on the DiT cores, 13.85 on the VAE core | 16.11 dB / 0.823 (latent cosine 0.872), against the CPU fp32 reference from native's noise | *reused*: 2026-10-25 at `4f6e1de`, `/home/ubuntu/bench-2026-10-25/qwen_image/native_tp4/`. Confirmed by hash on 2026-10-26 at `a87d89c`: `B/qwen_image/native_tp4_hashcheck/` |
| | AoT | 360.6 ms | 73.7 s | 515.6 s | 1209.9 s | 10.10 GiB | 21.48 dB / 0.912 (latent cosine 0.961), against the CPU fp32 reference from AoT's noise | 2026-10-26 at main `540e5f0`: `B/qwen_image/aot_main_tp4/` |
| Wan2.1 14B, 480x832x9, g5.0 (CFG) | native | 1063.6 ms | 128.5 s | 542.7 s (weight store) | 600.3 s | 15.15 GiB on the DiT cores, 18.40 on the VAE core | pending CPU references | *reused*: 2026-10-25 at `8b2408c`, `/home/ubuntu/bench-2026-10-25/wan2_1/native_tp4/`. Confirmed by hash on 2026-10-26 at `4f5d46a`: `B/wan2_1/native_tp4_hashcheck/` |
| | AoT | 956.4 ms | **123.16 s** with `--host-vae` (A2) | **449.4 s** with `--host-vae` (A2) | 5935.7 s (the transformer stage alone: 430.5 s) | 10.34 GiB | pending CPU references. The device-VAE frames are corrupted: 12.41 dB against a host fp32 decode | 2026-10-26 at main `540e5f0`: `B/wan2_1/aot_main_tp4_twocore/` and `…_hostvae/` |
| LTX-2, 480x704x49, g1.0 | native | 350.5 ms | 92.46 s | 594.76 s (Gemma-3 weight store) | 327.59 s | 16.51-16.52 GiB | 18.06 dB / 0.811 (latent cosine 0.930 video, 0.989 audio), against the CPU fp32 reference from native's noise | 2026-10-26: compile at `999c434`, the other runs at `009d3dd`. `B/ltx_2/native_tp4/` |
| | AoT | 400.1 ms | 66.55 s | 778.1 s | 1052.2 s | 9.81 GiB | 16.21 dB / 0.660 (latent cosine 0.725 video, 0.980 audio), against the CPU fp32 reference from AoT's noise. Far below the 2026-10-24 single-core 20-step run (cosine 0.991); not yet explained, see the LTX-2 phase 4 section | 2026-10-26 at main `540e5f0`: `B/ltx_2/aot_main_tp4_twocore/` |

**Native vs AoT**, computed as (native − AoT) / AoT from the unrounded run values:

| model | DiT step | warm e2e | cold e2e A | compile | peak HBM per core |
|---|---:|---:|---:|---:|---|
| HunyuanVideo 1.0 | +0.8% | −4.7% | −25.9% | −93.9% | +15.6% on the decoder core, −16.4% on the others |
| Qwen-Image | −0.4% | +11.6% | −2.6% | −84.5% | +7.0% on the DiT cores |
| Wan2.1 | +11.2% | +4.4% (against `--host-vae`) | +20.8% (against `--host-vae`) | −89.9% (+39.5% against the transformer stage alone) | +46.5% on the DiT cores |
| LTX-2 | −12.4% | +38.9% | −23.6% | −68.9% | +68.4% |

**What each comparison includes (details in the model sections)**
- **Placements differ between the backends** (plan D8):
  - Native encodes the Qwen-Image and HunyuanVideo prompts on rank 0's host; AoT encodes them on the device.
  - LTX-2 native runs Gemma-3 on the device, which is why its HBM is higher; LTX-2 AoT runs Gemma-3 on the host.
  - The Wan2.1 AoT e2e figures use `--host-vae`, an fp32 host decode, because main's compiled bf16 device VAE corrupts its frames (Wan2.1 section). Its compile figure is main's default compile, which includes 5486.1 s for that device-VAE stage.
- **Kept in fp32 on native** (plan decision 3, second round): Wan umT5, the Wan VAE and the Qwen-Image VAE.
- **Noise:**
  - Qwen-Image, HunyuanVideo and LTX-2 start from different seed-42 noise on the two backends, so their images and videos are not comparable pixel for pixel.
    - Qwen-Image and LTX-2 draw the noise in bf16 on both backends, and torch 2.14 and torch 2.9.1 give different bf16 draws.
    - HunyuanVideo native draws the noise in fp32, while AoT draws it in bf16.
  - Wan2.1 starts from identical noise on both backends.

## Method

**Code under test**
- **AoT:** `origin/main` at `540e5f086ffa5dae8e957aa94b1b23616091a7bc`, checked out as a separate detached worktree at `/home/ubuntu/aot-main`. The tree was clean in every run (`head: 540e5f0… dirty=0` in each `run.log`).
  - The five bf16 all-reduce fixes are ancestors of that commit: `fc6a580`, `b4bdf35` (Qwen-Image), `4d4cc60` (LTX-2), `7fe5308` (HunyuanVideo) and `1c8a64e` (Wan umT5). Each was checked with `git merge-base --is-ancestor`, exit 0.
  - The NxD venv `/home/ubuntu/venvs/difflet-neuron` was used with `PYTHONPATH=B/probe:/home/ubuntu/aot-main` and cwd `/home/ubuntu/aot-main`.
  - Before every run, the harness imports `difflet` with the run's exact env and refuses to start if `difflet.__file__` is not under `/home/ubuntu/aot-main` (`difflet_file_precheck`). The probe also records `difflet.__file__` in every process (`difflet_files_seen`).
  - Both fields read `/home/ubuntu/aot-main/difflet/__init__.py` in all 33 completed AoT runs: 14 in phase 1, then 7 for HunyuanVideo, 7 for Wan2.1 and 5 for the Wan2.1 `--host-vae` supplement in phase 2 (`B/<model>/summary.jsonl`). That rules out the venv's editable install, which points at another worktree.
- **Native:** branch `feat/hunyuan-neuron` in `/home/ubuntu/Difflet/.claude/worktrees/hunyuan-neuron`, with the native venv `/home/ubuntu/venvs/native-312`.
  - Every run uses `PYTHONPATH=B/probe:<worktree>`, the venv's `bin` on PATH, and `DIFFLET_COMPILE_CACHE=$R` plus `--cache-dir $R`.
  - The same `difflet_file_precheck` / `difflet_files_seen` check applies. In every native run of phase 2, both fields read `/home/ubuntu/Difflet/.claude/worktrees/hunyuan-neuron/difflet/__init__.py`.
  - The HEAD of each run is in its `run.log`, and the tree was clean (`dirty=0`) in all of them:
    - **LTX-2 cell:** the compile ran at `999c434`; cold, gen0-3 and q1 ran at `009d3dd`. The only difference between the two commits is plan files (`git diff --stat 999c434 009d3dd`: two files under `docs/superpowers/plans/`). Both commits contain F12 `05614a2`, the fix for the LTX-2 text-encoder index (`git merge-base --is-ancestor`, exit 0).
    - **Hash checks:** Qwen-Image at `a87d89c`, HunyuanVideo and Wan2.1 at `4f5d46a`.
  - `DIFFLET_NEURON_PG_TIMEOUT_S` was not set in any run: it is absent from every `env_extra`. The CLI default applies, which is 3 h.
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
- **Quality run (q1):** one more warm generate with lossless output: PNG for Qwen-Image, a frames tensor `out.pt` for the video models.
  - It also sets `DIFFLET_DUMP_LATENTS_DIR=<run>` and `DIFFLET_STEP_TIMES_FILE=<run>/steps.jsonl`. Both are supported on main: `difflet/utils/latent_dump.py` and `step_timing.py`, used in `qwen_image/pipeline.py` and `ltx_2/pipeline.py`.
  - Main has no latent dump for HunyuanVideo or Wan, and the native branch has none at all. For those cases, the q1 runs set the probe's opt-in `HV1_DUMP_LATENTS_DIR`.
    - On rank 0 (or in the stage process), it wraps the orchestrator's `_denoise`: `QwenImageOrchestrator`, `HunyuanVideoOrchestrator`, `WanOrchestrator` or `LTX2Orchestrator`.
    - It saves the latents entering the loop and the latents leaving it, and passes the arguments and results through unchanged.
    - The CPU self-tests in both venvs check it: `B/*/prep/probe_selftest_*.log`, all ok.
  - The dumped initial latents seed the CPU fp32 references (plan phase 4).
  - q1 is not part of the timing statistics.
- **Native cells:**
  - **LTX-2:** a full cell, run like the AoT cells: preread, compile on a fresh root `B/roots/ltx2_native_tp4`, cold, gen0, gen1-3, q1.
  - **Qwen-Image, HunyuanVideo and Wan2.1:** plan D10, option A. Each got one warm generate at the current HEAD, on a `cp -a` copy of the original cell's root. The copy was checked identical with `diff -r` before the run, and the original root was listed (size and mtime) before and after the run, unchanged.
    - **Pass rule:** `work/latents.pt` must be bit-identical to the reference cell. So must the PNG (Qwen-Image) or the mp4 re-encoded from the lossless `out.pt` with the native CLI's own writer (the video models).
    - If the rule passes, the reference cell's timings stand. The hash-check run's own wall time is not a timing figure (see the model sections).

**Metrics**
- **DiT step** = the median / p10 / p90 of the deltas between consecutive denoising steps, with step 0 excluded (49 deltas per run).
  - Qwen-Image: from the generate stage process (`QwenImageOrchestrator._scheduler_step` stamps).
  - LTX-2: from the generate process. One stamp per video+audio pair of `_scheduler_step` calls, checked in the probe self-test: 6 calls give 3 stamps.
  - The table value is the median of the per-run medians over gen1–3; p10 / p90 are pooled over gen1–3.
  - q1's `steps.jsonl`, written by main's own step timer, agrees: Qwen-Image 361.0 ms, LTX-2 400.8 ms (medians of 50 rows).
- **e2e** = the process wall of `python -m difflet.cli.main generate` (harness clock).
- **Cold-e2e breakdown** = probe spans in the cold run's `summary.json`.
- **Native DiT step:** from rank 0 of the generate launch, stamped the same way (one stamp per video+audio pair for LTX-2).
- **Peak HBM** = neuron-monitor's per-core peak of `memory_used` (the sum of its categories) from `nmon.jsonl`. The AoT stages are not torchrun ranks, so there is no `max_memory_allocated`. For native runs, rank 0's `torch_neuronx` `max_memory_allocated` is also recorded.
- **Outputs:** SHA-256 of every output and dump. The image or a frame strip was opened and checked against the prompt.

**Supervision**
- One device job at a time. Before each launch, the harness waited until `fuser /dev/neuron0` showed no process. After each run it confirmed the device was released: `device_released: true` in every `summary.json` of phases 1 and 2.
- Phase 1: the jobs ran under `scripts/supervise.sh` (Qwen-Image) and as a `systemd-run --user` unit (LTX-2 AoT).
- Phase 2: every job ran as a `systemd-run --user` unit, one after another:
  - `qwen_native_hashcheck`, 10:28 UTC;
  - `hv_aot_4m`, 10:46-12:29;
  - `wan_aot_4m`, 12:38-14:44;
  - `wan_aot_hostvae_4m`, from 14:51;
  - `hvwan_native_4m`, 15:20-15:41;
  - `ltx_native_4m`, 15:53:58-16:29:46, pre-read included.
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

**Native cell:** *reused* from the 2026-10-25 cell `native_tp4`, measured at `4f6e1de` (`/home/ubuntu/bench-2026-10-25/qwen_image/native_tp4/`). Phase 2 confirmed it by hash at the current HEAD (plan D10, option A; details below).

| metric | AoT on main (`aot_main_tp4`) | native | native vs AoT | AoT before the bf16 fix (2026-10-25, `4f6e1de`, *reused*) |
|---|---|---|---|---|
| DiT step median / p10 / p90 | **360.6** / 360.2 / 361.8 ms (runs 360.6, 360.9, 360.5) | **359.0** / 358.2 / 360.1 ms (*reused*; the hash check at HEAD gave 358.6 / 357.7 / 360.7) | −0.4% | 418.2 ms (now −13.8%) |
| compile cold, checkpoint in the page cache | **1209.9 s** (text stage 128.7, generate stage 660.5, vae stage 401.1) | **187.0 s** (*reused*) | −84.5% | 1238.5 s (−2.3%) |
| e2e warm, n=3 (gen0 discarded) | **73.7 s** mean (74.16, 76.06, 70.98; median 74.2; gen0 72.9) | **82.3 s** (81.42, 82.13, 83.41; *reused*) | +11.6% | 75.4 s (−2.2%) |
| e2e cold, page cache dropped | **515.6 s** (breakdown below) | **502.0 s** (*reused*) | −2.6% | 504.2 s (+2.3%) |
| peak HBM per core (neuron-monitor) | **10.10 GiB** on all 4 cores in gen0–3 (10.04 in cold and q1) | **10.81 GiB** on the three DiT cores, 13.45–13.85 on the VAE core (*reused*) | +7.0% on the DiT cores | 10.29 GiB |
| output | PNG `506c09f4…`, `work/latents.pt` `35e26fb7…`, identical in cold, gen0–3 and q1 | PNG `b1906d53…`, latents `4da4c89e…` in the reused cell and in the HEAD hash check | different noise (torch 2.14 vs 2.9 bf16 draw) | PNG `e24798fe…` |
| logs | `B/qwen_image/aot_main_tp4/{compile,cold,gen0..gen3,q1}/` | `/home/ubuntu/bench-2026-10-25/qwen_image/native_tp4/`; hash check `B/qwen_image/native_tp4_hashcheck/` | | |

*Native vs AoT* is (native − AoT) / AoT, as in the 2026-10-25 report.

**The native numbers (*reused*, confirmed by hash at HEAD)**
- They come from `native_vs_aot_tp4_2026-10-25.md`, cell `native_tp4`, measured on 2026-10-25 05:36–05:59 UTC at `4f6e1de` on this host, with the native venv listed under Toolchain. Each figure was re-read from that cell's `summary.json`: compile 187.0 s, cold 501.99 s, gen1–3 81.42 / 82.13 / 83.41 s, peak HBM 10.81 GiB on cores 0, 1 and 3.
- The cell has no weight store (bf16 checkpoint, "no gain"), so it has one cold number, under definition A and B alike.
- **Hash check** (2026-10-26 10:28 UTC, HEAD `a87d89c`, clean; run dir `B/qwen_image/native_tp4_hashcheck/q1`, record `hashcheck.json`):
  - **How it ran:** one warm generate with the flags of the 2026-10-25 gen3 run, on a `cp -a` copy of the original root (`B/roots/qwen_image_native_tp4_copy1025`). The page cache was dropped and the checkpoint read twice first (100% resident).
  - **Result:** `out.png` `b1906d53…` and `work/latents.pt` `4da4c89e…` are bit-identical to every run of the reference cell, so the timings and HBM above stand. I opened the PNG: a red fox trotting through snow between snow-covered trees.
  - **Step:** the rank-0 step is 358.6 / 357.7 / 360.7 ms, against 359.0 / 358.2 / 360.1 ms in the reference cell.
  - **Its 117.2 s wall is not a warm figure.** One NEFF was compiled inside the run (`neff/39/1e/391e973e…`, neuronx-cc 20.2 s). That NEFF equals the existing `cde0c8b3…` except for ten backend debug blobs of the same length, so the compiled program is unchanged. Why the cache key changed was not found (*inference*: possibly the `attention.py` edit since `4f6e1de`).
  - **Noise:** the probe dumps `initial.pt` `95a4306b…`, packed (1,4096,64) bf16, drawn on rank 0's CPU. It equals the torch 2.14 CPU draw `pack(torch.randn((1,1,16,128,128), generator=Generator().manual_seed(42), dtype=bf16))`. `final.pt` `94d160cb…` equals `work/latents.pt`.
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

**Native cell:** `native_tp4`, a full cell under definition A, measured in phase 2 (subsection "LTX-2 native" below). The earlier native measurements in `ltx_2_neuron.md` (2026-10-18/20) were taken on another instance with an older toolchain, and their raw evidence is not on this host, so they were not reused; they are compared below.

| metric | AoT on main, two-core (`aot_main_tp4_twocore`) | native | native vs AoT |
|---|---|---|---|
| DiT step median / p10 / p90 | **400.1** / 398.1 / 400.9 ms (runs 398.2, 400.3, 400.1) | **350.5** / 349.4 / 352.0 ms (runs 350.9, 350.5, 350.1) | −12.4% |
| compile cold, checkpoint in the page cache | **1052.2 s** (model build 542.2, pre-sharding 470.5) | **327.59 s** (DiT load 87.5, Gemma-3 load and store write 197.4) | −68.9% |
| e2e warm, n=3 (gen0 discarded) | **66.55 s** mean (66.00, 68.00, 65.64; median 66.0; gen0 66.1) | **92.46 s** mean (94.40, 92.19, 90.78; median 92.19; gen0 90.95) | +38.9% |
| e2e cold, page cache dropped | **778.1 s** (breakdown below) | **594.76 s**, definition A: Gemma-3 from its bf16 weight store (breakdown below); B later | −23.6% |
| peak HBM per core (neuron-monitor) | **9.81 GiB** on all 4 cores (cold, gen2, gen3, q1; 9.75 in gen0 and gen1) | **16.51–16.52 GiB** on all 4 cores in every run; rank 0 `max_memory_allocated` 15.66 GiB | +68.4%: Gemma-3 is on the device |
| output | `out.mp4` `5aab9e4e…`, identical in cold, gen0–3; q1 `out.pt` `1a875738…` | `out.mp4` `cd11731b…`, identical in cold, gen0–3; q1 `out.pt` `33fa80c2…` | different noise (torch 2.14 vs 2.9.1 bf16 draw) |
| logs | `B/ltx_2/aot_main_tp4_twocore/{compile,cold,gen0..gen3,q1}/` | `B/ltx_2/native_tp4/{compile,cold,gen0..gen3,q1}/` | |

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

### LTX-2 native (phase 2, cell `native_tp4`)

**Runs:** 2026-10-26, 15:53:58 to 16:29:46 UTC, pre-read included. The cell ran as the systemd user unit `ltx_native_4m` on the fresh root `B/roots/ltx2_native_tp4`.
- **HEAD:** the compile ran at `999c434`; cold, gen0-3 and q1 ran at `009d3dd`. The two commits differ only in plan docs, and both contain F12 (`05614a2`). Every `run.log` shows `dirty=0`.
- **Outcome:** all 7 runs returned rc 0 and released the device. Every process imported difflet from the hunyuan-neuron worktree.
- **No Difflet bug** came up, so there is no code commit.

**Placement:** the native default, recorded from the rank argv in `cold/probe/*-rank*.json`, which shows `--device-text-encoder`.
- **Gemma-3:** on the device in bf16, TP4 beside the DiT.
- **Connectors:** bf16 on rank 0's host.
- **Video VAE, audio VAE and vocoder:** decoded on the host, in bf16, in the driver process after the generate launch.
- **Command:** `--backend neuron --tp-degree 4`, plus the harness flags: 480x704x49, `--steps 50 --guidance-scale 1.0 --seed 42`, the prompt, and `HF_HUB_OFFLINE=1`.

**Pre-read before the compile** (`cell_record.json` `preread_rec`)
- The checkpoint is the same 85.71 GiB in 23 files as the AoT cell's.
- After the drop, `Cached` was 0.29 GiB and the checkpoint 0% resident.
- It was then read twice, in 702.3 s and 39.4 s. Afterwards it was 100% resident, with `Cached` at 86.16 GiB.

**Compile, 327.59 s** (`compile/run.log`; rank-0 spans and component phases in `cell_record.json`)
- **DiT load, 87.5 s:**
  - checkpoint 5.22 s, from the page cache;
  - warm-up forward 59.64 s, which includes the block NEFF compile.
  - The weight store logged `not written: no gain` for the DiT on every rank: its checkpoint is already bf16 (9.96 GB either way).
- **Gemma-3 load, 197.4 s:**
  - checkpoint read plus store write 181.64 s, then warm-up 15.24 s.
  - Every rank logged `[weight store] rank r/4 NeuronLtx2TextEncoderApplication: written 5.88 GB in 118.7–178.4 s` into `neuron/weights/Lightricks--LTX-2__text_encoder__47da56e2__bfloat16__tp4__979a2453cd7057f8`.
- **Blocks:** every rank logged `compiled_blocks=96`, with no load or generate fallbacks.

**F12 confirmed on the device** (store manifests `*.manifest.json` in that directory)
- The load report of every manifest has `missing: 0`.
- The store holds 626 tensors: `replicated` 289, plus 337 in each `tpN-of-4`. These cover the 624 layer tensors, `embed_tokens` and `norm`.
- The 439 checkpoint keys not loaded (`unexpected`) are `vision_tower` (437) and `multi_modal_projector` (2). These are the skipped prefixes.

**Cold e2e A, 594.76 s** (`cold/summary.json`; seconds from driver start)
- Before the run, `Cached` was 0.27 GiB and the checkpoint 0% resident.

| part | at | duration |
|---|---:|---:|
| driver start to rank start (torchrun launch) | 0 | 14.9 |
| rank imports and runtime set-up | 14.86 | 14.2 |
| DiT load (bf16 checkpoint read 286.55, build on meta 9.87, warm-up forward 12.86) | 29.01 | **310.2** |
| Gemma-3 load from the weight store (store read 178.78, warm-up 1.36; every rank logged `hit 5.88 GB` in 175.6–178.4 s) | 340.28 | **180.2** |
| prompt encode (connectors 24.95) | 520.55 | 25.3 |
| denoise, 50 steps | 545.88 | 17.56 |
| ranks exit / generate stage ends | 563.5 / 568.3 | |
| host decode (pipeline load 3.01, decode 19.3, save 0.73) | 568.31 | 25.41 |
| driver exit / **harness wall** | 593.73 | **594.76** |

- **Connectors:** they take 24.95 s here but 4.9 s in the warm runs, although their load span is only 0.06 s. *Inference:* their weights are read from disk on first use.
- **Against AoT's cold run (778.1 s):** AoT spends 373.8 s loading its host pipeline (Gemma-3 from its fp32 checkpoint, the connectors and the VAEs) and 309.7 s loading the DiT pre-shards. Native's DiT load takes about as long (310.2 s, of which the bf16 checkpoint read is 286.55 s). It reads Gemma-3 from its bf16 store, 5.88 GB per rank, in 180.2 s.

**Warm e2e, by part** (gen1, seconds from process start)

| part | native (`native_tp4/gen1`, wall 94.40) | AoT (`aot_main_tp4_twocore/gen1`, wall 66.00) |
|---|---|---|
| start-up | ranks start at 2.8; DiT load starts at 25.5 | imports 4.7 |
| text encoder load | Gemma-3 4.4 (store 3.0) | host pipeline 5.0 |
| DiT load | 17.9 (checkpoint 5.3, warm-up 9.2) | 11.6 |
| prompt encode | 5.3 | 10.6 |
| denoise | 17.55 | 19.9 |
| decode and save | stage ends at 74.5; host decode 19.0 (pipeline load 2.6, decode 13.6, save 0.5) | decode 9.4, save 1.2 |
| process exit | 93.5 | 62.5 |

- **Where native's 26 s go.** Native's denoise is 2.4 s faster, but it spends more on two things:
  - the launch: 22.7 s from rank start to the start of the DiT load. Across gen0-3 this was 19.2–22.7 s, against 14.2 s in the cold run. *Inference:* this is the torchrun launch jitter recorded in `neuron_final.md`.
  - the separate host decode: 19.0 s, against AoT's 10.6 s.

  This is the same split `ltx_2_neuron.md` reported.

**Two-core attention in the native NEFFs** (`B/ltx_2/native_tp4/neff_attn_scan.json`)
- 5 of the 90 NEFFs in the root contain `attention_cte`. All 5 split into `sg00:attention_cte_lnc0` and `sg01:attention_cte_lnc1`, and no attention NEFF has a single subgraph.
- The newest NEFF is from 16:11:43 UTC, before the cold run started (16:11:50). So nothing was compiled during the cold or warm runs.

**Outputs and q1** (`q1_checks/checks.json`)
- `out.mp4` `cd11731b…` is identical in cold and gen0-3, and `work/latents.pt` `b9c745bd…` is identical in all 6 generates.
- q1's `out.pt` `33fa80c2…` is (1,49,3,480,704) bf16 in [0,1]. Re-encoding it with the native `_save_video` gives `cd11731b…`, so q1 produced exactly the frames of the timed runs.
- **Probe dumps** (rank 0, `HV1_DUMP_LATENTS_DIR`, wrapper on `LTX2Orchestrator._denoise`):

  | dump | file sha256 | raw sha256 | shape |
  |---|---|---|---|
  | initial_video | `16e09464…` | `55b90133…` | (1,2310,128) bf16 |
  | initial_audio | `411c72d9…` | `24900b1b…` | (1,51,128) bf16 |
  | final_video | `c218a480…` | `887027a8…` | (1,2310,128) bf16 |
  | final_audio | `b868488a…` | `fac8052a…` | (1,51,128) bf16 |

  - The final dumps equal `work/latents.pt` bit for bit.
  - The initial noise is drawn on rank 0's CPU in bf16. A torch 2.14 CPU draw reproduces it bit for bit: `Generator().manual_seed(42)`, bf16 `randn` (1,128,7,15,22) packed, then (1,8,51,16) packed (`q1_checks/cpu_noise_seed42.pt`).
- **Noise against AoT:** native's noise differs from AoT's q1 noise (`initial_video` `4458e4cf…`, `initial_audio` `8e52babb…`).
  - The dtype and shapes are the same on both sides, so the difference is the torch version: 2.14.0 against 2.9.1.
  - The first video element is 1.9297 on native and −0.8086 on AoT, the values `ltx_2_neuron.md` recorded.
  - Final latents against AoT: cosine 0.363 (video) and 0.641 (audio), as expected from different noise.
- **Content:** I opened the mid frame and the strip of frames 0, 12, 24, 36 and 48 (`q1/strip_0_12_24_36_48.png`).
  - It shows a snowy conifer forest with a red-orange, fox-like animal running across the snow.
  - It is more recognisable than the AoT output, but blurred and distorted, with a cyan cast on the snow.
  - *Inference:* this is guidance 1 (no CFG), as for AoT. The CPU reference (phase 4) will settle it.

**Against the earlier native numbers** (`ltx_2_neuron.md`, 2026-10-19/20, *reused*: another instance, an older toolchain, no raw logs on this host)

| metric | earlier | now | change |
|---|---:|---:|---:|
| DiT step | 350.7 ms | 350.5 ms | −0.1% |
| warm e2e | 90.8 s | 92.46 s | +1.8% |
| cold e2e (with the weight store) | 588.3 s | 594.76 s | +1.1% |
| compile | 306.6 s | 327.59 s | +6.8% |

## HunyuanVideo 1.0

**Model:** `hunyuanvideo-community/HunyuanVideo` at revision `e8c2aaa66fe3742a32c11a6766aecbf07c56e773`.  
**Workload:** 320x512x61, 50 steps, `--guidance-scale 6.0` (embedded guidance: one DiT forward per step), seed 42.

**AoT cell:** `aot_main_tp4`, root `B/roots/hunyuan_video_aot_main_tp4`, run on 2026-10-26 from 10:46:43 to 12:29:13 UTC as unit `hv_aot_4m`, at main `540e5f0`, clean. Main's own placement uses three stage processes:
- CLIP on one core;
- Llama on the device at TP4;
- then the DiT at TP4, with the compiled bf16 VAE replicated on the four ranks.

**Native cell:** *reused* from `s4_tp4` (`hunyuan_video_neuron.md`, measured 2026-10-25 23:42 to 2026-10-26 00:25 UTC at `eadddec`, raw evidence under `/home/ubuntu/bench-2026-10-26/hunyuan_video/s4_tp4/`). Phase 2 confirmed it by hash at `4f5d46a` (below). Native's placement:
- Llama and CLIP encode on rank 0's host;
- the DiT runs at TP4;
- the compiled bf16 decoder runs on rank 0's core inside the generate launch.

| metric | AoT on main (`aot_main_tp4`) | native (`s4_tp4`, *reused*) | native vs AoT | AoT before the bf16 fix (2026-10-25, `d6ffb9a`, *reused*) |
|---|---|---|---|---|
| DiT step median / p10 / p90 | **801.8** / 800.9 / 802.5 ms (runs 801.8, 802.0, 801.2) | **808.2** / 807.0 / 810.7 ms (the hash check at HEAD gave 808.7 / 807.5 / 810.0) | +0.8% | 813.9 ms (now −1.5%) |
| compile cold, checkpoint in the page cache | **4526.7 s** (clip stage 41.4, llama stage 163.6, generate stage 4315.7) | **274.5 s** | −93.9% | 4347.7 s, a lower bound: it reused pre-shards |
| e2e warm, n=3 (gen0 discarded) | **133.0 s** mean (134.67, 131.73, 132.59; median 132.6; gen0 129.2) | **126.7 s** (129.18, 130.65, 120.26; gen0 130.5) | −4.7% | 132.9 s (n=2) |
| e2e cold, page cache dropped | **617.5 s** (breakdown below) | **457.8 s** (load 356.0, of which the DiT checkpoint 194.8 and host Llama/CLIP 123.3; includes a 6.4 s neuronx-cc on rank 3) | −25.9% | 617.3 s |
| peak HBM per core (neuron-monitor) | **15.53 GiB** on each of the 4 cores in every generate | **17.95 GiB** on the decoder core (DiT shard + decoder), 12.98 on the other three | +15.6% / −16.4% | 15.52 GiB |
| output | mp4 `334301fc…` in cold and gen0-3; q1 `out.pt` `33a8a130…` | mp4 `edf8d6be…`, latents `b62d0891…` | different noise (native fp32 draw, AoT bf16 draw) | `201e5ef7…` |
| logs | `B/hunyuan_video/aot_main_tp4/{compile,cold,gen0..gen3,q1}/` | `/home/ubuntu/bench-2026-10-26/hunyuan_video/s4_tp4/`; hash check `B/hunyuan_video/native_tp4_hashcheck/` | | `hunyuan_video_neuron.md` |

**AoT compile, 4526.7 s** (`compile/run.log`, `cell_record.json`)
- **Pre-read:** 39.01 GiB, read in 318.6 s and then 15.4 s, 100% resident.
- **Stages:** clip stage 41.4 s (build 20.8), llama stage 163.6 s (build 148.5, of which neuronx-cc 75.7), generate stage 4315.7 s.
- **Inside the generate stage:**
  - DiT build 421.5 s, of which neuronx-cc 171.3 s;
  - DiT pre-sharding 208.0 s (44.4 GB), the sharding the 2026-10-25 compile did not pay;
  - VAE build 3679.1 s, of which neuronx-cc 3571.4 s. **The VAE is 81% of the compile.**

**AoT cold e2e, 617.5 s** (`cold/summary.json`; `Cached` 0.27 GiB before)
- clip stage 26.4 s (load 7.0);
- llama stage 134.6 s (load 129.3);
- generate stage 448.3 s: DiT and VAE pre-shard load 385.0, denoise 40.3, VAE decode 17.05, save 0.96.

**AoT warm gen1, 134.67 s:** clip stage 12.2, llama stage 16.4 (load 11.0), generate stage 100.7 (app load 37.3, denoise 40.3, decode 17.05, save 0.76).

**Two-core attention and bf16** (`hlo_nki_grid.jsonl`, `hlo_allreduce.jsonl`)
- **Two-core:** main's CLI sets `_VIRTUAL_CORE_SIZE = 2` (`hunyuan_video.py:31`). A parse-only capture shows `NEURON_RT_VIRTUAL_CORE_SIZE=2` on the clip, llama and generate stages. In the DiT module `MODULE_f4b5a7e3…`, all 60 attention calls are `attention_cte_lnc0_de6532dd…` with grid 2.
- **bf16:** the DiT has 124 all-reduces, all bf16: 40 on `[1,10240,3072]`, 40 on `[1,10496,3072]` and 44 on `[1,256,3072]`. The two Llama graphs have 64 + 64 bf16 all-reduces. The VAE (`MODULE_f27c3584…`) and CLIP have none.
- **Native two-core:** native's NEFFs carry the `attention_cte_lnc0` / `lnc1` split (plan D1).

**AoT q1**
- `out.pt` `33a8a130…` is (1,3,61,320,512) bf16, the raw VAE output in [−1.27, 1.04]. Re-encoding it with main's `_save_video` gives `334301fc…`, the timed runs' mp4.
- **Probe dumps:** main has no HunyuanVideo latent dump, so the probe dump was used. It saved `initial.pt` `e13a3f46…` and `final.pt` `ec23d9af…`, both (1,16,16,40,64) bf16.
- **AoT noise on CPU:** in torch 2.9.1, `torch.manual_seed(42); torch.randn(1,16,16,40,64, dtype=torch.bfloat16)` equals `initial.pt` bit for bit (`q1_noise_check.json`).
- **Content:** I opened `cold/mid.png` and `q1/mid.png`. Both show a red fox running across snow, with a snowy conifer forest behind it.

**Native hash check** (2026-10-26 15:20 UTC, HEAD `4f5d46a`, clean; unit `hvwan_native_4m`; `B/hunyuan_video/native_tp4_hashcheck/`)
- **How it ran:** one warm generate on a `cp -a` copy of the `s4_tp4` root. `diff -r` showed it identical before the run, and the original root's listing was unchanged after.
- **Result:**
  - `work/latents.pt` `b62d0891…` is bit-identical to every `s4_tp4` run.
  - The lossless `out.pt` `a17c9714…` ((1,3,61,320,512) bf16) re-encodes with the native `_write_video` to `edf8d6be…`, the reference mp4.
  - So the reused numbers stand.
- **Noise:** the noise entering the loop is fp32 (1,16,16,40,64). It equals the CPU draw `torch.randn(…, generator=Generator().manual_seed(42), dtype=float32)` bit for bit. It differs from AoT's bf16 draw.
- **Step and HBM:** 808.7 / 807.5 / 810.0 ms; neuron-monitor 17.89 GiB on the decoder core and 12.92 on the others.
- **Its 156.1 s wall is not a warm figure.** Nothing was compiled: the root copy gained no files. Start-up and load were slower: the stage started 11.5 s later, and the app load took 56.3 s against 42.1 s. *Inference:* the page-cache drop before the pre-read also evicted venv and NEFF files that the reference warm runs had cached.

**Main's issue #110** (warm e2e up from 109.8 to 123.2 s after the bf16 fix, n=1) does not reproduce at this workload: warm is 133.0 s here, against 132.9 s before the fix.

## Wan2.1 14B

**Model:** `Wan-AI/Wan2.1-T2V-14B-Diffusers` at revision `38ec498cb3208fb688890f8cc7e94ede2cbd7f68` (DiT and umT5 checkpoints in fp32).  
**Workload:** 480x832x9, 50 steps, `--guidance-scale 5.0` (CFG: two DiT forwards per step), seed 42.

**AoT cell:** `aot_main_tp4_twocore`, root `B/roots/wan21_aot_main_tp4_twocore`, run on 2026-10-26 from 12:38 UTC as unit `wan_aot_4m`, at main `540e5f0`, clean.
- `NEURON_RT_VIRTUAL_CORE_SIZE=2` was exported for the compile and every generate:
  - main's Wan CLI sets `_VIRTUAL_CORE_SIZE = None`; only ring CP sets 2;
  - `run_stage` passes the exported value to both the transformer stage and the vae stage (parse capture, `B/wan2_1/prep/parse_check_vcs2.json`).
- **Placement** (main's default): a transformer stage runs umT5 and the DiT at TP4, then a vae stage runs the compiled bf16 VAE on one core.
- **Supplement A2** (`aot_main_tp4_twocore_hostvae`, unit `wan_aot_hostvae_4m`, from 14:51 UTC): the same root with `--host-vae`, no compile. The driver decodes on the host in fp32, which is the placement of the 2026-10-25 AoT cells.

**Native cell:** *reused* from `native_tp4` (`native_vs_aot_tp4_2026-10-25.md`, measured 2026-10-25 07:00–07:36 UTC at `8b2408c`, raw evidence `/home/ubuntu/bench-2026-10-25/wan2_1/native_tp4/`). Phase 2 confirmed it by hash at `4f5d46a` (below). Native's placement:
- umT5 runs in fp32 on the device, TP-sharded beside the DiT;
- the DiT is read from the bf16 weight store;
- the VAE decodes in fp32 in a one-core launch of its own.

| metric | AoT on main, device bf16 VAE (`aot_main_tp4_twocore`) | AoT A2, `--host-vae` (`…_hostvae`) | native (`native_tp4`, *reused*) | native vs AoT A2 | AoT two-core before the bf16 fix (2026-10-25, `8b2408c`, `--host-vae`, *reused*) |
|---|---|---|---|---|---|
| DiT step median / p10 / p90 | **956.4** / 955.3 / 958.0 ms (runs 956.4, 955.8, 956.9) | 956.8 / 956.0 / 958.2 ms | **1063.6** / 1061.5 / 1065.9 ms (the hash check at HEAD gave 1063.8 / 1062.2 / 1067.3) | +11.2% | 955.8 ms |
| compile cold, checkpoint in the page cache | **5935.7 s** (transformer stage 430.5, VAE stage 5486.1) | none: same root | **600.3 s** (generate launch 428.8, vae launch 169.7) | −89.9% (+39.5% against the transformer stage alone) | 445.1 s, no VAE stage |
| e2e warm, n=3 (gen0 discarded) | 105.99 s (106.97, 104.34, 106.65; gen0 105.61). **The frames are corrupted** | **123.16 s** (123.04, 123.43, 123.00; gen0 123.9) | **128.5 s** (125.98, 130.82, 128.83; gen0 129.4) | +4.4% | 120.0 s |
| e2e cold, page cache dropped (A) | 432.3 s. **The frames are corrupted** | **449.4 s** | **542.7 s** (DiT from the store 217.0, umT5 checkpoint 172.5) | +20.8% | 447.5 s |
| peak HBM per core (neuron-monitor) | **10.34 GiB** on each core | 10.34 GiB (10.37 in gen0) | **15.15 GiB** on the DiT cores; VAE core 18.12–18.40 | +46.5% on the DiT cores | 10.37 GiB |
| output | mp4 `941daa35…` in cold and gen0-3; latents `5ae9311e…` in all 6 generates | mp4 `60463ea2…`, latents `5ae9311e…` | mp4 `9b914e81…`, latents `3cafa271…` | same noise | mp4 `d2ecce24…` |
| logs | `B/wan2_1/aot_main_tp4_twocore/{compile,cold,gen0..gen3,q1}/` | `B/wan2_1/aot_main_tp4_twocore_hostvae/{cold,gen0..gen3}/` | `/home/ubuntu/bench-2026-10-25/wan2_1/native_tp4/`; hash check `B/wan2_1/native_tp4_hashcheck/` | | `native_vs_aot_tp4_2026-10-25.md` |

**Which AoT e2e figures are valid.** The headline AoT e2e figures are A2's (`--host-vae`), because the device-VAE frames are corrupted (below). With the device VAE, AoT is 17.2 s faster warm and 17.1 s faster cold, but it decodes wrong frames. The compile column is main's default compile, which builds that device VAE. The transformer stage alone takes 430.5 s, and A2 needs only that stage's artifacts.
- *Not checked:* whether `difflet compile --host-vae` on main skips the VAE stage.

**AoT compile, 5935.7 s** (`compile/run.log`, `cell_record.json`)
- **Pre-read:** 74.86 GiB in 18 files, read in 612.1 s and then 28.5 s, 100% resident.
- **Transformer stage, 430.5 s:**
  - umT5 build 35.8 s (neuronx-cc 6.8), pre-shard 5.7 s / 10.8 GB;
  - DiT build 252.7 s (neuronx-cc 62.0), pre-shard 114.2 s / 28.6 GB.
- **VAE stage, 5486.1 s:** neuronx-cc 5468.4 s.

**AoT cold breakdowns** (`Cached` 0.26 GiB before each)
- **Device VAE, 432.3 s:** transformer stage 394.8 (pre-shard load of umT5 + DiT 324.6, encode 0.64, denoise 48.0), vae stage 31.3 (load 24.15, decode 1.4, save 0.86).
- **A2, 449.4 s:** pre-shard load 321.9, denoise 48.0, host decode 50.2.

**AoT warm gen1 breakdowns**
- **Device VAE, 106.97 s:** transformer stage 72.1 (load 18.4, denoise 47.98), vae stage 31.43 (load 24.49, decode 1.40).
- **A2, 123.04 s:** transformer stage 70.2 (load 16.0, denoise 48.0), host decode 49.1, save 0.6.

**Two-core attention and bf16** (`hlo_nki_grid.jsonl`, `hlo_allreduce.jsonl`)
- **DiT module `MODULE_3c8d124d…`:**
  - All 80 attention calls are `attention_cte_lnc0_*` with grid 2.
  - It is neither the single-core stock module (`MODULE_c984433f`, by name from the 2026-10-25 logs; that root was deleted under decision 6) nor the 2026-10-25 two-core module (`MODULE_5ae08501`).
  - 120 bf16 all-reduces on `[1,4680,5120]`. The only f32 ones are on `[N,1]` shapes: 120 on `[1,4680,1]` and 40 on `[1,512,1]`. *Inference:* these are q/k norm statistics.
- **umT5 `MODULE_f0822b32…`:** 24 bf16 all-reduces, no f32.
- **VAE `MODULE_5dd36239…`:** no all-reduces and no NKI calls.

**Finding: main's compiled bf16 Wan VAE corrupts its frames on the device**
- **What it looks like:** the device-VAE output shows the fox only as a silhouette under coloured block noise (`cold/mid.png`). After frame 0, the frame std is 15–52 grey levels, against 62–81 for a host decode.
- **The latents are fine:** I decoded q1's `work/latents.pt` (`5ae9311e…`) on the CPU with diffusers' `AutoencoderKLWan`, in fp32 and in bf16. Both decodes are clean and on prompt (`q1_host_vae/host_{fp32,bf16}_mid.png`).
- **PSNR of the device frames** (`q1/out.pt` `4380f37c…`) against the fp32 host decode of the same latents (`q1_host_vae/host_fp32_vs_device.json`): mean 12.41 dB, worst 9.36 dB. Per frame: 21.4, 12.67, 10.99, 9.36, 11.05, 12.17, 11.29, 11.04, 11.71.
- **Runtime VCS is not the cause:** re-running only the vae stage on q1's latents gives `out.pt` `4380f37c…` bit for bit, with `NEURON_RT_VIRTUAL_CORE_SIZE=2` and with it unset (`vae_vcs_ab/`, `vae_vcs_ab.log`).
- **Compile-time VCS is inferred not to be the cause, not tested.** A test would need another 1.5 h compile. The evidence:
  - the VAE HLO has no NKI calls, and only the attention ops read the variable;
  - the 2026-10-24 Wan2.2 bf16 device-VAE run (`/home/ubuntu/campaign/bf16-reduce/runs/wan22/bf16/warm1`, no VCS export; mid frame copied to `ref_wan22_devvae/`) shows the same kind of corruption.
- **Status:** a defect in `origin/main`'s compiled Wan VAE (`trainium/wan/vae.py`), recorded as a follow-up. `aot-main` was not modified.
- **Native is not affected:** its VAE is a different, eager fp32 path.

**AoT q1 and noise**
- `out.pt` `4380f37c…` is (1,3,9,480,832) bf16, the corrupted device frames. Re-encoding it with main's `_save_video` gives `941daa35…`, the timed runs' mp4.
- **Probe dumps:** main has no Wan latent dump, so the probe dump was used. It saved `initial.pt` `03b6284b…` (fp32, raw sha `16aeaed7…`) and `final.pt` `c7332d35…`, which equals `work/latents.pt`.
- **One noise for both backends:** `initial.pt` equals the CPU draw `torch.randn((1,16,3,60,104), generator=Generator().manual_seed(42), dtype=float32)` in torch 2.9.1 and in torch 2.14 (`q1_noise/`). The native hash check's dump is the same file (`03b6284b…`). So both backends start from identical noise, and a single CPU reference covers both.
- **Final latents compared** (rel-L2 and cosine, `latents_vs_1025.json`):
  - against the 2026-10-25 AoT two-core run: rel-L2 0.178, cosine 0.984;
  - against native `3cafa271…`: rel-L2 0.216, cosine 0.977;
  - for scale, the 2026-10-25 native against the 2026-10-25 AoT: 0.245 / 0.970.
- **Content:** A2's mp4 `60463ea2…` is clean and shows a red fox in a snowy forest.

**Native hash check** (2026-10-26 15:30 UTC, HEAD `4f5d46a`, clean; unit `hvwan_native_4m`; `B/wan2_1/native_tp4_hashcheck/`)
- **How it ran:** one warm generate on a `cp -a` copy of the 2026-10-25 root (`B/roots/wan21_native_tp4_copy1025`). Before the run, the harness's new `--preread-files` read exactly what the run reads: the umT5 and VAE checkpoint files plus the copy root's store, 48.25 GiB, 100% resident.
- **Result:**
  - `work/latents.pt` `3cafa271…` is bit-identical to the reference cell.
  - The lossless `out.pt` `8d8a1779…` ((1,3,9,480,832) fp32) re-encodes with the native `_save_video` to `9b914e81…`, the reference mp4.
  - So the reused numbers stand.
- **Step and HBM:** 1063.8 / 1062.2 / 1067.3 ms; weight store hit 7.50 GB per rank; neuron-monitor 15.15 GiB on the DiT cores and 18.38 on the VAE core.
- **Its 209.9 s wall is not a warm figure.** Four NEFFs were compiled inside the run, one neuronx-cc per rank, which made the warm-up forward 73.9 s against 40.1 s.
  - Each new NEFF equals one existing per-rank DiT attention NEFF except for 10 debug blobs of the same length (`new_neffs_members.txt`), so the compiled program is unchanged.
  - This is the same pattern as the Qwen-Image hash check. The cause of the cache-key change was not established.

## Problems during phase 1 (harness, not Difflet)

1. **The first Qwen-Image compile exited in 0.1 s with rc=2.**
   - Cause: the harness passed `--backend trainium`, which main's CLI does not have.
   - Fix: the job scripts pass `DIFFLET_BACKEND=trainium` instead. Every compile and generate argument list for both models then passed a parse-only check against main's parser.
   - The failed run is kept as `B/qwen_image/aot_main_tp4/compile.failed-backend-flag`. The root had not been created, so it was still fresh for the rerun.
2. **The first Qwen-Image cold run was killed about 2 min in (08:56:42).**
   - Cause: the job had been started under `supervise.sh` from a Monitor task. It died when that Monitor reached its 30-minute limit, even with `setsid`.
   - The job quarantined the run as `cold.partial-1793005519` and reran cold after a fresh page-cache drop. All later runs are clean.
   - The LTX-2 cell ran as a `systemd-run --user` unit instead, which survived a Monitor expiry.

## Phase 2: notes and decisions (harness, not Difflet)

Phase 2 found no Difflet bug on the native branch, so it produced no code commits. `aot-main` stayed unmodified at `540e5f0`. The harness and probe changes below live outside the repo, in `B/scripts` and `B/probe`. Each change is additive, and the previous version is kept as `*.as-used-*.py`.

1. **Probe latent dumps for HunyuanVideo, Wan and native runs** (approved D9 tooling).
   - **Problem:** main's `DIFFLET_DUMP_LATENTS_DIR` exists only for Qwen-Image and LTX-2, and the native branch has none.
   - **Choice:** add an opt-in `HV1_DUMP_LATENTS_DIR` that wraps the orchestrators' `_denoise`, and set it only in q1 and the hash checks.
   - **Reason:** this gives the CPU references the exact device noise. The wrapper only observes, which the CPU self-tests check: HV AoT 15/15, Wan AoT 15/15, native Qwen 8/8, native HV/Wan 15/15, native LTX-2 23/23.
2. **Hash checks on root copies.**
   - **Problem:** a native hash check could write new cache entries into the original evidence root.
   - **Choice:** run each one on a `cp -a` copy, checked identical with `diff -r` before the run.
   - **Reason:** the originals keep their evidence. Two of the three runs did write NEFFs into the copy (Qwen-Image and Wan2.1).
3. **Hash-check wall times not used as timing figures.**
   - **Problem:** the hash-check walls were slower than the reference warm runs. Qwen-Image and Wan2.1 compiled NEFFs inside the run; HunyuanVideo's start-up and load were slower.
   - **Choice:** keep the reused numbers, which the pass rule (bit identity) allows, and record the walls as notes.
   - **Reason:** the step medians agree within 0.2–0.5 ms, and each in-run NEFF equals an existing one except for debug blobs.
4. **Wan2.1 AoT device-VAE corruption.**
   - **Problem:** the device-VAE frames are corrupted.
   - **Choice:**
     - isolate the cause without code changes (host fp32/bf16 decodes, plus a VAE-only rerun with VCS=2 and with VCS unset);
     - run the plan's optional A2 (`--host-vae`) on the same root, with no compile;
     - report A2's e2e as the valid AoT figure;
     - record the defect as a follow-up on main.
   - **Reason:** `aot-main` must not be modified, and a fix needs repeated 1.5 h VAE compiles.
5. **Harness additions.**
   - `bench4m.py` gained per-model entries for HunyuanVideo, Wan2.1 and LTX-2 native.
   - `root_prefix`: Wan2.1 uses `wan21`, so the root path matches the one the stage names.
   - `--preread-files`: the native Wan hash check pre-reads what the run actually reads, the umT5 and VAE checkpoints plus the bf16 store, instead of the unused fp32 transformer.
6. **LTX-2 native HEAD changed during the cell.**
   - **Problem:** plan commit `009d3dd` landed between the compile (`999c434`) and the cold run.
   - **Choice:** keep the runs, and record both HEADs.
   - **Reason:** `git diff --stat 999c434 009d3dd` touches only two files under `docs/superpowers/plans/`, so the difflet code is identical.
7. **Supervision.**
   - **Problem:** a job started from a Monitor dies when the Monitor expires (a phase-1 finding).
   - **Choice:** every phase-2 job ran as a `systemd-run --user` unit, watched with timeout-bounded loops.

**Disk:** the phase-2 roots take 60 GB (HunyuanVideo AoT), 40 GB (Wan2.1 AoT) and 22 GB (LTX-2 native). The native root copies take 27 GB (Wan2.1), 76 MB (HunyuanVideo) and 41 MB (Qwen-Image). `/` has 120 GB free after phase 2.

## Phase 4 preparation: CPU reference tooling for Qwen-Image and LTX-2

Tooling only, 2026-10-26 16:50–17:50 UTC. No 50-step reference has run yet, and the Neuron device stayed idle throughout.

**Scripts** (in `B/scripts`, outside the repo; main's originals are kept unmodified as `*.main-540e5f0.py`)
- `cpu_ref_qwen.py` and `cpu_ref_ltx2.py`, adapted from main's `scripts/cpu_fp32_reference.py`:
  - diffusers' own `QwenImagePipeline` / `LTX2Pipeline`, all components in fp32 on the CPU, at the model's pinned snapshot;
  - the prompt is encoded with only the text encoder loaded, then the encoder is freed (as on main);
  - the run starts from a device q1 dump: Qwen-Image through diffusers' `latents=` argument, LTX-2 by replacing `prepare_latents` / `prepare_audio_latents` (as on main; diffusers' own path would re-normalize the audio latents);
  - `--steps` defaults to 50.
- `quality_metrics.py`, adapted from main's `scripts/quality_vs_reference.py`. The metric definitions are main's (see Metrics below); it also reads lossless `.pt` frames and refuses mp4.
- `cpu_ref_common.py` and `cpu_only.py` hold shared helpers. `job_cpu_ref.sh` runs one reference as a `systemd-run --user` unit, under `/usr/bin/time -v`.
- Prepared but not started: `job_cpu_ref_queue_qwen_ltx2.sh` (the four 50-step references, one after the other) and `metrics_qwen_ltx2.sh` (the four comparisons).

**Environment**
- **Venv:** `/home/ubuntu/venvs/difflet-neuron`: torch 2.9.1, diffusers 0.38.0, transformers 4.57.6. The native venv has the same diffusers (0.38.0, torch 2.14).
  - Every reference uses this one venv, so a model's two references differ only in their noise.
  - It is also the venv of the 2026-10-24 20-step references.
- **Threads:** torch's default, 6 intra-op and 6 inter-op threads. The host has 6 physical cores (12 vCPU, Xeon 8488C).
- **No Neuron in the process:**
  - In this venv, diffusers, accelerate and transformers import `torch_xla` when it is installed. That initializes `torch_neuronx`, and the diffusers pipelines then call `xm.mark_step()` every step. The 2026-10-24 references ran that way.
  - In the native venv, `import torch` autoloads `torch_neuronx`.
  - `cpu_only.py` prevents both: it sets `USE_TORCH_XLA=0` and `TORCH_DEVICE_BACKEND_AUTOLOAD=0` before torch is imported, sets diffusers' `_torch_xla_available` to False, and refuses every import of the Neuron and XLA packages.
  - Each run records `cpu_only.report()`: Neuron modules loaded, open `/dev/neuron*` fds. A non-empty report fails the run.
  - In every smoke: no such module was loaded, no such fd was open, and the pipelines' `XLA_AVAILABLE` was False.

**Guidance, mirrored from the device runs**
- **Qwen-Image:** true CFG was off on both backends.
  - Both q1 runs passed `--guidance-scale 4.0` and no `--true-cfg-scale`.
  - Native runs true CFG only with `--true-cfg-scale > 1` (`_true_cfg_scale`). AoT on main has no negative branch.
  - The 4.0 is dropped by the DiT on both backends (`guidance_embeds` false).
  - The reference therefore passes `guidance_scale=4.0`, which diffusers ignores and logs, and `true_cfg_scale=1.0` with no negative prompt, so `do_true_cfg` is False.
- **LTX-2:** CFG was off on both backends at g1.0.
  - Native's `do_cfg` and AoT's `do_cfg` both require guidance > 1.
  - In diffusers, CFG needs `guidance_scale > 1` or `audio_guidance_scale > 1`, and the audio scale defaults to the video one.
- Both scripts count the DiT calls and fail unless there is exactly one per step.

**Injection is bit-exact**
- A forward pre-hook captures the transformer's first `hidden_states` (and, for LTX-2, `audio_hidden_states`): what actually enters the loop.
- That tensor must equal the fp32 upcast of the dump bit for bit (bf16 to fp32 is exact), and it must round back to the dump's bf16 bits.
- It held in every smoke, for both noises of both models:

  | dump (device q1) | dtype, shape | file / raw sha256 | first DiT input |
  |---|---|---|---|
  | Qwen AoT `initial.pt` | bf16 (1,4096,64) | `c2d9d8b3…` / `4dfaec64…` | bit-identical (`smoke_aot_cut2`) |
  | Qwen native `initial.pt` | bf16 (1,4096,64) | `95a4306b…` / `750e9824…` | bit-identical (`smoke_native_full2`, `smoke_native_cut2_final`) |
  | LTX-2 AoT `initial_video` / `initial_audio` | bf16 (1,2310,128) / (1,51,128) | `4458e4cf…` / `8e52babb…` (files) | both bit-identical (`smoke_aot_cut2`) |
  | LTX-2 native `initial_video` / `initial_audio` | bf16 (1,2310,128) / (1,51,128) | `16e09464…` / `411c72d9…` (files); raw `55b90133…` / `24900b1b…` | both bit-identical (`smoke_native_full2`, `smoke_native_cut2_final`) |

**Outputs** (in `B/<model>/cpu_ref/<name>/`)
- **Final latents:** the packed fp32 loop outputs, at the same point as the device dumps.
  - Qwen-Image: `final.pt` (1,4096,64), checked equal to the step callback's last latents.
  - LTX-2: `final_video.pt` (1,2310,128) and `final_audio.pt` (1,51,128), captured where diffusers unpacks and denormalizes them. Main recorded video only.
- **Decoded output, lossless:**
  - Qwen-Image: `out.pt`, fp32 (1,3,1024,1024) in [0,1], and `out.png` written from it with the device PNG rule;
  - LTX-2: `out.pt`, fp32 (1,49,3,480,704) in [0,1], the device `out.pt` layout. Also `audio.pt` (the vocoder waveform) and `mid.png`.
- **`reference.json`:** argv, versions, threads, script hashes, input and output hashes, the schedule, per-step and per-forward times, the time of each phase, VmRSS / RssAnon / RssFile peaks with the host's lowest MemAvailable, and the injection and CPU-only checks.

**Metrics (`quality_metrics.py`)**
- Latents: rel-L2 and cosine in float64 (main's definitions).
- Frames: RGB uint8, per-frame PSNR (mean and worst), and SSIM per Wang 2004: 11x11 Gaussian window, σ 1.5, K1 0.01, K2 0.03, valid region, per channel then averaged. These are main's definitions. Worst SSIM and per-channel SSIM are added.
- **Preprocessing:** float frames become uint8 by one rule, `uint8(clamp(x·255 + 0.5, 0, 255))` in float32. bf16 is upcast exactly first.
  - This is the rule of both device CLIs' PNG writers (torchvision `save_image` on AoT; native `_save_image_pil`, which reproduces it pixel for pixel).
  - The device PNGs (Qwen-Image) are read as they are.
- **Self-test passes:**
  - PSNR matches the closed form;
  - SSIM matches an independent scipy build to 1e-12;
  - main's script gives identical numbers on the Qwen device PNGs (12.1470 dB, SSIM 0.645899);
  - native vs AoT device latents for LTX-2 give cosine 0.363 / 0.641, as reported above.

**Smoke runs** (real shapes, 2 steps; `--cut-layers 2` builds a 2-block DiT)

| run | DiT | load (text encoder + encode / DiT and decoders) | step | decode | peak VmRSS | wall |
|---|---|---|---|---|---|---|
| Qwen `smoke_native_full2` | 60 blocks, 20.43 B params | 4.2 s (warm cache) / 301.2 s | 78.9, 79.0 s | 19.0 s | 85.2 GiB | 486 s |
| Qwen `smoke_aot_cut2` | 2 blocks | 127.3 s (cold) / 13.6 s | 2.8 s | 19.1 s | 33.1 GiB | 169 s |
| LTX-2 `smoke_native_full2` | 48 blocks, 18.88 B params | 29.7 s / 290.6 s | 79.6, 79.5 s (forward) | 26.4 s | 117.3 GiB | 515 s |
| LTX-2 `smoke_aot_cut2` | 2 blocks | 362 s (cold Gemma-3 fp32 read) / 59.3 s | 3.4 s | 25.3 s | 48.0 GiB | 462 s |

- **LTX-2 memory:** the 117.3 GiB VmRSS includes reclaimable file-backed pages.
  - An extra 1-step run (`smoke_aot_full1.failed-1step`) measured, after the DiT load: RssAnon 82.0 GiB, RssFile 20.4 GiB, host MemAvailable 39.9 GiB.
  - Gemma-3's fp32 checkpoint is mmapped as it is: 44 GiB RssFile during the encode, 1.6 GiB anonymous.
  - `job_cpu_ref.sh` refuses to start below 112 GiB MemAvailable and refuses a second concurrent reference.
- **That 1-step run failed** in diffusers' scheduler: `FlowMatchEulerDiscreteScheduler` with `shift_terminal` divides by zero on a one-sigma schedule (IndexError in `index_for_timestep`). Both scripts now refuse `--steps < 2`.

**Estimated 50-step wall time:**
- **Qwen-Image:** about 1.2–1.3 h each. Load about 7 min cold, plus 50 x 79 s, plus a 19 s decode. The 2026-10-24 20-step reference had six slow early steps (117–133 s), which would add about 4 min.
- **LTX-2:** about 1.3 h each. Encode about 6 min cold, DiT load about 5 min, 50 x 79.5 s, and a 26 s decode.
- **The four references in sequence:** about 5–5.5 h.

**Known differences, for the report:**
- **Qwen-Image text:** diffusers encodes the real prompt length (17 tokens) and drops the all-ones mask. Both device runs zero-pad to `TEXT_SEQ_LEN` and attend the padding unmasked.
- **Text encoders:** fp32 here, at the device runs' own dtypes and placements there (plan D2, D8; for example Qwen2.5-VL in bf16 at TP4 on the AoT device, Gemma-3 in bf16 on the native device).
- **VAEs:** fp32 diffusers VAEs here. On the device, Qwen-Image decodes with an fp32 VAE on native and a compiled bf16 VAE on AoT. LTX-2 decodes on the host on both backends (in bf16 on native).

## Phase 4: Qwen-Image CPU fp32 references (50 steps)

2026-10-26 17:49–20:12 UTC, CPU only: the Neuron device was idle throughout (`neuron_users` empty at each start, no Neuron module or `/dev/neuron*` fd in either process).

**How they ran**
- One `systemd-run --user` unit, `cpuref_qwen_queue`, ran `B/scripts/job_cpu_ref_queue_qwen.sh`: the two Qwen-Image entries of `job_cpu_ref_queue_qwen_ltx2.sh`, one after the other, each through `job_cpu_ref.sh` under `/usr/bin/time -v`. Queue log: `B/cpu_ref_queue_qwen.log`.
- Script `cpu_ref_qwen.py` (sha256 `90e537fe…`; `cpu_ref_common.py` `f88cce3f…`), unchanged since the tooling stage.
- diffusers' `QwenImagePipeline`, every component in fp32, snapshot `75e0b4be…`, 1024x1024, 50 steps, `guidance_scale=4.0` (ignored: `guidance_embeds` false), `true_cfg_scale=1.0` with no negative prompt. 50 DiT calls in each run (one per step), as on both device backends.
- **Environment:** venv `/home/ubuntu/venvs/difflet-neuron`, Python 3.12.3, torch 2.9.1, diffusers 0.38.0, transformers 4.57.6. torch threads: 6 intra-op, 6 inter-op (6 physical cores; `time -v` shows 545% and 563% CPU).

**Injection and schedule**
- In both runs, the transformer's first `hidden_states` is bit-identical to the fp32 upcast of the device dump and rounds back to its bf16 bits.
- `B/qwen_image/cpu_ref/schedule_check.json` (`B/scripts/schedule_check_qwen.py`) rebuilds the device schedule from the snapshot's scheduler config:
  - the formula is the one both device CLIs use: `sigmas = linspace(1, 1/50, 50)`, dynamic shift `mu` = 0.693548 at image_seq_len 4096 (main `difflet/cli/orchestrators/qwen_image.py:277-283`; native `difflet/models/qwen_image/neuron_application.py:661-672`);
  - both references' recorded timesteps and sigmas are identical to it (max |Δt| 0).

| reference | initial (device q1 dump) | text-encoder load + encode | DiT + VAE load | denoise, 50 steps | step median (min–max) | decode | wall | peak RSS (`time -v`) | lowest host MemAvailable |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ref_aot_noise_50` | `B/qwen_image/aot_main_tp4/q1/initial.pt` `c2d9d8b3…` (raw `4dfaec64…`) | 2.8 + 1.5 s (in the page cache) | 298.6 s | 4026.4 s | 80.54 s (79.62–82.82) | 19.3 s | 4352.4 s (1:12:39) | 85.44 GiB | 36.4 GiB |
| `ref_native_noise_50` | `B/qwen_image/native_tp4_hashcheck/q1/initial.pt` `95a4306b…` (raw `750e9824…`) | 126.0 + 1.5 s (evicted by the previous run) | 24.9 s | 4073.1 s | 81.52 s (79.59–83.50) | 19.0 s | 4249.0 s (1:10:53) | 85.44 GiB | 36.5 GiB |

**Outputs** (in `B/qwen_image/cpu_ref/<reference>/`, with `run.log`, `rc` 0 and `reference.json`)
- `ref_aot_noise_50`: `final.pt` `6b8a5dea…` (fp32 (1,4096,64)), `out.pt` `a0a9edf3…` (fp32 (1,3,1024,1024), range 0.029–0.972), `out.png` `464155ec…`.
- `ref_native_noise_50`: `final.pt` `b357f08a…`, `out.pt` `2c8ff472…` (range 0.035–0.927), `out.png` `09035397…`.
- I opened both PNGs: both show a red fox running through snow in a forest, on prompt.

**Metrics** (`quality_metrics.py`, main's definitions; frames: the device `out.png` as it is against `to_uint8` of the reference's fp32 `out.pt`; latents: the device `final.pt` (bf16) against the reference's `final.pt`, both packed loop outputs)

| pair (same noise) | PSNR | SSIM (R / G / B) | latent rel-L2 | latent cosine | file |
|---|---:|---:|---:|---:|---|
| AoT device q1 vs `ref_aot_noise_50` | **21.48 dB** | **0.9115** (0.915 / 0.911 / 0.908) | 0.276 | 0.9614 | `B/qwen_image/cpu_ref/metrics_aot_vs_ref.json` |
| native device q1 vs `ref_native_noise_50` | **16.11 dB** | **0.8234** (0.813 / 0.828 / 0.829) | 0.522 | 0.8722 | `B/qwen_image/cpu_ref/metrics_native_vs_ref.json` |
| control: native device vs `ref_aot_noise_50` (mismatched noise) | 11.96 dB | 0.634 | 0.798 | 0.675 | `…/control_native_dev_vs_aotnoise_ref.json` |
| control: AoT device vs `ref_native_noise_50` (mismatched noise) | 11.47 dB | 0.586 | 0.912 | 0.607 | `…/control_aot_dev_vs_nativenoise_ref.json` |

Reading the numbers:
- Both matched pairs sit well above the mismatched-noise controls, so the pairing is right and each device run follows its reference's trajectory.
- The native image keeps its reference's composition (a fox facing right in the same place) but not its pose: the reference fox leaps, the device fox trots. A late-trajectory divergence of that kind costs far more PSNR than it changes the picture. The AoT pair keeps the composition and position; its leg pose differs too (the reference fox gallops, the device fox walks; see "Quality vs CPU fp32 reference").
- Compare with care: the two backends start from different noise, so 21.5 vs 16.1 dB measures how far each run drifts from its own fp32 trajectory. It is not a pixel comparison of the backends. Both runs pass through the same known differences from the reference (below), and a single seed cannot separate them. *Inference:* the text conditioning (padding, below) is the largest common difference.
- The 20-step AoT figure from the bf16-reduce campaign (24.72 dB, SSIM 0.957, cosine 0.981, same AoT noise `c2d9d8b3…`) is not comparable: more steps leave more room to diverge.

**Known differences from the device runs (carried into the metrics)**
- **Text:** diffusers encodes the real 17 tokens, (1,17,3584) fp32, and drops the all-ones mask. Both device runs zero-pad to `TEXT_SEQ_LEN` and attend the padding unmasked.
- **Text encoder:** fp32 here. On the device, Qwen2.5-VL runs in bf16: at TP4 on the device for AoT, on rank 0's host for native.
- **DiT:** fp32 here, bf16 at TP4 on both device backends.
- **VAE:** fp32 here. On the device, native decodes with an fp32 VAE and AoT with a compiled bf16 VAE.

## Phase 4: LTX-2 CPU fp32 references (50 steps)

The references ran 2026-10-26 20:15–22:51 UTC and a 20-step cross-check ran 22:57–23:38 UTC, all on the CPU. The Neuron device was idle throughout: `neuron_users` was empty at each start, and no process loaded a Neuron module or opened a `/dev/neuron*` fd.

**How they ran**
- One `systemd-run --user` unit, `cpuref_ltx2_queue`, ran `B/scripts/job_cpu_ref_queue_ltx2.sh`. The script holds the two LTX-2 entries of `job_cpu_ref_queue_qwen_ltx2.sh` and runs them one after the other, each through `job_cpu_ref.sh` under `/usr/bin/time -v`. Queue log: `B/cpu_ref_queue_ltx2.log`.
- Script `cpu_ref_ltx2.py` (sha256 `e9fd953b…`; `cpu_ref_common.py` `f88cce3f…`), unchanged since the tooling stage.
- diffusers' `LTX2Pipeline` at snapshot `47da56e2…`, with every component in fp32: Gemma-3, connectors, a 48-block DiT of 18.88 B parameters, the video VAE, the audio VAE and the vocoder.
  - Workload: 480x704x49 at 24 fps, 50 steps, `guidance_scale=1.0`, so CFG is off.
  - Each run made 50 DiT calls, one per step, as on both device backends.
  - Prompt embeds: (1,1024,188160) fp32 with 13 real tokens (`mask_sum` 13).
- **Environment:** venv `/home/ubuntu/venvs/difflet-neuron`, Python 3.12.3, torch 2.9.1, diffusers 0.38.0, transformers 4.57.6.
  - torch threads: 6 intra-op and 6 inter-op, on 6 physical cores.
  - `time -v` shows 511% and 524% CPU.

**Injection and schedule**
- In both runs, the transformer's first `hidden_states` and `audio_hidden_states` are bit-identical to the fp32 upcast of the device dumps, and both round back to the dumps' bf16 bits.
- `B/ltx_2/cpu_ref/schedule_check.json` (from `B/scripts/schedule_check_ltx2.py`) rebuilds the device schedule from the snapshot's scheduler config.
  - Both backends use the same formula: `sigmas = linspace(1, 1/50, 50)`, and `mu = ltx_2_scheduler_mu` at `max_image_seq_len` = 2.05, with `shift_terminal` 0.1 from the config.
    - Native: `difflet/models/ltx_2/pipeline.py:493-517,1172-1186`.
    - AoT: main `difflet/models/ltx_2/pipeline.py:511-517,1181-1195`.
  - diffusers' `calculate_shift` groups the `mu` expression differently but gives the same bits.
  - Both references' recorded timesteps and sigmas are identical to the rebuilt schedule (max |Δt| 0, max |Δσ| 0).

| reference | initial (device q1 dumps, file / raw sha256) | Gemma-3 load + encode | DiT + decoders load | denoise, 50 steps | step median (min–max) | decode | wall | peak RSS (`time -v`) | peak RssAnon | lowest host MemAvailable |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ref_aot_noise_50` | `B/ltx_2/aot_main_tp4_twocore/q1`: video `4458e4cf…` / `1b2149d9…`, audio `8e52babb…` / `fb75005f…` | 1.9 + 360.0 s (cold) | 329.2 s | 3980.5 s | 79.49 s (79.04–85.20) | 25.9 s | 4701.3 s (1:18:28) | 117.24 GiB | 85.29 GiB | 36.54 GiB |
| `ref_native_noise_50` | `B/ltx_2/native_tp4/q1`: video `16e09464…` / `55b90133…`, audio `411c72d9…` / `24900b1b…` | 2.2 + 358.6 s (evicted by the previous run) | 184.5 s | 4023.7 s | 80.13 s (79.60–85.47) | 27.5 s | 4600.7 s (1:16:49) | 116.00 GiB | 85.21 GiB | 36.41 GiB |

- In both runs, the 117 GiB RSS peak includes the mmapped fp32 checkpoint pages: Gemma-3 during the encode, then the DiT. Anonymous memory peaked at 85.3 GiB, and the host never had less than 36.4 GiB available.

**Outputs** (in `B/ltx_2/cpu_ref/<reference>/`, beside `run.log`, `rc` 0 and `reference.json`)

| file | `ref_aot_noise_50` | `ref_native_noise_50` |
|---|---|---|
| `final_video.pt`, fp32 (1,2310,128), the packed loop output | `5d88c6a0…` | `5e239f4d…` |
| `final_audio.pt`, fp32 (1,51,128) | `c1f702ac…` | `82d883e9…` |
| `out.pt`, fp32 (1,49,3,480,704) in [0,1], lossless frames | `95f016cc…`, range 0–0.952 | `ed701374…`, range 0–1 |
| `audio.pt`, the vocoder waveform, fp32 (1,2,48240) | `054b4f53…` | `c8904bb7…` |
| `mid.png`, frame 24 | `957a0a24…` | `6e08645e…` |

- **Content:** I opened both mid frames and compared each with its device counterpart.
  - **Native noise:** a red fox runs across deep snow in a conifer forest, with the same cyan cast as the native device output. So the cyan cast and the soft fox come from the workload (guidance 1, no CFG), not from the device. The device frame has the same fox, in the same pose and place; only the background trees differ.
  - **AoT noise:** a snowy forest with pine branches and a blurred red-brown animal, only loosely on prompt. The AoT device frame shares the colour scheme but not the layout: it shows a large smeared shape where the reference has trees and the animal.

**Metrics** (`B/scripts/metrics_ltx2.sh`, which runs `quality_metrics.py` with main's definitions)
- Frames: the device q1 `out.pt` (bf16, upcast exactly) against the reference's fp32 `out.pt`. Both are turned into RGB uint8 by the one rule `uint8(clamp(x·255 + 0.5, 0, 255))`, and there is no other preprocessing. All 49 frames are compared.
- Latents: the device `final_video.pt` / `final_audio.pt` (bf16) against the reference's fp32 ones, all packed loop outputs.

| pair | PSNR mean (worst) | SSIM mean (worst); R / G / B | video latent rel-L2 / cosine | audio latent rel-L2 / cosine | file |
|---|---:|---:|---:|---:|---|
| **AoT device q1 vs `ref_aot_noise_50`** (same noise) | **16.21 dB** (15.45, frame 5) | **0.660** (0.555, frame 48); 0.687 / 0.666 / 0.627 | 0.786 / **0.725** | 0.335 / 0.980 | `B/ltx_2/cpu_ref/metrics_aot_vs_ref.json` |
| **native device q1 vs `ref_native_noise_50`** (same noise) | **18.06 dB** (16.49, frame 27) | **0.811** (0.743, frame 15); 0.820 / 0.808 / 0.807 | 0.414 / **0.930** | 0.229 / 0.989 | `B/ltx_2/cpu_ref/metrics_native_vs_ref.json` |
| control: native device vs `ref_aot_noise_50` (mismatched noise) | 6.88 dB | 0.437 | 1.504 / 0.198 | 1.070 / 0.595 | `…/control_native_dev_vs_aotnoise_ref.json` |
| control: AoT device vs `ref_native_noise_50` (mismatched noise) | 9.13 dB | 0.490 | 1.128 / 0.315 | 0.997 / 0.563 | `…/control_aot_dev_vs_nativenoise_ref.json` |
| control: the two references against each other (noise only) | 8.15 dB | 0.470 | 1.414 / 0.176 | 1.039 / 0.522 | `…/control_ref_native_vs_ref_aot.json` |

**Reading the numbers**
- Both matched pairs sit well above the three mismatched-noise controls, so the pairing is right.
- **Native** follows its fp32 trajectory closely: video latent cosine 0.930, and the same fox in the same pose.
- **AoT** follows its trajectory much more loosely: video latent cosine 0.725 and 16.2 dB, with a different layout in the frames. Its audio latents stay close (cosine 0.980).
- As for Qwen-Image, the two backends start from different noise, so this compares each run with its own fp32 trajectory. It is not a pixel comparison of the backends.

**The AoT gap against the 2026-10-24 run** (an open finding, not explained yet)
- **2026-10-24:** single-core attention (`MODULE_20da5719`), bf16 all-reduce, 20 steps, same noise `4458e4cf…`. Against its 20-step CPU reference, it scored video latent cosine 0.991 and 34.14 dB (`/home/ubuntu/campaign/bf16-reduce/runs/ltx2/quality_bf16_vs_cpu.json`, mp4 against mp4).
- **Today:** the two-core module (`MODULE_e1043f48…`, `attention_cte_lnc0` with grid 2), 50 steps, scored cosine 0.725.
- **The reference side is ruled out:**
  - A 20-step reference with today's script from the same AoT dumps (`B/ltx_2/cpu_ref/xcheck_aot_noise_20/`, rc 0, wall 2414 s, `final_video.pt` `5b953937…`) gives a `final_video` bit-identical to the 2026-10-24 CPU reference (`/home/ubuntu/campaign/bf16-reduce/runs/cpu_ref/ltx_2/final_video.pt`; `torch.equal`, max |Δ| 0).
  - Against it, the 2026-10-24 device output (`bf16/warm1/final_video.pt` `b426bc92…`) gives video cosine 0.9907 and audio cosine 0.9994 (`xcheck_dev20_2026-10-24_vs_ref20.json`, `xcheck_ref20_vs_ref20_2026-10-24.json`).
  - That 2026-10-24 reference ran with `torch_xla` loaded (`xm.mark_step()` every step) and today's does not. The identical result shows that this did not change the numbers.
- **What remains is the device side:** 50 against 20 steps, and two-core against single-core attention. A device run can separate them; the device stays idle in this phase.
  - *Comparison:* Qwen-Image AoT went from cosine 0.981 at 20 steps to 0.961 at 50 steps, and LTX-2 native at 50 steps reaches 0.930. So more steps alone would not obviously cost this much.
  - *Recommendation for the next device block:* run LTX-2 AoT two-core at 20 steps from the same seed and compare it with `xcheck_aot_noise_20` and with `b426bc92…`. If it stays near 0.99, the steps explain the gap; if it falls, two-core attention does. Optionally also run single-core at 50 steps against `ref_aot_noise_50`.

**Known differences from the device runs (carried into the metrics)**
- **Text encoder:** Gemma-3 runs in fp32 here. Native runs it in bf16 on the device at TP4. AoT runs it in bf16 on the host (main's host pipeline loads with `torch_dtype=bfloat16`, `difflet/models/ltx_2/application.py:425-445`; the orchestrator's `_dtype` is bf16).
- **Connectors:** fp32 here, bf16 on both backends.
- **DiT:** fp32 here, bf16 at TP4 on both backends.
- **Decoders:** the video VAE, audio VAE and vocoder run in fp32 here. Both backends decode on the host in bf16.

## Quality vs CPU fp32 reference (part 1: Qwen-Image and LTX-2)

This section collects the quality numbers for the two models that have references, recomputed from scratch on 2026-10-26 after both reference blocks had finished. CPU only, the Neuron device stayed idle. The run was one `systemd-run --user` unit, `quality_part1`, running `B/scripts/quality_part1.sh` (sha256 `9518c2b9…`); all outputs are in `B/quality_part1/`, and `summary.txt` there lists every number with its input hashes.

**Pairs.** Each device q1 output is compared with the 50-step CPU fp32 reference (diffusers, fp32 everywhere, CFG off, the same schedule) that started from **that run's own initial latents**, bit-identical at the first DiT call (sections "Phase 4: … CPU fp32 references"):
- native device q1 against `ref_native_noise_50`;
- AoT device q1 against `ref_aot_noise_50`.

The two backends start from different noise, so this measures how far each device run drifts from its own fp32 trajectory. It is **not** a native-vs-AoT pixel comparison.

**Definitions** (`B/scripts/quality_metrics.py`, sha256 `8f6e1b18…`, main's `scripts/quality_vs_reference.py` definitions at `540e5f0`):
- **Frames:** RGB uint8. PSNR per frame with peak 255; the table gives the mean over frames and the worst frame. SSIM per Wang 2004: 11x11 Gaussian window, sigma 1.5, K1 0.01, K2 0.03, L 255, valid region only, computed per channel and then averaged over R, G and B; mean and worst over frames.
- **Latents:** relative L2 `||cand − ref|| / ||ref||` and cosine, both in float64, on the packed loop-output latents (Qwen-Image `final.pt` (1,4096,64); LTX-2 `final_video.pt` (1,2310,128) and `final_audio.pt` (1,51,128)). The device latents are bf16 and are upcast exactly.
- **Preprocessing, and nothing else:**
  - Qwen-Image: the device `out.png` as written by the CLI (8-bit RGB, 1024x1024) against the reference's fp32 `out.pt` (1,3,1024,1024) turned into uint8 by `uint8(clamp(x·255 + 0.5, 0, 255))`. That is the rule both device CLIs use for their PNG, so the reference passes through the same rounding as the device output.
  - LTX-2: the device bf16 `out.pt` (1,49,3,480,704) against the reference's fp32 `out.pt`, both turned into uint8 by the same rule, all 49 frames. No mp4 is involved on either side. No resizing, cropping, colour conversion or frame alignment anywhere.

**Tool sanity checks** (all pass)

| check | expected | result | file |
|---|---|---|---|
| `--selftest`: closed-form PSNR, SSIM against an independent scipy build, the uint8 rule, latent cosine of scaled and negated copies | all ok | PASS, SSIM equal to 1e-12 | `selftest.txt` |
| identical pair: Qwen-Image reference `out.pt` and `final.pt` against themselves | inf dB, SSIM 1, rel-L2 0, cosine 1 | inf / 1.000000 / 0 / 1.000000 | `sanity_identical_qwen.json` |
| identical pair: LTX-2 reference, 49 frames, video and audio latents against themselves | inf dB, SSIM 1, rel-L2 0, cosine 1 | inf (worst inf) / 1.000000 / 0 / 1.000000 | `sanity_identical_ltx2.json` |
| the reference's `out.png` against its own fp32 `out.pt` (checks the uint8 rule end to end) | inf dB | inf dB, SSIM 1 | `sanity_png_rule_qwen.json` |
| known pair: Qwen-Image native device PNG against the AoT device PNG; main's unmodified script gave 12.1470 dB, SSIM 0.645899 at the tooling stage | the same | 12.1470 dB, SSIM 0.645899 | `sanity_known_qwen_devices.json` |
| known pair: the 2026-10-24 LTX-2 AoT device latents (`b426bc92…`) against the 20-step reference (`xcheck_aot_noise_20`) | video cosine 0.9907 (LTX-2 phase 4) | 0.990696 | `sanity_known_ltx2_xcheck20.json` |

The four recomputed pairs equal the earlier `B/<model>/cpu_ref/metrics_*_vs_ref.json` with the same input hashes. The one exception is the last digit of one float64 cosine (Qwen-Image AoT 0.9614398560993154 against …155), from a multi-threaded reduction.

**Results**

| model | pair (same noise) | PSNR mean (worst frame) | SSIM mean (worst frame); R / G / B | latent rel-L2 / cosine | audio latent rel-L2 / cosine |
|---|---|---:|---:|---:|---:|
| Qwen-Image 1024x1024 | native device q1 vs `ref_native_noise_50` | **16.11 dB** (1 frame) | **0.823**; 0.813 / 0.828 / 0.829 | 0.522 / **0.872** | n/a |
| Qwen-Image 1024x1024 | AoT device q1 vs `ref_aot_noise_50` | **21.48 dB** (1 frame) | **0.912**; 0.915 / 0.911 / 0.908 | 0.276 / **0.961** | n/a |
| LTX-2 480x704x49 | native device q1 vs `ref_native_noise_50` | **18.06 dB** (16.49, frame 27) | **0.811** (0.743, frame 15); 0.820 / 0.808 / 0.807 | video 0.414 / **0.930** | 0.229 / 0.989 |
| LTX-2 480x704x49 | AoT device q1 vs `ref_aot_noise_50` | **16.21 dB** (15.45, frame 5) | **0.660** (0.555, frame 48); 0.687 / 0.666 / 0.627 | video 0.786 / **0.725** | 0.335 / 0.980 |

Files: `B/quality_part1/{qwen,ltx2}_{native,aot}_vs_ref.json`. Mismatched-noise controls, which score far lower and confirm the pairing, are in the two phase 4 sections (Qwen-Image 11.5–12.0 dB; LTX-2 6.9–9.1 dB).

**What the references themselves show** (I opened the Qwen-Image PNGs and a strip of LTX-2 frames 0, 12, 24, 36 and 48 of each reference, `B/ltx_2/cpu_ref/ref_*_noise_50/strip_0_12_24_36_48.png`, made from the lossless `out.pt` by `B/scripts/ref_strip.py`; AoT `ba3e8b9c…`, native `e059c7dc…`)
- **Qwen-Image: both references show the prompt.**
  - AoT noise: a sharp, photographic red fox galloping right across snow, snow-covered trunks behind. The AoT device image has the same fox in the same place and framing, but walking with all four feet down rather than galloping. *(This corrects "the AoT pair keeps the pose" in the Qwen-Image phase 4 section: it keeps the composition and position, not the leg pose.)*
  - Native noise: a red fox leaping right through a snowy forest; the fox's head is malformed (a smeared muzzle). The native device image has a fox in the same place and direction, trotting, with a clean head.
- **LTX-2 at g1.0: the CPU fp32 reference is soft and colour-cast too**, so this comes from the workload (guidance 1.0, no CFG), not from the device.
  - Native noise: on prompt. A small red fox runs across deep snow in a conifer forest, past a large trunk, toward the camera. The whole video has a strong cyan cast, the fox is smeared, and the last frames show checkerboard-like texture on the fox. The native device video matches it closely: the same fox, path, trunk and cyan cast, frame for frame.
  - AoT noise: only loosely on prompt. Snow-laden pine branches in the foreground against a blue-violet sky, blowing snow, and a blurred red-brown animal that appears from frame 12 onward and is never clearly a fox.
  - The AoT device video is **worse than its reference**, not just different: the same blue palette, but the frames are dominated by crumpled, foil-like smeared shapes and a large red-brown blob; neither the branches nor the animal of the reference are recognisable. That matches its low latent cosine (0.725) and is the open AoT finding of the LTX-2 phase 4 section.

**Reading the numbers**
- **Qwen-Image:** AoT 21.5 dB / 0.912, native 16.1 dB / 0.823. Both device images are on prompt and keep their reference's composition. The PSNR gap mostly reflects how late in the trajectory each run diverges (pose), which a single seed cannot attribute to a backend. Both device runs share the same known differences from the reference: the padded, unmasked text conditioning (the largest common difference, *inference*), bf16 text encoder and DiT, and for AoT a compiled bf16 VAE.
- **LTX-2:** native 18.1 dB / 0.811 with video cosine 0.930 follows its reference closely. AoT 16.2 dB / 0.660 with video cosine 0.725 does not, and its frames show artefacts that its reference does not. Audio latents stay close on both (cosine 0.989 and 0.980).
- PSNR on these generative outputs is low in absolute terms: a small shift of a fox's legs or of background trees costs several dB. SSIM and latent cosine track what the frames show better here than PSNR.

## Next

- **Phase 4 (CPU):** the 7 references, 50 steps each, in fp32, Qwen-Image and LTX-2 first. Inputs:
  - **Qwen-Image:**
    - AoT noise: `B/qwen_image/aot_main_tp4/q1`, `initial.pt` `c2d9d8b3…`;
    - native noise: `B/qwen_image/native_tp4_hashcheck/q1`, `initial.pt` `95a4306b…`.
  - **LTX-2:**
    - AoT noise: `B/ltx_2/aot_main_tp4_twocore/q1`;
    - native noise: `B/ltx_2/native_tp4/q1`, `initial_video` / `initial_audio` raw `55b90133…` / `24900b1b…`.
    - The reference needs a lossless frames output (plan D9): done, `cpu_ref_ltx2.py` writes `out.pt`.
  - Qwen-Image: **done** (section "Phase 4: Qwen-Image CPU fp32 references").
  - LTX-2: **done** (section "Phase 4: LTX-2 CPU fp32 references"). It leaves an open finding, the AoT gap against 2026-10-24, which needs a 20-step two-core AoT device run.
  - **HunyuanVideo:**
    - native noise: fp32, `B/hunyuan_video/native_tp4_hashcheck/q1/initial.pt`;
    - AoT noise: bf16, `B/hunyuan_video/aot_main_tp4/q1/initial.pt`.
    - Caveat: the AoT VAE tiling caps frame PSNR at about 42.5 dB (`hunyuan_video_neuron.md`).
  - **Wan2.1:** one reference serves both backends: noise `03b6284b…` (raw `16aeaed7…`).
    - Native frames: `B/wan2_1/native_tp4_hashcheck/q1/out.pt`.
    - AoT frames: use the fp32 host decode `B/wan2_1/aot_main_tp4_twocore/q1_host_vae/host_fp32_frames.pt`, which is the A2 placement, because the device-VAE frames are corrupted.
- **Phase 5:** PSNR / SSIM fill the last column of the summary table. Part 1 (Qwen-Image and LTX-2) is **done**: section "Quality vs CPU fp32 reference (part 1)". Part 2 (HunyuanVideo and Wan2.1) waits for their references.
- **Phase 6:** definition B, native cold without the weight store, for Wan2.1 and LTX-2.
- **Follow-up on main:** the compiled bf16 Wan VAE (`trainium/wan/vae.py`) corrupts frames on the device. Wan2.2 on 2026-10-24 shows the same kind of corruption.

## Toolchain

- **AoT:** `/home/ubuntu/venvs/difflet-neuron`: Python 3.12.3, torch 2.9.1, torch-neuronx 2.9.0.2.15.32035+de43f57c, neuronx-cc 2.26.6360.0+6f180f47.
- **Native:** `/home/ubuntu/venvs/native-312`: Python 3.12.3, torch 2.14.0, torch-neuronx 2.14.3.0.1+b1b6c66, neuronx-cc 2.28.7631.0+8f7b0cf5 (re-read from the venv after phase 2; the same as in `native_vs_aot_tp4_2026-10-25.md`).
- **Host runtime:** aws-neuronx-runtime-lib 2.35.30.0, collectives 2.35.43.0 (re-read with `dpkg -l` after phase 2).
- The *reused* native cells ran on this host; their toolchains are listed in their own reports.
