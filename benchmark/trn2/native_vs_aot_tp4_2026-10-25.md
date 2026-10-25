# Native vs AoT at TP4 on trn2.3xlarge, re-measured 2026-10-25

**Backends:** neuron (TorchNeuron: torchrun, one process per NeuronCore, per-block `torch.compile`; "native") and the Trainium AoT backend (NxD; "AoT"), the same `difflet` CLI on the same host  
**Device:** trn2.3xlarge: 1 device, 4 NeuronCores (LNC=2, 24 GiB HBM per core), 12 vCPU, 124 GB RAM, one EBS root volume  
**Code under test:** branch `feat/hunyuan-neuron` at `4f6e1de`, clean tree in every run (`head: 4f6e1de dirty=0` in each `run.log`)  
**Raw evidence:** kept out of git, under `/home/ubuntu/bench-2026-10-25/` on the benchmark host (`B` below). Each run has its own directory `B/<model>/<cell>/<run>/` with:
- `run.log`: the CLI output and `/usr/bin/time -v`;
- `summary.json`: the harness record (command, env, HEAD, meminfo, checkpoint page-cache residency, probe timeline, hashes, PNG check);
- `probe/`: the per-process events;
- `nmon.jsonl`: neuron-monitor, sampled every second;
- `out.png` and `work/latents.pt`.

`B/<model>/summary.jsonl` has one line per run.

Every number below comes from a run made for this report and names its run directory, unless it is marked *reused*. Inferences are labelled. HunyuanVideo's TP4 native-vs-AoT comparison from the same day is in `hunyuan_video_neuron.md`.

## Qwen-Image

**Model:** `Qwen/Qwen-Image` at revision `75e0b4be04f60ec59a75f475837eced720f823b6`.  
**Workload:** 1024x1024, 50 steps, seed 42, prompt "a cinematic shot of a red fox running through a snowy forest", `--guidance-scale 4.0` (the DiT drops it: `guidance_embeds` is off), `HF_HUB_OFFLINE=1`, the workload of `neuron_final.md`.  
**Campaign:** 2026-10-25, 05:36 – 06:50 UTC.

**Placements:**
- **Native:** the default placement. Qwen2.5-VL encodes on rank 0's host and the DiT runs at tp4 in the generate launch; the vae stage then decodes in fp32 in a one-core launch of its own.
- **AoT:** `--backend trainium` with the NxD venv, at its own default placement. Three stage processes run one after the other:
  - the text stage: Qwen2.5-VL at TP4 on the device;
  - the generate stage: the DiT at TP4, with the two-core attention already in Qwen-Image's AoT CLI (`_VIRTUAL_CORE_SIZE = 2`);
  - the vae stage: the compiled VAE on one core.

**Compile cold:** `difflet compile` on a fresh root, with the checkpoint in the page cache (see Method). For AoT, the fresh root also holds a fresh pre-sharded weight store and a fresh neuronx-cc cache, so the AoT compile includes the weight sharding and every NEFF compile.

| metric | native (`native_tp4`) | AoT (`aot_tp4`) | native vs AoT | 2026-10-20 values (`neuron_final.md`, *reused*): native / AoT / native vs AoT |
|---|---:|---:|---:|---|
| DiT step median / p10 / p90 | **359.0** / 358.2 / 360.1 ms (runs 359.1, 359.0, 358.8) | **418.2** / 417.2 / 418.9 ms (runs 418.2, 417.8, 418.3) | **−14.2%** | 352.6 / 418.4 ms / −15.7% |
| DiT step with `--sp` (`native_tp4_sp`, n=2) | **331.0** / 330.4 / 332.5 ms (runs 330.9, 331.0) | — | **−20.9%** | 329.9 / 418.4 ms / −21.1% |
| compile cold, checkpoint in the page cache | **187.0 s** (generate launch 91.2, vae launch 93.8) | **1238.5 s** (text stage 134.4, generate stage 665.2, vae stage 419.4) | **−84.9%** | 161.7 / 1480 s / −89% |
| e2e warm (n=3; gen0 discarded) | **82.3 s** (81.4, 82.1, 83.4; gen0 72.5) | **75.4 s** (74.2, 76.4, 75.7; gen0 75.0) | **+9.2%** | 79.6 / 76.9 s / +3.5% |
| e2e cold, page cache dropped | **502.0 s** (DiT checkpoint 310.9 s, host Qwen2.5-VL 99.6 s) | **504.2 s** (text stage 140.7 s, DiT load 315.9 s) | **−0.4%** | 496.1 / 505.9 s / −1.9% |
| peak HBM per DiT core: allocated / neuron-monitor | 10.15 / 10.81 GiB (vae core: 9.30 / 13.85 GiB) | — / 10.29 GiB (no `max_memory_allocated`: the AoT stages are not torchrun ranks) | +5.1% (neuron-monitor) | native 10.15 / 10.76 GiB, vae core 13.83 GiB; AoT not given |
| output (opened) | fox in a snowy forest; PNG `b1906d53…`, latents.pt `4da4c89e…` | fox walking in snow, a different composition (see "Outputs"); PNG `e24798fe…`, latents.pt `7b36ae14…` | | native PNG `a300a78e…` |
| logs | `B/qwen_image/native_tp4/{compile,cold,gen0..gen3}/` | `B/qwen_image/aot_tp4/{compile,cold,gen0..gen3}/` | | |

What the table says:

- **The DiT step reproduces the 2026-10-20 picture.**
  - Native is 14.2% faster than AoT per step, and 20.9% faster with `--sp`.
  - AoT's step is unchanged: 418.2 ms against 418.4 ms.
  - Native's step is 1.8% slower than on 2026-10-20 (see "Drift").
- **Warm e2e: AoT leads by 6.9 s (75.4 vs 82.3 s), up from 2.7 s.** Native's denoise is 3.3 s faster (18.0 vs 21.3 s), but two things cost it more (breakdown below):
  - its generate launch spends 21.5–24.4 s from rank start to load start in every timed run (launch jitter, open issue 1);
  - its eager fp32 vae launch takes 19.2–19.5 s, against 13.3–14.4 s for AoT's compiled vae stage.

  With a fast launch (11.7 s, as in native gen0), native's e2e is 72.5 s, which would beat AoT's.
- **Cold e2e: level (−0.4%).**
  - Native reads the bf16 checkpoint (310.9 s) and the host Qwen2.5-VL (99.6 s) in one load phase.
  - AoT reads its text encoder in the text stage (140.7 s), then its pre-sharded DiT (315.9 s).
- **Compile: native is 6.6x faster** (187.0 vs 1238.5 s).
  - AoT's time is the VAE's neuronx-cc compile (404.2 s), the DiT weight sharding (361.7 s), the DiT build (297.5 s, of which 85.2 s loading the module and 31.5 s compiling) and the text encoder build (106.2 s) (`aot_tp4/compile/run.log`).
  - Native writes no weight store ("no gain": the checkpoint is already bf16), the same as on 2026-10-20.
- **HBM: AoT uses 0.5 GiB less per DiT core** by neuron-monitor (10.29 vs 10.81 GiB). Native's vae core peaks at 13.85 GiB, in a launch of its own after the DiT launch has exited.

### Drift from 2026-10-20 (flag: more than 2%)

| cell, metric | now | 2026-10-20 (*reused*, `neuron_final.md`) | delta | flag | likely cause |
|---|---:|---:|---:|:---:|---|
| native tp4, step | 359.0 ms | 352.6 ms | +1.8% | — (below 2%) | See note 1. |
| native tp4, compile cold | 187.0 s | 161.7 s | **+15.6%** | **yes** | See note 2. |
| native tp4, e2e warm | 82.3 s | 79.6 s | **+3.4%** | **yes** | See note 3. |
| native tp4, e2e cold | 502.0 s | 496.1 s | +1.2% | — | The load phases match: DiT checkpoint 310.9 s now and on 2026-10-20, host Qwen2.5-VL 99.6 s now against 99.9 s. |
| native tp4, HBM alloc / nmon | 10.15 / 10.81 GiB | 10.15 / 10.76 GiB | 0 / +0.5% | — | |
| native `--sp`, step | 331.0 ms | 329.9 ms | +0.3% | — | |
| native `--sp`, e2e warm | 73.3 s (77.1, 69.6) | 75.0 s (76.2, 73.8) | **−2.3%** | **yes** | Launch jitter, in the other direction. Rank 0's start-to-load time was 19.1 s in gen1 and 11.9 s in gen2 (`native_tp4_sp/gen{1,2}`). n=2 is too few to average the jitter out. |
| native `--sp`, compile (page cache as left: 94% of the checkpoint resident) | 162.6 s (generate launch 69.3, vae launch 91.9) | 154.8 s (62.1 + 92.5) | **+5.0%** | **yes** | The generate launch is 7.2 s longer and the vae launch is level. *Inference:* the same toolchain change as note 1. The 2026-10-20 split below the launch level is not available to confirm it. |
| native `--sp`, HBM alloc | 10.00 GiB | 9.93 GiB | +0.7% | — | |
| AoT tp4, step | 418.2 ms | 418.4 ms | −0.05% | — | |
| AoT tp4, compile cold | 1238.5 s | 1480 s | **−16.3%** | **yes** | See note 4. |
| AoT tp4, e2e warm | 75.4 s | 76.9 s | −2.0% (−1.95%) | — (at the threshold) | |
| AoT tp4, e2e cold | 504.2 s | 505.9 s | −0.3% | — | |

Notes on the drift rows:

1. **Native tp4 step, +1.8%.** The 6.4 ms is a steady per-step offset: every warm step is between 358 and 361 ms. The output is no longer bit-identical to 2026-10-19: the PNG is `b1906d53…` against `a300a78e…`, so the compiled DiT itself changed. *Inference:* the cause is the toolchain and host change, not the Difflet code. Measured facts behind that:
   - This report runs on a different trn2.3xlarge instance. Its `/home/ubuntu` was created on 2026-10-19 at 22:11 UTC (`stat`), and the 2026-10-19 campaign's evidence directory, `/home/ubuntu/bench-neuron-final`, is not on it.
   - The native venv is different. Then: Python 3.14.8, neuronx-cc 2.28.4568.0, torch-neuronx 2.14.3.0.0. Now: Python 3.12.3, neuronx-cc 2.28.7631.0, torch-neuronx 2.14.3.0.1.
   - The Neuron runtime is different. Then: runtime 2.35.27.0, collectives 2.35.40.0. Now: 2.35.30.0, 2.35.43.0.
   - The earlier venv is not on this host, so no A/B was possible. `61022ec`, the commit tested on 2026-10-19, is not in this clone's history, so the code diff cannot be listed either.
2. **Native tp4 compile, +15.6%.** The whole extra time is in the generate launch: 91.2 s against 68.4 s. The vae launch is level (93.8 s against 93.0 s).
   - Within the generate launch, torchrun took 13.85 s to start rank 0, against 2.66 s in the `--sp` compile, and "build on meta" took 9.71 s against 2.77 s (`native_tp4/compile` vs `native_tp4_sp/compile`, `phase_seconds`).
   - *Inference:* the page-cache drop before the pre-read also evicted the venv's libraries, and the pre-read only brings back the checkpoint. The cold e2e, also run straight after a drop, shows the same 14.3 s rank start.
   - The `--sp` compile, with the page cache as left, took 162.6 s.
3. **Native tp4 warm e2e, +3.4%: launch jitter, plus 0.4 s of denoise.**
   - Rank 0's start-to-load time was in its slow mode in all three timed runs: 21.5, 22.4 and 24.4 s. In gen0 it was 11.7 s, and that run took 72.5 s.
   - Against the 2026-10-20 gen2 breakdown, the parts are within 1.5 s each (table below). The denoise is 18.0 s against 17.6 s.
4. **AoT compile, −16.3%.** The 1480 s is *reused* from an A/B at `03f3a84` (`neuron_feature_matrix.md`) and has no breakdown, so the difference cannot be attributed to a part.
   - This run is not a cache hit. It started with an empty pre-sharded store and an empty neuronx-cc cache in the fresh root, it wrote 39 GB of shards ("Done Sharding weights in 361.7"), and it compiled every NEFF, about 122 MB of new cache entries.
   - *Inference:* the checkpoint was in the page cache here (Method). The page-cache state of the 2026-10-17 run is unknown.

### Where the warm e2e goes (gen2, seconds)

Times come from the probe events in `B/qwen_image/<cell>/gen2/summary.json`; `B/scripts/qbreakdown.py` prints them. The brackets hold the 2026-10-20 native gen2 breakdown (*reused*, `neuron_final.md`).

| part | native (`native_tp4/gen2`) | AoT (`aot_tp4/gen2`) |
|---|---:|---|
| process wall | 82.1 [78.4] | 76.4 |
| text | (in the generate launch) | text stage 16.4 (Qwen2.5-VL TP4 on the device) |
| generate launch / stage, total | 61.6 [60.2] | 41.2 |
| — torchrun start to rank 0 start | 2.7 | — |
| — rank 0 start to load start (imports, runtime, process group) | 22.4 [21.7] | — |
| — load + warm-up | 16.1 [15.3]: build on meta 2.8, DiT checkpoint 3.7, warm-up forward 8.3, host Qwen2.5-VL load 0.9 | DiT load 15.2 |
| — host encode (rank 0) | 0.5 [0.5] | — |
| — denoise, 50 steps | 18.0 [17.6] | 21.1 |
| — rest (save latents, teardown; AoT: imports and setup) | 2.0 | 4.9 |
| vae launch / stage, total | 19.2 [18.0]: launch 2.5, imports 3.1, load 9.5, decode 2.5, PNG 0.3, exit 1.2 | 14.4 (compiled VAE, one core; PNG 0.3) |
| gaps between launches, driver exit | 1.3 | 4.4 |

- **Native's denoise and encode are faster; what it loses is launch time.** It spends 22.4 s from rank start to load start, and 5.6 s of its vae launch goes to launching and importing before the decoder loads.
- **AoT's text stage costs 16.4 s and native's host encode costs 0.5 s.** Native loads the host Qwen2.5-VL inside its load phase, in 0.9 s warm.

### Native per cell

| metric | `native_tp4` (headline) | `native_tp4_sp` |
|---|---:|---:|
| DiT step median / p10 / p90 (ms) | 359.0 / 358.2 / 360.1 (step 0: 0.36 s) | 331.0 / 330.4 / 332.5 |
| e2e warm (s) | 81.4, 82.1, 83.4 (mean 82.3; gen0 72.5) | 77.1, 69.6 (mean 73.3; gen0 79.6) |
| e2e cold, page cache dropped (s) | 502.0: generate launch 480.1 (rank start 14.3, imports 13.3, load + warm-up 429.6, of which DiT checkpoint 310.9 and host Qwen2.5-VL 99.6), vae launch 19.8 | not run (extra cell) |
| compile cold (s) | 187.0 (checkpoint pre-read: 100% resident) | 162.6 (page cache as left: 94% resident) |
| compile: generate launch / vae launch (s) | 91.2 / 93.8 (the vae warm-up decode fills the eager kernel cache: 85.2–86.6 s) | 69.3 / 91.9 |
| NEFFs in the root after compile | 135 | — |
| peak HBM allocated / neuron-monitor per DiT core; vae core (GiB) | 10.15 / 10.81; 9.30 / 13.85 | 10.00 / 10.63; 9.30 / 13.83 |
| weight store | not written ("no gain": 10.22 GB either way, bf16 checkpoint) | same |
| output | PNG `b1906d53…` and latents.pt `4da4c89e…` in the cold run and all 4 warm runs | PNG `cf5d5e02…` and latents.pt `fc6d04c4…` in all 3 runs |
| logs | `B/qwen_image/native_tp4/` | `B/qwen_image/native_tp4_sp/` |

SP takes 7.8% off the step (331.0 vs 359.0 ms) and 0.15 GiB off the allocated HBM. Its warm e2e is 9.0 s lower on average, but most of that is launch jitter: gen2 had the fast 11.9 s launch.

### Outputs

- **All three cells' outputs show the prompt's content.** One PNG was opened per cell (`native_tp4/cold`, `native_tp4_sp/gen1`, `aot_tp4/cold`): each shows a red fox in snow in front of a snowy forest.
  - The native PNG has pixel std 52.8 and mean neighbour difference 1.73 levels; AoT's has 70.3 and 1.67. These are natural images, not noise.
  - Within each cell, every run produced the same PNG and the same latents.pt, bit for bit (hashes in each `summary.json`).
- **The native image is no longer the 2026-10-19 golden.** That was PNG `a300a78e…` and latents.pt `aac25966…` (*reused*, `neuron_final.md` A1). Now: `b1906d53…` and `4da4c89e…`. See drift note 1.
- **AoT's composition differs from native's, and the seed-42 noise explains it.** AoT shows a fox walking across open snow; native shows one trotting between trees.
  - Measured on this host: `torch.randn((1, 1, 16, 128, 128), dtype=torch.bfloat16)` after seeding 42 gives different values in the two venvs. The first values are 1.930 in torch 2.14 and −0.809 in torch 2.9.
  - The fp32 draw is identical in both.
  - *Inference:* the pipeline's `prepare_latents` draws the noise in its bf16 dtype, so the two backends start from different noise. The SP image matches the tp4 image to the eye; its PNG differs because SP changes the reduction order.

## Method

The method is `neuron_final.md`'s "Method", with these differences:

- **Observer probe.**
  - The campaign probe that report names (`scripts/probe/sitecustomize.py`) is not on this host. It is not in `/home/ubuntu/campaign`, in any git worktree, or in `git log --all -- '*sitecustomize*'`; this was checked for the HunyuanVideo stage, `hunyuan_video_neuron.md`.
  - The equivalent timestamp-only observer written then, `B/probe/sitecustomize.py`, was extended additively with Qwen-Image targets. The file as the HunyuanVideo cells used it is kept as `B/probe/sitecustomize.hv1-as-used-2026-10-25.py`.
  - Each wrapper calls the original with the same arguments and returns its result unchanged: no argument, tensor or control flow changes.
  - Qwen-Image targets:
    - `QwenImageOrchestrator._scheduler_step`: a stamp after each step, on the native ranks and in the AoT generate stage alike;
    - `._denoise`;
    - `TorchNeuronQwenImageApplication.load` (with a read-only look at its `phase_seconds`) and `.encode_prompt`;
    - `NeuronQwenImageApplication.load` (AoT);
    - the vae stage decoder's `load` and call;
    - the CLI's `_write_image`;
    - the CLI class's `_stage_text`, `_stage_generate` and `_stage_vae`.
  - At exit, each process writes its events and, on a neuron rank, `torch_neuronx.max_memory_allocated()`.
- **DiT step** = median / p10 / p90 of rank 0's inter-step deltas of the real loop, step 0 excluded (49 deltas per run). For AoT the deltas come from the generate stage process.
  - The table value is the median of the per-run medians of the timed runs; p10 / p90 are pooled over the timed runs.
- **Runs per cell.**
  - **Compile:** a cold `difflet compile` on a fresh root, `B/roots/qwen_<cell>`. For native it is also exported as `DIFFLET_COMPILE_CACHE`. For AoT, the harness also sets `DIFFLET_SHARED_WEIGHTS_DIR=<root>/_shared_weights` and `NEURON_COMPILE_CACHE_URL=<root>/neuron-compile-cache`, the same in every run of the cell.
    - Without those two settings, the AoT compile would have reused the Qwen-Image tp4 shards already in `~/.cache/difflet/_shared_weights/` and the entries in `/var/tmp/neuron-compile-cache`. That is what made the HunyuanVideo AoT compile a lower bound.
  - **Page-cache-warm compile (the two headline cells):** drop the page cache, read the checkpoint (transformer, text_encoder, vae: 53.7 GiB) twice, then compile.
    - The first read took 439 s, the second 23–26 s.
    - `mincore` showed 100% residency before the compile (`preread_rec` in each compile's `summary.json`).
    - The `--sp` compile ran with the page cache as the previous run left it: 94% of the checkpoint resident.
  - **Cold e2e (the two headline cells):** a generate with the page cache dropped first (`sync; echo 3 > /proc/sys/vm/drop_caches` through passwordless sudo; `Cached` 0.2 GiB before).
  - **Warm runs:** one discarded warm generate (gen0), then n timed warm generates: n=3 for the headline cells, n=2 for `--sp`.
- **e2e** = the process wall of `python -m difflet.cli.main generate` (harness clock; `/usr/bin/time -v` agrees to 0.01 s).
- **HBM** = `torch_neuronx.max_memory_allocated()` per rank (max over ranks, native only), plus neuron-monitor's per-core peak of `memory_used` (sum of its categories) from `nmon.jsonl`.
- **No exported timeout:** `DIFFLET_NEURON_PG_TIMEOUT_S` is unset in every run (the harness removes it).
- **One device job at a time.** Before each run, the harness waited until `fuser /dev/neuron0` showed no process. After each run, it checked that every process had released the device: `device_released: true` in all 16 `summary.json` files.

## Open issues

1. **Native launch jitter** (`neuron_final.md` open issue 2, unchanged). Rank 0's start-to-load time is bimodal, about 11–12 s or 19–24 s: 11.7, 21.5, 22.4 and 24.4 s in `native_tp4` gen0–3, and 19.1 and 11.9 s in `native_tp4_sp` gen1–2. At 79–82 s per run, that is the whole native vs AoT warm gap.
2. **Native's step is 1.8% slower than on 2026-10-19 and its output hash changed.** The suspected cause is the toolchain (neuronx-cc 2.28.4568 → 2.28.7631) and host change. It is not confirmed: the earlier venv is not on this host. An A/B with the two compilers on one host would settle it.
3. **The native vae launch costs 19 s for a 2.5 s decode** (gen2: launch 2.5, imports 3.1, load 9.5, exit 1.2). AoT's vae stage takes 13–14 s. A resident decoder or a lighter launch would close most of the remaining warm gap.

## Reproduction

```bash
# native: the native venv's bin/ on PATH (neuronx-cc is spawned by name), this worktree on PYTHONPATH
R=<fresh root>
DIFFLET_COMPILE_CACHE=$R HF_HUB_OFFLINE=1 python -m difflet.cli.main compile --backend neuron \
  --model-id Qwen/Qwen-Image --revision 75e0b4be04f60ec59a75f475837eced720f823b6 \
  --height 1024 --width 1024 --tp-degree 4 [--sp] --cache-dir $R
DIFFLET_COMPILE_CACHE=$R HF_HUB_OFFLINE=1 python -m difflet.cli.main generate --backend neuron \
  --model-id Qwen/Qwen-Image --revision 75e0b4be04f60ec59a75f475837eced720f823b6 \
  --height 1024 --width 1024 --tp-degree 4 [--sp] --steps 50 --guidance-scale 4.0 --seed 42 \
  --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.png --cache-dir $R
# AoT: the NxD venv (/home/ubuntu/venvs/difflet-neuron), --backend trainium, no DIFFLET_COMPILE_CACHE;
#   DIFFLET_SHARED_WEIGHTS_DIR=$R/_shared_weights NEURON_COMPILE_CACHE_URL=$R/neuron-compile-cache for a
#   compile with nothing pre-sharded or cached; pass --steps 50 (the AoT Qwen-Image stage defaults to 4)
```

Harness: `B/scripts/qbench.py cell --cell <cell> --venv native|aot [--preread] --runs compile,cold,gen0,gen1,gen2[,gen3] -- <flags>`. It adds the model, the revision, the shape and `--cache-dir B/roots/qwen_<cell>`; for a generate it also adds the steps, guidance, seed, prompt, output and `--work-dir <run>/work --keep-work-dir`. `B/scripts/qbreakdown.py` prints the breakdowns and cell statistics.

## Toolchain

- **Native:** `/home/ubuntu/venvs/native-312`: Python 3.12.3, torch 2.14.0, torch-neuronx 2.14.3.0.1, neuronx-cc 2.28.7631.0.
- **AoT:** `/home/ubuntu/venvs/difflet-neuron`: torch 2.9.1, torch-neuronx 2.9.0.2.15.32035, neuronx-cc 2.26.6360.0.
- **Host runtime:** aws-neuronx-runtime-lib 2.35.30.0, collectives 2.35.43.0.
- **2026-10-19/20 (*reused*, `neuron_final.md`):** native Python 3.14.8, neuronx-cc 2.28.4568.0, torch-neuronx 2.14.3.0.0; runtime 2.35.27.0, collectives 2.35.40.0.
