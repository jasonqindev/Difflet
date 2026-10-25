# HunyuanVideo 1.0 on the PyTorch-native neuron backend vs the Trainium AoT backend (trn2.3xlarge)

**Model:** `hunyuanvideo-community/HunyuanVideo` at revision `e8c2aaa66fe3742a32c11a6766aecbf07c56e773` (diffusers layout; transformer in bf16)  
**Backends:** neuron (TorchNeuron: torchrun, one process per NeuronCore, per-block `torch.compile`) and the Trainium AoT backend (NxD), the same `difflet` CLI on the same host  
**Device:** trn2.3xlarge: 1 device, 4 NeuronCores (LNC=2, 24 GiB HBM per core), 12 vCPU, 124.8 GiB RAM, one EBS root volume  
**Workload:** 320x512x61 (16 latent frames, 10,240 image tokens; one cell at 720x1280x129), 50 steps, guidance 6.0 (embedded: HunyuanVideo is guidance-distilled, one DiT forward per step), seed 42, prompt "a cinematic shot of a red fox running through a snowy forest", `HF_HUB_OFFLINE=1`, pinned revision  
**Campaign:** 2026-10-25 00:18 – 05:30 UTC (TP4, AoT, VAE) and 09:24 – 13:35 UTC (the other layouts, 720p and the fix checks), branch `feat/hunyuan-neuron`. Native cells at `b38f3bc` (the tp4 default and `--device-vae` cells), `d6ffb9a` (the later TP4 cells, AoT included; that commit only flips the neuron decode default, see "Decode placement"), `4c38a58` (TP2, TP2xCP2 and the first 720p attempt; the commits between `d6ffb9a` and it change only benchmark docs), then `f4d9ac1`, `15b503f`, `ce9d212` and `195fc8c` (the fixes found at 720p, each checked with the runs named in its section). Clean tree up to `4c38a58` (`dirty=0` in each `run.log`); from `f4d9ac1` on, every run shows `dirty=1`. The tree state was recorded only during the `f4d9ac1` runs: this doc, the evidence doc and the feature catalog being edited, no code (`B/hunyuan_video/git_status_during_f4d9ac1_runs.txt`). For the `15b503f`, `ce9d212` and `195fc8c` runs no status was recorded, so "no code" is unverified there; only the `15b503f` 720p log records an attention call shape (`size=(3, 119168, 128)`, as that commit's code would make); the `ce9d212` and `195fc8c` logs record no shapes  
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
- **After `f4d9ac1`** (NKI trace cache, section below) native's warm e2e at tp4 is 222.5 s (one timed run, `tp4_tracecache/gen1`), 89.6 s behind AoT instead of 106 s; the step and the output are unchanged.
- **Where native could close the warm gap:** a compiled (traced or per-block `torch.compile`) decoder instead of the eager one (open issue 1). The DiT, the host encode and the launch are already level.

## Native layouts at 320x512x61: TP2, TP4, TP2xCP2

The three layouts the neuron CLI runs on one trn2.3xlarge at the default shape, each with the default placement after `d6ffb9a` (host encode on rank 0, decode in the one-core fp32 vae launch). The CP cell uses gather_kv: it was the fastest CP mode in the stage-2 DiT checks (full depth, 5x320x512: gather_kv 649.7, ring 672.7, ulysses 1148.3 ms per forward, *reused*, `docs/verification/hunyuan-video-neuron-evidence.md`). Measured at `4c38a58` (no code change since `d6ffb9a`; `dirty=0` in every `run.log`), 2026-10-25 09:24–10:25 UTC. TP4 is the `tp4_devvae` cell above (*reused* from this report, n=3).

| metric | TP2 (`tp2`) | TP4 (`tp4_devvae`, *reused*) | TP2xCP2 gather_kv (`tp2cp2_gather`) |
|---|---:|---:|---:|
| cores | 2 | 4 | 4 |
| DiT step median / p10 / p90 | **1461.8** / 1460.2 / 1463.5 ms (runs 1461.9, 1461.7) | **808.3** / 806.7 / 809.9 ms | **905.1** / 903.7 / 906.4 ms (runs 904.7, 905.4) |
| denoise, 50 steps (gen1) | 73.1 s | 40.4 s | 45.2 s |
| e2e warm | **300.9 s** (300.1, 301.7; gen0 290.3) | **239.1 s** (241.3, 233.7, 242.1) | **250.0 s** (249.2, 250.8; gen0 248.9) |
| e2e cold, page cache dropped | **629.4 s** | **570.4 s** | **567.8 s** |
| compile cold, fresh root: generate stage (DiT) | **559.1 s** (checkpoint not in the page cache: load 194.7 s + host Llama/CLIP load 123.1 s; warm-up forward with 118 neuronx-cc compiles 186.1 s) | 177.3 s (page cache warm) | **172.8 s** (page cache warm: load 5.9 s + 15.2 s; warm-up forward with 127 compiles 120.7 s) |
| compile: vae stage | seeded, see below | 1752.3 s + 927 s exit lag | seeded, see below |
| load + warm-up per warm generate (of which the warm-up forward) | 84.7 s (63.5 s) | 57.0 s (36.5 s, from `tp4_teacache2/gen1`: `tp4_devvae` recorded no `phase_seconds`) | 59.6 s (36.8 s) |
| peak HBM per DiT core: alloc / neuron-monitor | 16.18 / 18.71 GiB | 11.67 / 12.98 GiB | 15.89 / 18.00 GiB |
| vae core: alloc / neuron-monitor | 11.53 / 20.89 GiB | 11.53 / 20.89 GiB | 11.53 / 20.89 GiB |
| output | fox running through snow (`tp2/gen0/mid.png`); mp4 `8f21c67f…`, latents `3d0b647e…`, identical in all 4 generates | fox running through snow; latents `b62d0891…` | fox running through snow (`tp2cp2_gather/cold/mid.png`); mp4 `5231314c…`, latents `b50ce0c8…`, identical in all 4 generates |
| final latents vs TP4's | rel-L1 5.39e-3, cos 0.999978 | — | rel-L1 8.11e-3, cos 0.999922 |
| logs | `B/hunyuan_video/tp2/{compile,cold,gen0,gen1,gen2}/` | `B/hunyuan_video/tp4_devvae/` | `B/hunyuan_video/tp2cp2_gather/{compile,cold,gen0,gen1,gen2}/` |

Latent comparison: `B/hunyuan_video/layout_latents_vs_tp4.txt` (gen1 of each cell). Statistics: `B/scripts/breakdown.py --cell B/hunyuan_video/<cell> gen1,gen2`; load phases and compile counts: `B/scripts/hvinfo.py <run dirs>`.

- **TP4 is the fastest layout: 808.3 ms per step, against 905.1 ms for TP2xCP2 gather_kv (+12%) and 1461.8 ms for TP2 (+81%).** On the same four cores, splitting the 10,240 image tokens over two CP ranks and the heads over two TP ranks costs 97 ms per step more than splitting the heads over four. TP2 on two cores is 1.81x TP4's step, so TP4 scales at 90% of linear from TP2.
- **The warm e2e is dominated by the parts that do not depend on the layout.** The vae launch costs the same in all three (launch, load 11.7–12.3 s, decode 73.7–73.9 s, write 21.9–22.1 s), so TP2's 33 s longer denoise and 28 s longer load + warm-up make it 62 s slower than TP4 end to end (300.9 vs 239.1 s), and TP2xCP2 is 11 s slower (250.0 s).
- **The steps agree with the DiT check's timing-only forwards** at this shape: TP2 1460.8 ms, TP4 805.6 ms (*reused*, `J/dit_f61_tp{2,4}_timing.log`). Peak allocated HBM at TP2 is the same 16.18 GiB as there. By neuron-monitor the fullest DiT cores are TP2's at 18.71 GiB (5.29 GiB of 24 free), and the vae core peaks at 20.89 GiB (3.11 GiB free).
- **Compile cold.** The TP2 figure includes 318 s of cold checkpoint reads: the cell before it on this host (Wan2.1) had left other files in the page cache (`Cached` 40.2 GiB before the run; *inference* from the 194.7 s checkpoint load, which matches the dropped-cache runs' 195 s: none of it was this checkpoint). Without them, TP2's DiT compile is about the warm-up forward's 186 s, in line with TP4's 177 s and TP2xCP2's 173 s. *Inference:* the TP2 warm-up forward takes longer than TP4's because each of its 60 block graphs is twice as large per core.
- **The vae stage was not recompiled per layout.** Its stage identity has no layout in it (`hunyuan_video_vae`: model, fp32, shapes, toolchain, backend, eager), and the decoder runs on one core whatever the DiT layout, so its 3,751 eager NEFFs are the same files for every layout. Each cell compiled its DiT with `--host-vae` on a fresh root, then `B/scripts/seed_vae.sh <cell>` copied the vae stage's manifest and the NEFFs of `roots/tp4_devvae` into the cell's root without overwriting anything (`B/hunyuan_video/<cell>/seed_vae.log`: 118 → 3,898 NEFFs for `tp2`, 127 → 3,905 for `tp2cp2_gather`). A user compiling a new layout from scratch with the device decode also pays the vae's 1752 s compile and the 927 s exit (open issues 1 and 2), once per cache root.
- **The first generate after a compile can compile a small graph.** `tp2/cold` ran one neuronx-cc subprocess (5.7 s) right after the host encode, as `tp4_default/cold` and `tp4_devvae/cold` did (three subprocess records over 9 s, on one rank); `tp2cp2_gather/cold` ran none (its root holds `tp4_devvae`'s NEFFs, seeded for the vae, which include the ones `tp4_devvae/cold` compiled); no warm generate compiles anything, and `unwarmed_shapes` is empty in every run (`B/scripts/hvinfo.py <run>`). Which graph it is was not established; the probe records only neuronx-cc's generic command line (open issue 6).
- **Launch jitter is the same as at TP4:** rank 0's start-to-load time is 9.8–21.0 s across these runs (open issue 2 of `neuron_final.md`).

## 720x1280x129 at TP4: BLOCKED (the DiT's block graph does not compile on this host)

720x1280x129 is 33 latent frames x 45 x 80 = 118,800 image tokens, 119,056 joint tokens with the 256 text tokens (119,168 padded to a multiple of 128 for the bounded kernel), 11.6x the default shape. It ran at TP4, the fastest layout at 320x512x61 (above); the other layouts that fit four cores shard attention no further (TP2xCP2 gather_kv and ulysses give each rank the same heads x queries x keys product as TP4). Each attempt was `difflet compile --backend neuron --tp-degree 4 --host-vae` on a fresh root (the DiT stage only, so a DiT that does not compile fails before the vae's eager compile). **No attempt reached a forward, so there is no step time, e2e or activation HBM at 720p.** Every attempt loaded the DiT (11.83 GiB allocated per core, 12.29 GiB by neuron-monitor, before any block ran) and compiled the 102 small graphs (embedders, refiner, norms) in its first 75 s; the failure is in the 60 blocks' graph.

| attempt | HEAD | wall (s) | outcome | evidence (`B/hunyuan_video/`) |
|---|---|---:|---|---|
| 1 | `4c38a58` | 1806.0 (killed) | every rank at 100% CPU with no neuronx-cc running for 30 min after the small graphs; py-spy: nki tracing nkilib's `attention_cte` (`_qk_and_max_large_tile_impl`) under torch-neuronx's `get_kernel_config` in fake-tensor mode, from `flash_attention_bounded`. The trace is repeated in every process unless torch-neuronx's NKI trace cache keeps it, and HunyuanVideo 1.0's CLI did not opt in. **Fixed by `f4d9ac1`** (opt in, as HunyuanVideo 1.5 does). | `tp4_720p.killed_nki_trace/compile/`, `.../compile_pyspy_rank1_blocking.txt` |
| 2 | `f4d9ac1` | 2702.8 | after about 44 min of tracing, on every rank: `flash_attention_bounded(FakeTensor(size=(6, 119168, 128)), ...)`: `BIR emission failed ... error: colz: body blob exceeds uint32 offset range`. 6 heads x 119,168^2 = 8.5e10, twice nkilib's tested `_MAX_BS_TIMES_SEQLEN_QK` = 32 x 36864^2 = 4.35e10 (the kernel only warns above it, as the log does). | `tp4_720p.blob_overflow/compile/run.log` |
| 3 | `15b503f` (heads split so each call stays within the tested size: 2 calls of 3 heads) | 1476.6 | the same error at `size=(3, 119168, 128)` (4.26e10) after about 23 min of tracing: the tested size is not a safe bound for the bounded path at this length | `tp4_720p.blob_overflow_3heads/compile/run.log` |
| 4 | `ce9d212` (at most half the tested size per call: 6 calls of 1 head, 1.42e10) | 1305.7 | **the kernel's BIR emission passes** (trace about 5 min); then neuronx-cc compiling the block graph (`walrus_driver`, -O2) grew to 109.8 GiB of host memory (dmesg `anon-rss:115166160kB`) and the kernel OOM-killed it: `[F137] neuronx-cc was forcibly killed - This most commonly occurs due to insufficient system memory`, exit code 70, on all 4 ranks | `tp4_720p.host_oom_1head/compile/run.log`, `tp4_720p.host_oom_1head/dmesg_oom.txt`, `tp4_720p.host_oom_1head.mem.log` |

- **Where it stops:** the block graph with six one-head attention calls at 119,168 tokens needs more than the host's 124.8 GiB (MemAvailable 121.8 GiB before the run) to compile with neuronx-cc 2.28. This is a limit of the host and the compiler, not of HBM: the weights take 11.83 of 24 GiB per core, and the activations were never allocated.
- **The fixes kept from these attempts:** `f4d9ac1` (measured: it also takes 23 s off every launch at 320x512x61, section below); `15b503f`, `ce9d212` and `195fc8c` (the head split: it moves the 720p failure from the kernel's NKI compile to neuronx-cc; every shape the repo verified still makes one call on the whole tensors, as before the split, and that path ran end to end at TP4 320x512x61 with bit-identical output; the split path itself has only reached the failed compile above).
- **Not tried (each costs a 20–45 min cold attempt):** a lower neuronx-cc optimisation level for the block graph (it compiled at -O2; the small graph of a first generate, open issue 6, compiled at -O1); attention as a graph of its own, so neuronx-cc compiles one attention call per graph instead of six inside a block; TP2xCP2 ring (each kernel call then sees half the keys, but per-rank image tokens are 59,400, not a multiple of 128, so the ring rule may refuse it); a host with more memory (trn2.48xlarge); or fewer frames at 720x1280.

## NKI trace cache (`f4d9ac1`): the warm-up forward was mostly the attention kernel's trace

The first 720p compile (above) showed every rank tracing the key-bounded `attention_cte` NKI kernel for more than 30 minutes, a trace torch-neuronx repeats in every process unless its NKI trace cache keeps it; HunyuanVideo 1.5's CLI opts into that cache and HunyuanVideo 1.0's did not. `f4d9ac1` makes the neuron launches opt in (`DIFFLET_NKI_TRACE_CACHE=1`). Re-measured at TP4, 320x512x61, on a fresh root (`tp4_tracecache`: compile with `--host-vae`, vae seeded as above, gen0, gen1 at `f4d9ac1`; `dirty=1` in these runs is the uncommitted edits to this doc, the evidence doc and the feature catalog only, `B/hunyuan_video/git_status_during_f4d9ac1_runs.txt`):

| metric | before (`tp4_devvae`, `b38f3bc`) | after (`tp4_tracecache`, `f4d9ac1`) |
|---|---:|---:|
| warm-up forward per generate launch | 36.5 s (from `tp4_teacache2/gen1`, `d6ffb9a`, same root: `tp4_devvae`'s probe did not record `phase_seconds`) | **13.5 s** (gen1), 14.4 s (gen0) |
| load + warm-up | 57.0 s | **34.5 s** |
| e2e warm | 239.1 s (n=3) | **222.5 s** (gen1, n=1; gen0 228.1 s) |
| DiT step median | 808.3 ms | 807.8 ms |
| compile, generate stage, page cache warm | 177.3 s | 179.3 s |
| latents / mp4 | `b62d0891…` / `68adae23…` | the same, bit for bit |
| logs | `B/hunyuan_video/tp4_devvae/` | `B/hunyuan_video/tp4_tracecache/{compile,gen0,gen1}/` |

- **The warm-up forward falls by 23 s and the warm e2e by about 17–19 s; the output does not change.** So open issue 4's "36 s warm-up with every NEFF cached" was mostly the NKI trace, not NEFF loading (the earlier *inference* in "Where the warm e2e goes" is corrected here). The TP2 and TP2xCP2 cells above ran before `f4d9ac1`; *inference:* they gain about as much (their warm-up forwards were 63.5 and 36.8 s).
- The first trace of a kernel shape still costs its full time once, in `compile` (the cache is empty then).
- **Re-checked at the final HEAD (`195fc8c`, after the 720p head-split commits):** a TP4 generate on the same root gives the same latents and mp4 bit for bit (`b62d0891…`, `68adae23…`) at 809.1 ms per step (`tp4_tracecache/gen3_head`; 808.1 ms at `ce9d212` in `gen2_ce9d212`). Both runs compiled the two block graphs once more (69–75 s warm-up forward, 267.4 s and 600.9 s e2e; `gen2_ce9d212` also read the checkpoint cold after the 720p OOM had emptied the page cache), because the compiled graphs carry source-line metadata of `ops_impl/attention.py` and those commits move its lines (*inference*, evidence in the evidence doc's bug ledger, row 4). That is a one-time recompile per cache root; neither run is used for a timing.

## Method

The method is `benchmark/trn2/neuron_final.md`'s "Method", with these differences:

- **Observer probe.** The campaign probe that report names (`scripts/probe/sitecustomize.py`) is not on this host: not in `/home/ubuntu/campaign`, not in any git worktree, not in `git log --all -- '*sitecustomize*'`. An equivalent observer was written for this report, `B/probe/sitecustomize.py`. It only records timestamps: each wrapper calls the original with the same arguments and returns its result; no argument, tensor or control flow changes. It loads in every process through `PYTHONPATH` (the torchrun ranks inherit it; so do the `neuronx-cc` subprocesses, which only write an empty record). It stamps `HunyuanVideoOrchestrator._scheduler_step` (after each step), `_denoise`, `_decode_latents` (AoT decode), the neuron application's `load` and `encode_prompt`, the AoT application's `load`, the VAE decoder's `load` / `decode`, and the CLI's `_write_video` / `_save_video` / `_decode_latents_host`; at exit each process writes its events and, on a neuron rank, `torch_neuronx.max_memory_allocated()`. From 02:50 UTC (the TeaCache, SP and later cells) it also reads the application's own `phase_seconds` after `load` (read-only), which gives the load split.
- **DiT step** = median / p10 / p90 of rank 0's inter-step deltas of the real loop, step 0 excluded (49 deltas per run). The table value is the median of the per-run medians of the timed runs; p10 / p90 are pooled over the timed runs.
- **Runs per cell:** a cold `difflet compile` on a fresh root `B/roots/<cell>` (also exported as `DIFFLET_COMPILE_CACHE` for native), then for the two tp4 headline cells and AoT a generate with the page cache dropped first (`sync; echo 3 > /proc/sys/vm/drop_caches`, passwordless sudo; meminfo before shows `Cached` 0.2 GiB), then one discarded warm generate (gen0), then n timed warm generates: n=3 for the two tp4 headline cells, n=2 elsewhere. The compile runs see the page cache as the previous run left it (`meminfo_before` in each `summary.json`).
- **e2e** = the process wall of `python -m difflet.cli.main generate` (harness clock; `/usr/bin/time -v` agrees to 0.1 s).
- **HBM** = `torch_neuronx.max_memory_allocated()` per rank (max over ranks), plus neuron-monitor's per-core peak of `memory_used` (sum of its categories), from `nmon.jsonl`.
- **Outputs:** every run's mp4 was decoded and checked (61 frames, 320x512, moving content, not a frozen or uniform clip: in the native runs every frame's pixel std is at least 49 grey levels and the mean frame-to-frame difference 32 levels; in the AoT runs, at least 72 and 20.1 levels), and one `mid.png` per cell was opened: each shows a red fox running through snow in a forest. Within a cell every run, cold and warm, produced the same latents and mp4 bit for bit (hashes in `summary.json`; AoT keeps no latents file, its mp4s are identical).
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
- **The warm-up forward costs 36 s of every launch** even with every NEFF cached; the denoise itself is 40 s. *Inference:* most of it is loading the 60 block NEFFs onto the cores (the first decode of a fresh vae process shows the same effect, below). **Corrected after `f4d9ac1`:** most of the 36 s was the NKI trace of the attention kernel, repeated in every process; with the NKI trace cache the warm-up forward takes 13.5 s (see "NKI trace cache").

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

### F2: bf16 VAE precision on the device (2026-10-26)

Raw evidence under `/home/ubuntu/bench-2026-10-26/` (`B26` below); the decoder runs above (`J`, `B`) are unchanged. Native venv and toolchain as in "Toolchain". During these runs another session's unit tests held `/dev/neuron0` open (torch-neuronx opens it at `import torch`; each run's `summary.json` lists the holders); neuron-monitor recorded no runtime but ours.

**Cause: the device's bf16 GroupNorm.** Bisection on the first tile of the tiled decode (`latents[:, :, 0:5, 0:32, 0:32]` / scaling factor through `post_quant_conv` + decoder; forward hooks on all 174 module calls; device bf16 and host bf16 against host fp32; `B26/f2/bisect/`, script `B26/scripts/f2_bisect.py`):

- Accumulated: the tile's output is 0.329 rel-L1 on the device against 3.3e-3 for host bf16. The device stays within 2-6x of the host through the mid-block and up_blocks 0-1 (GroupNorm-led). It jumps at `up_blocks.2.resnets.0.norm1` (0.385 vs 1.8e-2) and `up_blocks.3.resnets.0.norm1` (1.55 vs 1.0e-2).
- Each module alone on the fp32 input cast to bf16: the device's GroupNorm error grows with the elements per group: 6e-3 to 2e-2 at 81,920; 3e-2 to 4e-2 at 327,680; 0.20 to 0.38 at 1.2M to 2.4M; 1.35 to 2.05 at 4.5M to 8.9M. Host bf16 is 2.2e-3 to 3.8e-3 at every size. Conv3d, SiLU, the upsampling and `to_out` match the host exactly (1.00x); `to_q/k/v` are 0.82-0.93x; the attention block is 2.4x, which is its own GroupNorm. *Inference:* the device kernel accumulates the group statistics at bf16 precision; the AoT path's `_Fp32GroupNorm` (`models/hunyuan_video/vae/modeling_vae.py`) works around the same thing.
- Fix `2e04372`: `install_group_norm_fp32` (`core/workarounds.py`, installed by `prepare_runtime`) computes bf16 `F.group_norm` on neuron tensors in fp32. No DiT on the backend uses GroupNorm, so DiT numerics and speed are unchanged by construction.

**Micro-test** (`B26/scripts/f2_gn_micro.py`, `B26/f2/gn_micro.log`): `F.group_norm` alone on one core, 32 groups, random inputs (mean 0 and mean 2) at the decoder's shapes plus the real input to `up_blocks.3.resnets.0.norm1`, rel-L1 against fp32 on the host:

| elements per group | host bf16 | device bf16, raw kernel | device bf16 through `F.group_norm` (the workaround) | device fp32 |
|---:|---:|---:|---:|---:|
| 81,920 | 2.3e-3 / 4.3e-3 | 3.4e-3 / 6.6e-3 | 2.3e-3 / 4.3e-3 | 7.9e-8 / 2.2e-7 |
| 327,680 | 2.4e-3 / 4.3e-3 | 7.0e-3 / 4.3e-2 | 2.4e-3 / 4.3e-3 | 9.3e-8 / 4.5e-7 |
| 1,179,648 | 2.4e-3 / 4.3e-3 | 0.50 / 1.47 | 2.4e-3 / 4.3e-3 | 1.4e-7 / 8.7e-7 |
| 2,359,296 | 2.4e-3 / 4.3e-3 | 1.12 / 2.91 | 2.4e-3 / 4.3e-3 | 2.2e-7 / 9.2e-7 |
| 4,456,448 | 2.4e-3 / 4.3e-3 | 1.91 / 3.21 | 2.4e-3 / 4.3e-3 | 2.4e-7 / 1.7e-6 |
| 8,912,896 | 2.4e-3 / 4.3e-3 | 3.11 / 4.83 | 2.4e-3 / 4.3e-3 | 3.2e-7 / 1.9e-6 |
| 8,912,896, real input | 2.4e-3 | 1.53 | 2.4e-3 | 6.9e-7 |

With the workaround the device's bf16 result equals the host's to within 1e-7 of rel-L1 at every size.

**Full decode** of `J/cli/tp2_host.work/latents.pt` against `J/vae_cpu_fp32.pt`, the same decoder as the table above (`NeuronHunyuanVideoVaeDecoderApplication`, eager, one core) at `2e04372`, through `B26/f1a/scripts/vaebench2.py` with `vae_repeat2.py` (`vae_repeat.py` plus the F1a options, defaults unchanged):

| decode | first decode in a fresh process (s) | later decodes, same process (s) | compile | peak HBM alloc / neuron-monitor (GiB) | vs fp32 host decode | log |
|---|---:|---:|---|---:|---|---|
| device, bf16, `2e04372` | 55.0 (warm root) | 20.2, 20.1; 20.5, 20.6 | first decode on the bf16 root of the run above: 357.0 s, 207 new NEFFs (3,784 → 3,991) | 6.80 / 13.75-13.84 | rel-L1 **3.80e-3**, max abs 0.067, mean abs 0.31 grey levels, PSNR **55.9 dB** (was 20.8 dB) | `B26/f2/f2_neuron_bf16_gnfp32/`, `f2_neuron_bf16_gnfp32_warm/` |
| device, fp32, `2e04372` | 100.4 | 43.2 | none (3,871 NEFFs before and after, on a copy of `tp4_devvae`'s root) | 11.53 / 20.27 | rel-L1 3.48e-7, PSNR 136.7 dB; **bit-identical** to the pre-fix decode (`torch.equal` with `B/hunyuan_video/vae/neuron_fp32.pt`) | `B26/f2/f2_neuron_fp32_gnfp32/` |

- **The device's bf16 decode is now as accurate as bf16 on the host** (55.9 dB, rel-L1 3.8e-3 in both), against the 50 dB gate. Every decode, in both processes, is bit-identical to the first.
- **fp32 is untouched**, as the wrapper only acts on bf16: same output bit for bit, no new kernel. Its times (100.4 / 43.2 s) are 5% above the earlier 95.0-95.9 / 40.7-41.0 s; the other session's tests were using the host CPUs at the time (*inference*: eager dispatch is host-bound).
- **bf16 decodes 2.1x faster than fp32** in the same process (20.1-20.6 s vs 43.2 s); the fp32 statistics cost about 10% against the inaccurate bf16 kernel (18.3-18.8 s before the fix).
- The CLI still builds the decoder in fp32 (amendment A6). A bf16 default goes through the plan's default-change rule.

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
3. **Fixed (`2e04372`): the eager bf16 decoder on a NeuronCore was inaccurate** (20.8 dB against the fp32 decode, visible banding): the device's bf16 GroupNorm loses precision as the group grows. With its statistics in fp32 the device's bf16 decode is at 55.9 dB, as on the host ("F2: bf16 VAE precision on the device"). The CLI still decodes in fp32.
4. **The warm-up forward of every generate launch** was 36.5 s at tp4 with all NEFFs cached (TP2: 63.5 s, TP2xCP2: 36.8 s), about as long as the 50-step denoise. **Mostly fixed by `f4d9ac1`** (the NKI trace cache): 13.5 s at tp4. The remaining 13.5 s was not broken down.
5. **Full-depth DiT parity on the t=900 / 20-token oracle** stays at 1.19x the control for every compiled layout (`docs/verification/hunyuan-video-neuron-evidence.md`); a gating decision is pending with the plan owner.
6. **The first generate on a freshly compiled root can compile a small graph** (one neuronx-cc subprocess of 5.7 s in `tp2/cold`, three records over 9 s in the tp4 cold runs; none in any warm run, `unwarmed_shapes` empty). Not identified; it costs seconds once per root.
7. **720x1280x129 does not compile on trn2.3xlarge** (section "720x1280x129"): with the bounded attention kernel split to one head per call, neuronx-cc's block-graph compile is OOM-killed at 109.8 GiB of host memory (of 124.8 GiB). Untried options and their cost are listed there.
8. **Editing `difflet/backends/neuron/ops_impl/attention.py` recompiles every cached block graph once** (the graphs carry its source lines; *inference* from the NEFFs' debug metadata and the recompiles after `ce9d212` and `195fc8c`). Not a Difflet bug, but a cache that survives code edits would need torch-neuronx to key graphs without source locations.

## Reproduction

```bash
# native: the native venv's bin/ on PATH (neuronx-cc is spawned by name), this worktree on PYTHONPATH
R=<fresh root>
DIFFLET_COMPILE_CACHE=$R HF_HUB_OFFLINE=1 python -m difflet.cli.main compile --backend neuron \
  --model-id hunyuanvideo-community/HunyuanVideo --revision e8c2aaa66fe3742a32c11a6766aecbf07c56e773 \
  --height 320 --width 512 --num-frames 61 --tp-degree 4 [--sp] [--host-vae|--device-vae] --cache-dir $R
#   other layouts: --tp-degree 2, or --tp-degree 2 --cp-degree 2 --cp-mode gather_kv; 720p: --height 720 --width 1280 --num-frames 129
DIFFLET_COMPILE_CACHE=$R HF_HUB_OFFLINE=1 python -m difflet.cli.main generate --backend neuron \
  --model-id hunyuanvideo-community/HunyuanVideo --revision e8c2aaa66fe3742a32c11a6766aecbf07c56e773 \
  --height 320 --width 512 --num-frames 61 --tp-degree 4 [--sp] [--host-vae|--device-vae] \
  [--teacache-cadence 2] --steps 50 --guidance-scale 6.0 --seed 42 \
  --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.mp4 --cache-dir $R
# AoT: the same with the NxD venv (/home/ubuntu/venvs/difflet-neuron) and --backend trainium
#   (no DIFFLET_COMPILE_CACHE); pass --steps 50 (the AoT HunyuanVideo CLI defaults to 4)
```

Harness: `B/scripts/hvbench.py cell --cell <cell> --venv native|aot --runs compile,cold,gen0,gen1,gen2[,gen3] -- <flags>` (`HVBENCH_SHAPE="720 1280 129"` overrides the shape; the file as the TP4 cells used it is `hvbench.tp4-as-used-2026-10-25.py`) (it adds the shape, revision, `--cache-dir B/roots/<cell>`, and for a generate the steps, guidance, seed, prompt, output and `--work-dir <run>/work --keep-work-dir`); `B/scripts/breakdown.py` (breakdowns, cell statistics); `B/scripts/vaebench.py` with `J/vae_dtype.py` or `B/scripts/vae_repeat.py` (VAE); `B/scripts/cmp_mp4.py`, `write_mp4.py`; for the layout cells `B/scripts/seed_vae.sh` (vae NEFFs), `B/scripts/queue_layouts.sh` and `queue_720p.sh` (the run order), `B/scripts/hvinfo.py` (load phases, compiles, HBM).

## Toolchain

- Native: `/home/ubuntu/venvs/native-312`, Python 3.12.3, torch 2.14.0, torch-neuronx 2.14.3.0.1, neuronx-cc 2.28.7631.0, nki 0.7.0, diffusers 0.38.0.
- AoT: `/home/ubuntu/venvs/difflet-neuron`, torch 2.9.1, torch-neuronx 2.9.0.2.15.32035, neuronx-cc 2.26.6360.0, neuronx-distributed 0.19.28492, neuronx-distributed-inference 0.10.18399, diffusers 0.38.0, transformers 4.57.6.
- Host runtime: aws-neuronx-runtime-lib 2.35.30.0, collectives 2.35.43.0, dkms 2.31.7.0, tools 2.33.38.0.
