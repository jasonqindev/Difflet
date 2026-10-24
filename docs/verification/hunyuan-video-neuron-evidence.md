# HunyuanVideo 1.0 on the neuron backend: on-device evidence (DiT parity per layout)

Campaign: 2026-10-24, device runs 21:31-22:35 UTC. Host: `trn2.3xlarge`, 1 Neuron device, 4 logical
NeuronCores (logical config 2), 24 GiB HBM per logical core, 12 vCPUs, 124 GB RAM. Branch
`feat/hunyuan-neuron` at `88c8e48`; no code changed during the campaign.

This doc covers Task 8, stage 2 of `docs/superpowers/plans/2026-10-24-hunyuan-video-neuron.md`.
That stage checks DiT parity on the device at every layout, at truncated depth and at full depth,
and runs the compiled mask-bounds regression. The end-to-end check, the CLI runs and the benchmark
doc are later stages and are not covered here.

Logs, result JSONs, oracles and diagnosis dumps are kept outside the repo, as the job rules
require, under `J = /home/ubuntu/.claude/jobs/28e8164e/tmp/hv1_device/`. Every number below comes
from a file in `J/`. Statements marked *inference* are explanations that no run measured directly.

Statuses: **PASS** (ran on the device in this campaign and met its gate) · **FAIL** (ran and
missed the gate) · **NOT MEASURED** (not run in this stage).

## Summary

* **Truncated depth (2 dual + 4 single blocks), all 8 cases PASS.** The cases are TP2 eager, TP2,
  TP4, TP2+SP, and TP2xCP2 with gather_kv, ulysses and ring (all compiled), plus TP2 at 3 valid text
  tokens. rel-L1 against fp32 diffusers is 0.948x-0.981x the bf16 control when compiled and 1.060x
  in eager mode; the gate is 1.15x. Every rank's output is identical, there are no CPU fallbacks and
  no unwarmed shapes, and all 6 of 6 blocks are compiled.
* **Full depth (20 + 40 blocks), oracle t=900 with 20 valid tokens: every compiled layout FAILs the
  1.15x gate, at 1.188x-1.191x.** That holds for TP2, TP4, TP2+SP and TP2xCP2 with gather_kv,
  ulysses and ring. Cosine is 0.999995. All the other checks pass: identical ranks, no fallbacks,
  60 of 60 blocks compiled. **TP2 eager passes at 1.006x.**
* **The failure is specific to that one input. It is not a layout, sharding or checkpoint bug, and
  this campaign fixed nothing.** We drew eight full-depth samples of the compiled TP2 DiT: t =
  950, 900, 850, 700, 500, 300 and 100 at 20 valid tokens, plus t=900 at 3 valid tokens. Their
  ratios are 0.838, **1.190**, 1.054, 0.940, 0.891, 0.943, 0.998 and 1.049x, with a mean of 0.988x.
  Only t=900/20 fails. Eager mode on the same 7 timesteps at 20 valid tokens gives 0.989-1.125x.
  The excess is confined to compiling the dual-stream block as a whole graph, and to the 12
  massive-activation channels of the residual stream. The diagnosis is below.
* **The other gate fails nowhere.** The cosine floor, finite outputs, identical ranks, zero
  fallbacks and every block compiled all pass in every case, and the cosine is at least 0.999970
  in every case.
* **HBM.** At full depth, peak HBM at 5x320x512 is 15.61 GiB per rank at TP2 and 11.10 GiB at
  TP4. At the default 61x320x512 (10,240 image tokens), timing only, it is 16.18 GiB at TP2 and
  11.67 GiB at TP4. **TP2 fits at the default shape, so amendment A9 (column-parallel modulation)
  is not needed.**
* **Time per full-depth forward (compiled, median of 2x5).** At 5x320x512: TP2 207.9 ms, TP4
  265.0 ms, TP2+SP 207.3 ms, TP2xCP2 gather_kv 649.7 ms, ring 672.7 ms, ulysses 1148.3 ms, and TP2
  eager 1802.1 ms. At 61x320x512: **TP2 1460.8 ms, TP4 805.6 ms.**
* **Compiled mask bounds: 72/72 counts exact on a NeuronCore.** The checks cover the key bounds,
  ulysses `key_valid_len` and ring `text_valid_len`, at 1,536, 10,496 and 20,096 keys.

## Environment

| item | value |
|---|---|
| instance | trn2.3xlarge, `neuron-ls`: 1 device, 4 cores (ids 0-3), 96 GB, logical-neuroncore-config 2 |
| python / venv | 3.12.3, `/home/ubuntu/venvs/native-312` (its `bin/` on PATH: `neuronx-cc` is spawned by name) |
| torch / torch-neuronx | 2.14.0+cu130 / 2.14.3.0.1+b1b6c66 |
| neuronx-cc | 2.28.7631.0+8f7b0cf5 |
| runtime / collectives / dkms | aws-neuronx-runtime-lib 2.35.30.0, aws-neuronx-collectives 2.35.43.0, aws-neuronx-dkms 2.31.7.0 |
| diffusers | 0.38.0 |
| checkpoint | `hunyuanvideo-community/HunyuanVideo` snapshot `e8c2aaa66fe3742a32c11a6766aecbf07c56e773` (transformer in bf16) |
| launch env (added by the scripts) | `HF_HUB_OFFLINE=1 DIFFLET_DISABLE_PREWARM=1 DIFFLET_NEURON_PG_TIMEOUT_S=10800 PYTHONPATH=<worktree>` |

The first attempt (`J/dit_l2s4_tp2_eager.nopath.log`) failed with `posix_spawn failed: No such
file or directory` while compiling the first eager kernel. The interpreter had been called by
absolute path without the venv's `bin/` on PATH, so `neuronx-cc` could not be found. This was a
spurious environment failure, not a Difflet bug. Every later run goes through `J/run.sh`, which
exports the PATH.

## Commands

Every command runs from the worktree, as `J/run.sh <name> <args>`. That is
`PYTHONPATH=. PATH=/home/ubuntu/venvs/native-312/bin:$PATH python <script> <args>`, with the
output in `J/<name>.log` ending in `WALL` and `EXIT` lines. Outside torchrun, the DiT check
starts torchrun itself and prints the exact `launch:` line into the log.

```
# oracles (host CPU, fp32 reference + bf16 control; 5x320x512, 1,280 image tokens, seed 0)
check_neuron_hv1_dit.py --make-oracle J/hv1_dit_l2s4_v20.pt --layers 2 --single-layers 4
check_neuron_hv1_dit.py --make-oracle J/hv1_dit_l2s4_v3.pt  --layers 2 --single-layers 4 --valid-text 3
check_neuron_hv1_dit.py --make-oracle J/hv1_dit_full_v20.pt --threads 6
check_neuron_hv1_dit.py --make-oracle J/hv1_dit_full_v3.pt  --valid-text 3 --threads 6
check_neuron_hv1_dit.py --make-oracle J/hv1_dit_full_v20_3t.pt --timesteps 900 500 100 --threads 6
check_neuron_hv1_dit.py --make-oracle J/hv1_dit_full_v20_4t.pt --timesteps 950 850 700 300 --threads 12
# device, one per layout (O = an oracle above)
check_neuron_hv1_dit.py --oracle O --tp-degree 2 --exec-mode eager   --out-json J/<name>.json
check_neuron_hv1_dit.py --oracle O --tp-degree 2 --exec-mode compile --out-json J/<name>.json
check_neuron_hv1_dit.py --oracle O --tp-degree 4 --exec-mode compile --out-json J/<name>.json
check_neuron_hv1_dit.py --oracle O --tp-degree 2 --sp                --out-json J/<name>.json
check_neuron_hv1_dit.py --oracle O --tp-degree 2 --cp-degree 2 --cp-mode {gather_kv,ulysses,ring} --out-json J/<name>.json
check_neuron_hv1_dit.py --oracle none --tp-degree {2,4} --frames 61 --iters 5 --out-json J/dit_f61_tp{2,4}_timing.json
check_neuron_hv1_mask_bounds.py --core 0 --work-dir J/mask-bounds
```

Oracle costs, from `J/oracle_*.log`:

| oracle | depth | t | control rel-L1 | load / fp32 / bf16 (s) | max RSS |
|---|---|---|---|---|---|
| `hv1_dit_l2s4_v20.pt` | 2+4 | 900 | 6.821e-03 | 10.8 / 2.7 / 1.8 | 7.4 GiB |
| `hv1_dit_l2s4_v3.pt` | 2+4 | 900 | 6.787e-03 | 0.4 / 2.7 / 1.9 | 7.4 GiB |
| `hv1_dit_full_v20.pt` | 20+40 | 900 | 2.515e-03 | 173.4 / 42.1 / 22.5 | 51.7 GiB |
| `hv1_dit_full_v3.pt` | 20+40 | 900 | 2.452e-03 | 5.1 / 43.5 / 22.1 | 51.7 GiB |
| `hv1_dit_full_v20_3t.pt` | 20+40 | 900/500/100 | 2.515 / 3.105 / 3.789e-03 | 5.0 / 130.0 / 54.7 | 51.7 GiB |
| `hv1_dit_full_v20_4t.pt` | 20+40 | 950/850/700/300 | 3.083 / 2.852 / 3.307 / 3.013e-03 | 5.2 / 118.2 / 42.8 | 51.7 GiB |

The low load times on the later full-depth oracles are a page-cache effect (*inference*: the first
full load read the 25.6 GB checkpoint from disk).

## Results

Columns: ours and the control are rel-L1 against fp32 diffusers. "ratio" is ours / control, and
the gate is at most 1.15x. `err_cos` is the cosine between our error and the control's error.
"load+warmup (warm-up fwd)" is seconds to build, load the checkpoint, compile and run the warm-up
forward. "compiles" is torch-neuronx's `TotalCompilations` counter. ms/fwd is the median device
forward over 2 rounds of 5 (`--iters 5`). Peak HBM is the maximum over ranks. Logs are in `J/`.
Every row also passed: checkpoint (0 missing, 0 unexpected), fallbacks (none in warm-up, forwards
or benchmark), unwarmed shapes (none), identical ranks (output sha256 equal on every rank),
compiled blocks (all, compile mode) or no compiled blocks (eager), and finite output.

**Compile time was measured on a warm cache.** The persistent NEFF cache already held most of
these graphs: the first compiled run reports `PersistentHits` 111 with 1 compilation. The
load+warmup column is therefore the warm figure. The only runs that compiled new block graphs are
the 61-frame timing runs: 16 compilations and 174.0 s warm-up at TP2, 6 compilations and 93.9 s at
TP4. A cold compile was not measured.

### Truncated depth: 2 dual + 4 single blocks, 5x320x512, t=900

| layout | sample | gate | ours | control | ratio | cosine | err_cos | load+warmup (warm-up fwd) s | compiles | ms/fwd | peak HBM GiB | log |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TP2 eager | vt=20 | **PASS** | 7.231e-03 | 6.821e-03 | 1.060x | 0.999970 | 0.724 | 103.8 (74.3) | 193 | 205.5 | 2.21 | `dit_l2s4_tp2_eager.log` |
| TP2 compile | vt=20 | **PASS** | 6.498e-03 | 6.821e-03 | 0.953x | 0.999976 | 0.618 | 39.7 (38.1) | 1 | 40.6 | 2.02 | `dit_l2s4_tp2_compile.log` |
| TP4 compile | vt=20 | **PASS** | 6.521e-03 | 6.821e-03 | 0.956x | 0.999976 | 0.595 | 30.8 (19.6) | 7 | 42.6 | 1.48 | `dit_l2s4_tp4_compile.log` |
| TP2 + SP compile | vt=20 | **PASS** | 6.694e-03 | 6.821e-03 | 0.981x | 0.999975 | 0.637 | 20.3 (19.1) | 1 | 39.3 | 2.04 | `dit_l2s4_tp2_sp.log` |
| TP2xCP2 gather_kv | vt=20 | **PASS** | 6.464e-03 | 6.821e-03 | 0.948x | 0.999977 | 0.616 | 23.0 (21.6) | 3 | 80.4 | 1.98 | `dit_l2s4_tp2cp2_gather.log` |
| TP2xCP2 ulysses | vt=20 | **PASS** | 6.467e-03 | 6.821e-03 | 0.948x | 0.999977 | 0.622 | 24.7 (23.2) | 1 | 134.2 | 2.00 | `dit_l2s4_tp2cp2_ulysses.log` |
| TP2xCP2 ring | vt=20 | **PASS** | 6.486e-03 | 6.821e-03 | 0.951x | 0.999976 | 0.590 | 30.1 (28.6) | 0 | 85.8 | 2.00 | `dit_l2s4_tp2cp2_ring.log` |
| TP2 compile | vt=3 | **PASS** | 6.496e-03 | 6.787e-03 | 0.957x | 0.999977 | 0.624 | 6.2 (4.9) | 0 | 39.1 | 2.02 | `dit_l2s4_tp2_v3.log` |

`skipped=1080` in each checkpoint line is the truncated blocks' keys, which are left out on
purpose (amendment A2).

### Full depth: 20 dual + 40 single blocks, 5x320x512, t=900 (plan Step 2 cases)

| layout | sample | gate | ours | control | ratio | cosine | err_cos | load+warmup (warm-up fwd) s | compiles | ms/fwd | peak HBM GiB | log |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TP2 eager | vt=20 | **PASS** | 2.530e-03 | 2.515e-03 | 1.006x | 0.999996 | 0.680 | 8.6 (3.8) | 0 | 1802.1 | 15.79 | `dit_full_tp2_eager.log` |
| TP2 compile | vt=20 | **FAIL** | 2.994e-03 | 2.515e-03 | 1.190x | 0.999995 | 0.495 | 10.3 (5.0) | 0 | 207.9 | 15.61 | `dit_full_tp2_compile.log` |
| TP4 compile | vt=20 | **FAIL** | 2.997e-03 | 2.515e-03 | 1.191x | 0.999995 | 0.494 | 13.0 (7.4) | 0 | 265.0 | 11.10 | `dit_full_tp4_compile.log` |
| TP2 + SP compile | vt=20 | **FAIL** | 2.992e-03 | 2.515e-03 | 1.190x | 0.999995 | 0.494 | 12.9 (6.7) | 0 | 207.3 | 15.60 | `dit_full_tp2_sp.log` |
| TP2xCP2 gather_kv | vt=20 | **FAIL** | 2.995e-03 | 2.515e-03 | 1.191x | 0.999995 | 0.497 | 16.6 (9.1) | 0 | 649.7 | 15.56 | `dit_full_tp2cp2_gather.log` |
| TP2xCP2 ulysses | vt=20 | **FAIL** | 2.989e-03 | 2.515e-03 | 1.188x | 0.999995 | 0.494 | 14.4 (7.7) | 0 | 1148.3 | 15.57 | `dit_full_tp2cp2_ulysses.log` |
| TP2xCP2 ring | vt=20 | **FAIL** | 2.992e-03 | 2.515e-03 | 1.190x | 0.999995 | 0.493 | 11.5 (4.8) | 0 | 672.7 | 15.57 | `dit_full_tp2cp2_ring.log` |
| TP2 compile | vt=3 | **PASS** | 2.572e-03 | 2.452e-03 | 1.049x | 0.999996 | 0.423 | 14.2 (8.4) | 0 | 213.2 | 15.61 | `dit_full_tp2_v3.log` |

The failure reason is quoted verbatim from `dit_full_tp2_compile.log`: `rel_l1 2.994e-03 is
1.190x the control's 2.515e-03, above max_ratio_vs_control 1.15`. The note with it reads
`err_cos 0.495 < 0.5: the result's error is decorrelated from the bf16 control's, so it does not
round like upstream bf16`.

### Full depth: more timesteps, TP2 (is t=900/vt=20 representative?)

| timestep | compile ratio | eager ratio | control rel-L1 | logs |
|---|---|---|---|---|
| 950 | 0.838x PASS | 0.989x PASS | 3.083e-03 | `dit_full4t_tp2_{compile,eager}.log` |
| **900** | **1.190x FAIL** | 1.006x PASS | 2.515e-03 | `dit_full3t_tp2_{compile,eager}.log` |
| 850 | 1.054x PASS | 1.125x PASS | 2.852e-03 | `dit_full4t_tp2_{compile,eager}.log` |
| 700 | 0.940x PASS | 0.999x PASS | 3.307e-03 | `dit_full4t_tp2_{compile,eager}.log` |
| 500 | 0.891x PASS | 1.025x PASS | 3.105e-03 | `dit_full3t_tp2_{compile,eager}.log` |
| 300 | 0.943x PASS | 0.991x PASS | 3.013e-03 | `dit_full4t_tp2_{compile,eager}.log` |
| 100 | 0.998x PASS | 1.054x PASS | 3.789e-03 | `dit_full3t_tp2_{compile,eager}.log` |
| 900, 3 valid tokens | 1.049x PASS | NOT MEASURED | 2.452e-03 | `dit_full_tp2_v3.log` |

Compiled: mean 0.988x over the 8 samples, range 0.838-1.190x. Eager: mean 1.027x over 7
samples, range 0.989-1.125x. The t=900 control has the smallest error of the seven 20-token
samples.

### Default shape 61x320x512 (10,240 image tokens), full depth, timing only (`--oracle none`)

| layout | gate (no oracle: finite, ranks, fallbacks, blocks) | load+warmup (warm-up fwd) s | compiles | ms/fwd | peak HBM GiB | log |
|---|---|---|---|---|---|---|
| TP2 compile | **PASS** | 180.5 (174.0) | 16 | 1460.8 | 16.18 | `dit_f61_tp2_timing.log` |
| TP4 compile | **PASS** | 100.1 (93.9) | 6 | 805.6 | 11.67 | `dit_f61_tp4_timing.log` |

### Compiled mask bounds (one NeuronCore)

**PASS, 72/72 counts exact on neuron.** The counts are the key bounds, the joint count (ulysses
`key_valid_len`) and the text count (ring `text_valid_len`). They were checked at joint lengths
1,536, 10,496 and 20,096, including 19,858 valid keys at 20,096, the Trainium failure case.
Wall time 36 s. Log `J/mask_bounds.log`, results `J/mask-bounds/results.json`.

## Diagnosis of the full-depth t=900 failure

We followed the systematic-debugging protocol and gathered evidence before trying any fix. The
scripts and their outputs are in `J/`.

1. **The port and the checkpoint match at full depth.** TP2 eager on the device gives 1.006x. Our
   port in bf16 on the host CPU at tp1 (`cpu_port.py`, log `cpu_port_full_bf16.log`) also gives
   1.006x, with err_cos 0.763. Loading uses every key (0 missing, 0 unexpected, 0 skipped).
2. **The layout does not matter.** TP2, TP4, SP and the three CP modes all land at
   2.989-2.997e-03, so the cause is shared by every compiled layout. The block graph is the one
   thing they all share.
3. **Compiling the dual-stream blocks whole reproduces the failure; nothing smaller does.** These
   are eager-mode runs with selected pieces compiled in place (`debug_dit.py` `sub:`, logs
   `dbg_sub_*.log`, `dbg_blk_*.log`):

   | compiled pieces | ratio |
   |---|---|
   | whole dual blocks only (singles eager) | **1.186x** |
   | whole single blocks only (duals eager) | 1.000x |
   | every `ff`, `ff_context`, `proj_mlp`, `act_mlp`, `proj_out_mlp` | 1.009x |
   | every `attn` | 1.007x |
   | every `norm1`, `norm1_context`, `norm` | 1.001x |
   | every `norm2`, `norm2_context` | 1.006x |

4. **The error sits in the massive-activation channels.** We compared per-block outputs on the
   device with diffusers fp32 and bf16 on the host (`cpu_blocks.py`, `analyze_blocks.py`,
   `analyze_blocks2.py`; output `analyze_blocks_full_v20.txt`, `analyze_blocks2.txt`). In the
   image stream, 12 channels exceed 20x the median magnitude (386, 453, 915, 1028, 1111, 1181,
   1213, 1461, 1541, 1907, 1997, 2573), and channel 1028 reaches -4.8e7. In those channels the
   compiled path's accumulated error is 1.52-2.93x the control's from single block 4 onward. In
   the other 3,060 channels it stays at 0.82-0.99x over all 60 blocks. Eager stays at 0.97-1.16x
   in the massive channels and 1.01-1.08x in the rest.
   *Inference:* these channels dominate the LayerNorm statistics of every later norm and of
   `norm_out`, so a relative error there shifts the whole output.
5. **Locally, each compiled block is as accurate as bf16 diffusers or more so.** We fed each block
   the device's own input and compared its output with fp32 and bf16 diffusers on that input
   (`local_blocks.py`, `local_dual.py`, `local_dualx.py`; outputs `local_blocks*.txt`,
   `local_dual.txt`, `local_dualx.txt`):
   * Single blocks are at 0.98-1.01x the control.
   * Dual-block outputs are at 0.50-0.94x in the massive channels.
   * The dual block's intermediates (norm1 output, attention, first residual, norm2, modulation,
     ff) are within 0.55-1.31x when compiled, against 0.84-1.28x in eager mode.
   * In dual blocks 10-19 the compiled block's signed local error in channel 1028 is negative in
     7 of 8 sampled blocks. That makes it magnitude-increasing, because the channel is negative.
     The control's errors in the same blocks have mixed signs.
   * *Inference:* the compiled graph rounds the residual stream once per gated update where
     diffusers rounds twice. Its errors in the massive channels are smaller but stay correlated
     from block to block, so they add up, and the final metric for one input is luck-dependent.
     For this input it came out at 1.19x; err_cos is 0.495.
6. **Numerics changes tried. None reached 1.15x at t=900/vt=20, and none was committed:**

   | experiment (compiled TP2, full depth) | ratio | log |
   |---|---|---|
   | baseline | 1.190x | `dit_full_tp2_compile.log` |
   | GELU in fp32 (A9's accuracy contingency) | 1.190x | `dbg_full_tp2_gelu.log` |
   | single-block tail in fp32 (sum, all-reduce, bias, gate, residual) | 1.168x | `dbg_full_tp2_tail_a.log` |
   | single-block bias/gate/residual in fp32 | 1.178x | `dbg_full_tp2_tail_b.log` |
   | single-block partial sum + all-reduce in fp32 | 1.181x | `dbg_full_tp2_tail_c.log` |
   | every TP all-reduce in fp32 | 1.196x (and 408 ms/fwd) | `dbg_full_tp2_red32.log` |
   | dual-block residual updates explicit fp32, rounded once | 1.188x | `dbg_glue_dualfused.log` |
   | `NEURON_CC_FLAGS=--internal-disable-fma-on-ios` (the eager kernels' flag) | 1.190x | `dbg_full_tp2_nofma.log` |
   | LayerNorm as explicit two-pass fp32 | 1.190x | `dbg_glue2_ln2pass.log` |
   | (host CPU) port bf16 with fused residual rounding | 0.978x | `cpu_port_full_bf16_fused.log` |

   Two device probes ruled out further suspects:
   * `round_probe.py` (`round_probe.txt`): bf16 rounding at exact ties is round-to-nearest-even on
     the host CPU, in eager mode and in compiled graphs.
   * `ln_graph_probe.py`: the fp32 LayerNorm workaround reaches the Dynamo graph (`float` ->
     `layer_norm` -> `to(bfloat16)`).
7. **Conclusion.** This is not a Difflet bug, so there is no fix commit. Under the plan's gate as
   written, every compiled layout FAILs on this one oracle, at 1.188-1.191x. Across 8 full-depth
   samples the compiled DiT averages 0.988x the bf16 control, with 7 of 8 under the gate. The
   suspected cause is a decorrelated (err_cos ~0.5) but locally no-worse rounding pattern of the
   whole-graph compiled dual block. It concentrates in HunyuanVideo's massive-activation channels,
   where one input's final metric scatters by about ±15%; eager scatters too, reaching 1.125x at
   t=850. **Follow-ups:**
   * **Gate over several samples.** Gate the full-depth DiT on the mean or median over several
     timesteps and inputs instead of a single sample (the HV1.5 and Wan checks gate per timestep).
     Cost: one longer oracle, about 2 min of host fp32 per extra timestep, plus seconds on the
     device.
   * **Find the compiler cause.** If an exact pin of the compiled rounding is wanted, the next
     step is an HLO-level diff of the dual-block graph to find which convert the compiler fuses
     differently. Cost: tooling time; no product change is expected.
   * **The e2e check (stage 3) is the integration-level judge,** with 3 steps and the gate on the
     latents.

## Bug ledger

No Difflet bugs were found and no fix commits were made. The one environment failure, the
missing PATH entry for `neuronx-cc`, is recorded under Environment.

## How to inspect manually

* **Pass/fail per case:** the last `PASS|FAIL <name>` lines and the `[rank N] PASS|FAIL ...` summary
  in `J/<name>.log`, which then ends in `EXIT 0` or `EXIT 1`. The exact torchrun command is the
  `launch:` line.
* **Metrics:** `J/<name>.json` holds rank 0's results, and `J/<name>.rankN.json` the others':
  `parity.<t>.vs_fp32.rel_l1`, `parity.<t>.control.rel_l1`, `ratio_vs_control`,
  `device_evidence.counters` (compilations and cache hits), `peak_device_mem_gb`,
  `benchmark.median_ms` and `rank_output_hashes`, which must be equal on every rank.
* **The full-depth failure, per block:** `J/analyze_blocks2.txt` lists accumulated error by
  channel group, with columns `c_big` and `c_rest`. `J/drift.txt` gives the signed drift per
  massive channel.
* **Re-run one cell:** `bash J/run.sh <name> --oracle J/hv1_dit_full_v20.pt --tp-degree 2
  --exec-mode compile --out-json J/<name>.json` from the worktree. Check first that no
  `torch.distributed.run` process is alive.

## Not covered by this stage

* **NOT MEASURED** in this stage (later Task 8 stages): end-to-end parity, the CLI runs, the
  native vs AoT benchmark doc, the feature-catalog row and the full unit suite.
* A cold compile time, because the NEFF cache was warm.
* Full-depth parity at 61 frames (only timing ran at that shape). TP4+SP was not run.
