# HunyuanVideo 1.0 on the neuron (torch-native) backend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `difflet generate --model hunyuan_video` runs end to end on the neuron backend (TorchNeuron, one process per NeuronCore, torchrun) with TP, SP and CP (gather_kv, ulysses, ring), TeaCache and a device VAE, matching diffusers within the repo's parity gate.

**Architecture:** Follow the registry-application pattern Wan / Qwen-Image / LTX-2 use (not HV1.5's registry-bypassing pipeline): a `TorchNeuronApplicationBase` DiT that wraps Difflet's **backend-neutral** `HunyuanVideoTransformer3DModel` (`difflet/models/hunyuan_video/modeling_hunyuan_video.py`, which already builds TP/SP/CP from `difflet.ops`, and `difflet.ops` already dispatches to `difflet/backends/neuron/ops_impl`), composed by a `TorchNeuronModelApplication` with a rank-0 host text encoder and the existing `HunyuanVideoOrchestrator`. From HV1.5's neuron port we reuse the VAE application shape (diffusers' autoencoder, eager, one core, tiled), the stage layout (DiT launch, then a one-core vae launch), the TeaCache probe wiring and the manual-check/oracle methodology.

**Tech Stack:** torch 2.14 + torch-neuronx native (`/home/ubuntu/venvs/native-312`), diffusers 0.38, gloo CPU ranks for unit tests, trn2.3xlarge (4 logical cores × 24 GB HBM) for device checks.

**Spec:** none written — the user asked for "HunyuanVideo 1.0 torch-native (neuron) backend, studying HV1.5's implementation". The design decisions below are the spec.

## Design decisions (the spec)

1. **DiT module = Difflet's port, not diffusers' class.** HV1.5 wraps diffusers only because no Difflet port exists. HV1.0's port is the class the AoT path runs, is parity-tested against diffusers on CPU (`tests/numerical/test_hunyuan_video_vs_diffusers.py`), and its single blocks already split the fused `proj_out` (loaded via `CheckpointSlice`, exactly as Flux does in `difflet/backends/neuron/flux/transformer.py:78-85`). Its attention already routes the text padding through `attention(..., bound_min, bound_max)`, which the neuron op serves with attention_cte's key-bounded flash kernel.
2. **No CFG.** HV1.0 is guidance-distilled (`guidance = scale × 1000` into the timestep embedding). `cfg_parallel_enabled` is refused, as `entry.py` already does for Trainium.
3. **Text encoding on rank 0's host, inside the generate launch** (LTX-2's `HostTextEncoder` pattern), using diffusers' `HunyuanVideoPipeline.encode_prompt` (Llama template, crop 95, hidden layer −3, 256 tokens; CLIP pooler at 77 tokens): the same tensors the AoT `llama`/`clip` stages produce, without NxD. No device text encoder (YAGNI).
4. **Decode in a separate one-core vae launch** (HV1.5/LTX-2/Wan layout) with diffusers' `AutoencoderKLHunyuanVideo`, which tiles in space **and time** — so HV1.5's long-clip VAE OOM does not recur.
5. **Topology floor TP2.** The 24 GB checkpoint does not fit one 24 GB core; `hbm.dit_fits` refuses tp1 before torchrun starts. Supported layouts on one trn2.3xlarge: TP2, TP4, TP2+SP, TP4+SP, TP2×CP2 (gather_kv / ulysses / ring). 24 heads ⇒ `tp ∈ {2, 3, 4, 6, …}`.
6. **Ring CP needs one cross-backend change:** the port's ring branch raises on any mask, and the port always builds one. Pass the valid-text count as `text_valid_len` (the neuron op already takes it, added for HV1.5); add the keyword to the `difflet.ops` facade, implement it in the CPU reference, refuse a non-None value on Trainium (whose behaviour is unchanged: it raised before too).
7. **Out of scope:** the planner (`difflet/planner/*`, Trainium-only model of feasibility), serving (`difflet serve` metadata), multi-shape bucketing, batch > 1, device text encoder, I2V (`image_condition_type`).

## Global Constraints

- Device dtype bf16 only (`require_bf16_on_device`); CPU tests run fp32 on gloo with `DIFFLET_BACKEND=neuron`.
- One static shape per process; default `height=320, width=512, num_frames=61`; `num_frames % 4 == 1`; `height % 16 == 0` and `width % 16 == 0` (VAE ×8, patch 2).
- Text: `text_seq_len = 256`, `encoder_attention_mask` int64 right-padded; pooled projection `[B, 768]`; guidance tensor `[B]` = `guidance_scale × 1000.0`.
- Image tokens `N = ((F−1)/4+1) × (H/16) × (W/16)` (10,240 at the default shape) must divide by `cp` (CP) or `tp` (SP).
- Parity gate on device (repo convention, `tests/manual/check_neuron_hv15_dit.py`): rel-L1 vs fp32 diffusers ≤ 1.15 × the bf16-diffusers control's rel-L1.
- Commit per task; messages in the repo's style (`feat(neuron): HunyuanVideo …`), ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Run unit tests with `/home/ubuntu/venvs/native-312/bin/python -m pytest`.

## Review Focus

1. **Very short prompts** (3 valid text tokens of 256): the joint key bound is `N + 3`; the token refiner's symmetric mask has almost all rows padded (column 0 forced True). Expected: finite output matching diffusers. Pinned in Task 2 (`test_dit_matches_diffusers_short_prompt`) and on device in Task 8.
2. **Full-length prompts** (256 valid, no padding): bound equals the full key length. Pinned in Task 2 (`valid_text=256` case).
3. **Shapes CP/SP cannot split** (e.g. 33 frames at 480×848 = 9×30×53 = 14,310 tokens at cp=4): a `ValueError` naming the token count and a divisor before anything loads. Pinned in Task 1.
4. **A request at a shape other than the compiled one**: a clear `ValueError` from the DiT adapter, not a silent recompile. Pinned in Task 5 (`test_dit_adapter_rejects_other_shape`).
5. **Guidance scaling**: the DiT must receive `guidance_scale × 1000`; a missed factor gives plausible-looking but wrong video. Pinned in Task 5 (`test_generate_sends_guidance_times_1000`).

---

### Task 1: `NeuronHunyuanVideoConfig` and the HBM estimate

**Files:**
- Create: `difflet/backends/neuron/hunyuan_video/__init__.py` (docstring only, like `hunyuan_video15/__init__.py`)
- Create: `difflet/backends/neuron/hunyuan_video/config.py`
- Create: `difflet/backends/neuron/hunyuan_video/hbm.py`
- Test: `tests/unit/backends/test_neuron_hv1_config.py`

**Interfaces:**
- Produces: `NeuronHunyuanVideoConfig` (dataclass, modelled on `difflet/backends/neuron/ltx_2/config.py:NeuronLtx2Config`): shape fields `height=320, width=512, num_frames=61, batch_size=1, text_seq_len=256`; every key of the transformer `config.json` (`in_channels … rope_axes_dim`, defaults = the HunyuanVideo snapshot's values); parallel fields `tp_degree=1, cp_degree=1, context_parallel_enabled=False, cp_mode="gather_kv", sp_enabled=False, cfg_parallel_enabled=False`. Properties `latent_frames, latent_height, latent_width` (÷4+1, ÷8, ÷8), `num_image_tokens`, `inner_dim`. `validate() -> None`. `from_pretrained(transformer_path, **overrides) -> NeuronHunyuanVideoConfig`. The config is also what `HunyuanVideoTransformer3DModel(config)` reads (it takes any object with those attributes, `modeling_hunyuan_video.py:1022-1043`).
- Produces: `hbm.dit_fits(model_dir: str, *, tp_degree: int) -> tuple[bool, int, int]` (same contract as `difflet/backends/neuron/ltx_2/hbm.py`, built on `difflet/backends/neuron/hbm.py` `Component`/`check_fits`; reads safetensors headers only).

- [ ] **Step 1: Write the failing tests**

```python
def test_defaults_match_the_snapshot_shape():
    c = NeuronHunyuanVideoConfig()
    assert (c.latent_frames, c.latent_height, c.latent_width) == (16, 40, 64)
    assert c.num_image_tokens == 16 * 20 * 32 == 10240

def test_validate_refuses_what_hunyuan_video_does_not_support():
    with pytest.raises(NotImplementedError, match="guidance-distilled"):
        NeuronHunyuanVideoConfig(cfg_parallel_enabled=True).validate()
    with pytest.raises(ValueError, match="4k\\+1"):
        NeuronHunyuanVideoConfig(num_frames=60).validate()
    with pytest.raises(ValueError, match="multiple of 16"):
        NeuronHunyuanVideoConfig(height=328).validate()
    with pytest.raises(ValueError, match="tp=5"):
        NeuronHunyuanVideoConfig(tp_degree=5).validate()
    with pytest.raises(ValueError, match="mutually exclusive"):
        NeuronHunyuanVideoConfig(tp_degree=2, sp_enabled=True, cp_degree=2,
                                 context_parallel_enabled=True).validate()

def test_validate_names_an_unsplittable_token_count():
    c = NeuronHunyuanVideoConfig(height=480, width=848, num_frames=33, cp_degree=4,
                                 context_parallel_enabled=True)
    with pytest.raises(ValueError, match="14310"):
        c.validate()

def test_ulysses_needs_heads_per_rank_divisible_by_cp():
    NeuronHunyuanVideoConfig(tp_degree=2, cp_degree=2, context_parallel_enabled=True,
                             cp_mode="ulysses").validate()          # 12 heads / 2: ok
    with pytest.raises(ValueError, match="ulysses"):
        NeuronHunyuanVideoConfig(tp_degree=8, cp_degree=4, context_parallel_enabled=True,
                                 cp_mode="ulysses").validate()      # 3 heads / 4

def test_dit_fits_refuses_tp1_and_accepts_tp2(tmp_path, monkeypatch):
    # a fake snapshot whose safetensors header declares 12.8e9 bf16 params
    ...
    assert hbm.dit_fits(str(snapshot), tp_degree=1)[0] is False
    assert hbm.dit_fits(str(snapshot), tp_degree=2)[0] is True
```

(Build the fake snapshot the way `tests/unit/backends/test_neuron_hbm.py` does.)

- [ ] **Step 2: Run to verify they fail** — `pytest tests/unit/backends/test_neuron_hv1_config.py -v` → ImportError.
- [ ] **Step 3: Implement** `config.py` with the checks from `difflet/backends/neuron/core/config_checks.py` (`check_heads_divide_tp`, `check_cp_mode`, `check_cp_split`, `check_ulysses_heads`, `exclusive_modes`, `check_even_split` for SP); and `hbm.py` (DiT component = whole transformer checkpoint ÷ tp, plus the activation margin `ltx_2/hbm.py` uses).
- [ ] **Step 4: Run to verify they pass.**
- [ ] **Step 5: Commit** — `feat(neuron): HunyuanVideo config and HBM estimate (tp1 refused: 24 GB DiT, 24 GB core)`.

---

### Task 2: The DiT application (TP, eager and compile)

**Files:**
- Create: `difflet/backends/neuron/hunyuan_video/transformer.py`
- Create: `tests/unit/backends/_neuron_hv1_toy.py` (tiny snapshot writer + diffusers reference)
- Modify: `tests/unit/backends/_neuron_workers.py` (add `hv1_dit_worker`)
- Test: `tests/unit/backends/test_neuron_hv1_dit.py`

**Interfaces:**
- Consumes: `NeuronHunyuanVideoConfig` (Task 1).
- Produces: `checkpoint_key(name: str) -> str` (module → checkpoint: `single_transformer_blocks.{i}.proj_out_attn.bias` → `single_transformer_blocks.{i}.proj_out.bias`, identity otherwise).
- Produces: `NeuronHunyuanVideoTransformerApplication(TorchNeuronApplicationBase)`:
  - `__init__(*, model_path, config: NeuronHunyuanVideoConfig | None = None, parallel=None, dtype=torch.bfloat16, exec_mode=None, device="neuron", **overrides)` — builds the config with `from_pretrained` when None, then `check_tp_matches` / `resolve_cp_degree` / `check_flag_matches(..., "sp_enabled")` / `validate()` / `require_bf16_on_device`, as `difflet/backends/neuron/wan/transformer.py:80-92` does.
  - `block_attrs = ("transformer_blocks", "single_transformer_blocks")`.
  - `build_module()` → a module-private subclass `_HunyuanVideoTensorSignature(HunyuanVideoTransformer3DModel)` whose `forward(hidden_states, timestep, encoder_hidden_states, encoder_attention_mask, pooled_projections, guidance) -> Tensor` returns `super().forward(..., return_dict=False)[0]` (HV1.5's `_tensor_signature_model` idea: the base warms up and compiles with positional tensors).
  - `checkpoint_slices() -> dict[str, CheckpointSlice]`: per single block, `proj_out_attn.weight = CheckpointSlice("single_transformer_blocks.{i}.proj_out.weight", 1, 0, inner_dim)` and `proj_out_mlp.weight = CheckpointSlice(same, 1, inner_dim, mlp_ratio × inner_dim)`.
  - `get_example_inputs()` → the six tensors at the static shape: latents `[B,16,F',H/8,W/8]`, timestep `[B]`, text `[B,256,4096]`, mask int64 `[B,256]` with the first 64 ones, pooled `[B,768]`, guidance `[B]` (6000.0).
  - `teacache_mod_input(hidden_states, timestep, pooled_projections, guidance) -> Tensor` — `module.teacache_mod_input(x, t, None, None, pooled, guidance)` on this rank's device (the port's method drops the text arguments; the modulation linears are diffusers' replicated `nn.Linear`, so the result is whole on every rank).

- [ ] **Step 1: Write the tiny toy.** `_neuron_hv1_toy.write_tiny_hv1(root) -> str` saves a diffusers `HunyuanVideoTransformer3DModel(in_channels=4, out_channels=4, num_attention_heads=4, attention_head_dim=16, num_layers=1, num_single_layers=2, num_refiner_layers=1, patch_size=2, patch_size_t=1, text_embed_dim=32, pooled_projection_dim=16, rope_axes_dim=(4, 6, 6), guidance_embeds=True)` with a fixed seed; `tiny_inputs(valid_text: int)` builds a `[1,4,5,8,8]` latent (5×4×4 = 80 image tokens after the 2×2 patch: divisible by cp 2 and 4), 16-token text with `valid_text` valid, pooled, timestep 500, guidance 6000; `diffusers_reference(path, inputs)` runs diffusers' forward in fp32.
- [ ] **Step 2: Write the failing tests**

```python
def test_checkpoint_mapping_and_slices(): ...            # as test_neuron_flux_dit.py:test_checkpoint_mapping
@pytest.mark.parametrize("valid_text", [3, 9, 16])
def test_dit_matches_diffusers_tp1(tiny, valid_text):    # eager, cpu, fp32
    assert torch.allclose(ours, ref, atol=1e-5, rtol=1e-4)
def test_dit_matches_diffusers_short_prompt(tiny): ...   # valid_text=1: finite and allclose
@pytest.mark.parametrize("exec_mode", ["eager", "compile"])
def test_dit_matches_diffusers_tp2_gloo(tiny, exec_mode):
    outs = run_ranks(hv1_dit_worker, 2, tiny, {"tp_degree": 2}, exec_mode)
    assert all(torch.allclose(o, ref, atol=1e-5, rtol=1e-4) for o in outs)
def test_teacache_mod_input_matches_the_port(tiny): ...  # equals HunyuanVideoTransformer3DModel.teacache_mod_input on CPU
```

- [ ] **Step 3: Run to verify they fail.**
- [ ] **Step 4: Implement** `transformer.py` (module docstring in the house style: what it wraps, the lifecycle, why the port and not diffusers, numerics workarounds relied on — fp32 LayerNorm and the bool bitwise guard installed by `prepare_runtime`).
- [ ] **Step 5: Run to verify they pass** (compile mode on CPU uses the `aot_eager` backend).
- [ ] **Step 6: Commit** — `feat(neuron): HunyuanVideo DiT application (Difflet's port, TP, per-block compile, sliced proj_out)`.

---

### Task 3: Context parallelism (gather_kv, ulysses) and sequence parallelism

**Files:**
- Modify: `tests/unit/backends/_neuron_workers.py` (reuse `hv1_dit_worker` with the parallel dict)
- Test: `tests/unit/backends/test_neuron_hv1_dit.py` (new cases)
- Modify only if a test fails: `difflet/backends/neuron/hunyuan_video/transformer.py`, `difflet/models/hunyuan_video/modeling_hunyuan_video.py`

**Interfaces:**
- Consumes: Task 2's application with `parallel=DiffletParallelConfig(tp_degree=…, cp_degree=…, cp_mode=…, sp_enabled=…)`.
- Produces: nothing new — this task proves the port's existing CP/SP code paths on the neuron collectives (`scatter_to_process_group_spmd`, `SPMDRank` as a Python-int rank, `seq_to_heads`, `reduce_scatter_to_sequence_parallel_region`, `_sp_unbias` with the neuron `RowParallelLinear`'s full-bias partials).

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("parallel,world", [
    ({"tp_degree": 1, "cp_degree": 2, "cp_mode": "gather_kv"}, 2),
    ({"tp_degree": 1, "cp_degree": 2, "cp_mode": "ulysses"}, 2),
    ({"tp_degree": 2, "cp_degree": 2, "cp_mode": "gather_kv"}, 4),
    ({"tp_degree": 2, "cp_degree": 2, "cp_mode": "ulysses"}, 4),
    ({"tp_degree": 2, "sp_enabled": True}, 2),
])
@pytest.mark.parametrize("valid_text", [3, 16])
def test_dit_parallel_matches_diffusers(tiny, parallel, world, valid_text):
    outs = run_ranks(hv1_dit_worker, world, tiny, parallel, "eager", valid_text)
    assert all(torch.allclose(o, ref, atol=1e-5, rtol=1e-4) for o in outs)
```

Plus one compile-mode case for `tp2 cp2 ulysses` (the collectives must stay functional inside the block graphs).

- [ ] **Step 2: Run them.** Any that pass need no code. For each failure, use superpowers:systematic-debugging; fix in the smallest owning file; one commit per bug (`fix(neuron): …` or `fix(hunyuan_video): …`, the message naming the failing layout).
- [ ] **Step 3: Run the whole file to verify it passes.**
- [ ] **Step 4: Commit** — `test(neuron): HunyuanVideo DiT under CP (gather_kv, ulysses) and SP against diffusers`.

---

### Task 4: Ring context parallelism with the text mask

**Files:**
- Modify: `difflet/ops/attention.py:62` (facade gains `text_valid_len=None`)
- Modify: `difflet/backends/cpu/ops_impl/attention.py:73` (reference: mask text keys `>= text_valid_len[b]` in the joint softmax)
- Modify: `difflet/backends/trainium/ops_impl/attention.py:260` (accept the keyword; `NotImplementedError` if not None)
- Modify: `difflet/backends/tpu/ops_impl/attention.py:298` (signature unchanged: it already raises for everything)
- Modify: `difflet/models/hunyuan_video/modeling_hunyuan_video.py:468-487` (ring branch: replace the `attention_mask is not None` refusal with `text_valid_len = attention_mask.reshape(B, -1)[:, -S_txt:].sum(-1, dtype=torch.float32).to(torch.int32)` — the joint mask's text columns (it is built over `[latent_full ‖ text]`, so the last `S_txt = context_k.shape[1]` columns are the text), with the float-sum idiom of the ulysses branch (Trainium LNC2 integer-reduction bug) — and pass it)
- Test: `tests/unit/ops/test_joint_ring_text_valid_len.py`, `tests/unit/backends/test_neuron_hv1_dit.py` (ring cases)

**Interfaces:**
- Produces: `difflet.ops.joint_ring_attention(q, image_k, image_v, text_k, text_v, *, scale, causal=False, text_valid_len=None)`; `text_valid_len` is int32 `[B]` counting valid text keys (text right-padded).

- [ ] **Step 1: Write the failing tests** — CPU reference with `text_valid_len=[5]` equals dense SDPA over `[image ‖ text[:5]]`; Trainium impl raises `NotImplementedError` for a non-None value (import-guarded like the other trainium op tests); and in `test_neuron_hv1_dit.py` the ring layouts `{"cp_degree": 2, "cp_mode": "ring"}` (world 2) and `{"tp_degree": 2, "cp_degree": 2, "cp_mode": "ring"}` (world 4) at `valid_text ∈ {3, 16}` match diffusers.
- [ ] **Step 2: Run to verify they fail** (the modeling raises "ring cp_mode does not support attention_mask").
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run these tests plus `tests/unit/models/hunyuan_video/` to verify they pass and nothing on the AoT side regressed.**
- [ ] **Step 5: Commit** — `feat(hunyuan_video): ring context parallelism masks the padded text by count (neuron); difflet.ops.joint_ring_attention gains text_valid_len`.

---

### Task 5: The composed application, the host text encoder, TeaCache and the registry

**Files:**
- Create: `difflet/models/hunyuan_video/neuron_application.py`
- Modify: `difflet/models/hunyuan_video/entry.py:19-22` (a `backend == "neuron"` branch returning `TorchNeuronHunyuanVideoApplication`, after the existing CFG refusal; as `difflet/models/ltx_2/entry.py:27-38`)
- Modify: `difflet/registry.py:324` (`backends=("trainium", "neuron")`)
- Modify: `tests/unit/models/hunyuan_video/test_hunyuan_entry.py:49,56` (neuron is now accepted; another backend still rejected)
- Modify: `tests/unit/backends/_neuron_hv1_toy.py` (`write_tiny_hv1_pipeline(root)`: tiny transformer + tiny `AutoencoderKLHunyuanVideo` + scheduler + tiny random-weight Llama and CLIP text models with the snapshot's real tokenizers copied from the local HF cache — the Llama template's crop of 95 tokens needs the real tokenizer; skip when the snapshot is absent — as `_neuron_ltx2_toy.py` builds its tiny pipeline)
- Test: `tests/unit/backends/test_neuron_hv1_app.py`

**Interfaces:**
- Consumes: Tasks 1–2; `HunyuanVideoOrchestrator`, `HunyuanVideoDiTInputBundle`, `_bundle_from_tensors` (`difflet/models/hunyuan_video/pipeline.py`); `TorchNeuronModelApplication`, `HostTextEncoder`, `OnDeviceDiT`, `broadcast_from_rank0` (`core/model_application.py`); `NeuronTeaCacheProbe` (`core/teacache.py`).
- Produces: `TorchNeuronHunyuanVideoApplication(TorchNeuronModelApplication)`:
  - `MODEL_LABEL = "HunyuanVideo"`, `DIT_BUNDLE = HunyuanVideoDiTInputBundle`, `TEACACHE_KWARGS = ()` (TeaCache is implemented, as Wan's).
  - `KWARG_DEFAULTS`: `device="neuron", exec_mode=None, text_seq_len=256, shapes=None, batch_size=1, text_encoder_device="host", encoder_threads=<Wan's default>, enable_vae_decoder=False, teacache_speedup=None, teacache_calibration_path=None, teacache_cadence=None, teacache_online_delta_alpha=None` plus whatever the HV CLI stage passes the Trainium application today (read `_stage_generate` in `difflet/cli/orchestrators/hunyuan_video.py:247-352` and accept exactly those keys).
  - `_reject_model_options`: `text_encoder_device != "host"` → `NotImplementedError`; `shapes` with more than one shape → `NotImplementedError`.
  - `generate(*, prompt, seed=None, num_inference_steps=4, guidance_scale=6.0, output_type="latent", **kwargs)` → host-encode on rank 0 and broadcast, initial latents drawn on rank 0 (`_draw_on_rank0`), then `self.pipeline(latents=…, encoder_hidden_states=…, encoder_attention_mask=…, pooled_projections=…, guidance_scale=…, num_inference_steps=…, output_type=…)`.
- Produces: `_HostHunyuanVideoTextEncoder(HostTextEncoder)`: diffusers `HunyuanVideoPipeline.from_pretrained(model_path, transformer=None, vae=None, torch_dtype=bf16)` on rank 0; `__call__(prompt) -> (embeds [1,256,4096] bf16, mask [1,256] int64, pooled [1,768] bf16)` via `pipe.encode_prompt(prompt=prompt, prompt_2=None, max_sequence_length=256)`; the mask is cast to int64 before the broadcast.
- Produces: `_OnDeviceHunyuanVideoTransformer(OnDeviceDiT)`: `__call__(bundle)` checks every tensor's shape against the config first (`ValueError` naming the input, its shape and the compiled shape), then `_forward(bundle.as_model_inputs())`; attributes `teacache_probe_fused` (True when a probe is set) and `teacache_delta(bundle)` → `NeuronTeaCacheProbe(app.teacache_mod_input).teacache_delta(hidden, timestep, pooled, guidance)` — the orchestrator's existing fused branch (`pipeline.py:220-226`) then needs no change.

- [ ] **Step 1: Write the failing tests**

```python
def test_registry_builds_the_neuron_application(tiny_pipeline): ...   # create_application(backend="neuron", device="cpu") is a TorchNeuronHunyuanVideoApplication
def test_generate_matches_diffusers_pipeline(tiny_pipeline):           # latents, 3 steps, seed 42, vs HunyuanVideoPipeline(output_type="latent"), fp32 cpu
def test_generate_tp2_gloo_matches_tp1(tiny_pipeline): ...
def test_generate_sends_guidance_times_1000(tiny_pipeline):            # record bundles at the adapter: guidance == 6000.0
def test_dit_adapter_rejects_other_shape(tiny_pipeline):               # ValueError mentions 'hidden_states' and both shapes
def test_text_encoder_matches_diffusers_encode_prompt(tiny_pipeline): ...
def test_refuses_cfg_parallel_and_device_text_encoder(tiny_pipeline): ...
def test_teacache_cadence_skips_and_probe_mode_runs(tiny_pipeline): ...   # cadence 2 halves DiT calls; adaptive with a calibration uses teacache_delta
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** (module docstring in the house style of `difflet/models/wan/neuron_application.py`).
- [ ] **Step 4: Run these tests plus `tests/unit/registry tests/unit/models/hunyuan_video tests/unit/backends/test_neuron_model_application.py`.**
- [ ] **Step 5: Commit** — `feat(neuron): HunyuanVideo end-to-end application (host Llama/CLIP encode, orchestrator, TeaCache) and its registry entry`.

---

### Task 6: The VAE decoder application

**Files:**
- Create: `difflet/backends/neuron/hunyuan_video/vae.py`
- Test: `tests/unit/backends/test_neuron_hv1_vae.py`

**Interfaces:**
- Produces: `NeuronHunyuanVideoVaeDecoderApplication(*, model_path, height=320, width=512, num_frames=61, tile_size=256, tile_frames=16, dtype=torch.bfloat16, device="neuron")` with `load(skip_warmup=False)`, `decode(latents) -> [B,3,F,H,W]` in [−1, 1], `latent_shape`, `scaling_factor`, `phase_seconds` — HV1.5's `vae.py` with `AutoencoderKLHunyuanVideo`; tiling sets diffusers' `tile_sample_min_height/width = tile_size`, `tile_sample_stride_height/width = tile_size × 3 // 4`, `tile_sample_min_num_frames = tile_frames`, `tile_sample_stride_num_frames = tile_frames × 3 // 4` (diffusers' defaults are 256/192/16/12); the encoder is dropped after loading.

- [ ] **Step 1: Write the failing tests** — on the tiny VAE from Task 5's toy: our decode equals diffusers' tiled decode with the same tile settings (allclose fp32); `latent_shape` at 320×512×61 is `(1, 16, 16, 40, 64)`; encoder is None after load.
- [ ] **Step 2: Run to verify they fail.** **Step 3: Implement.** **Step 4: Run to verify they pass.**
- [ ] **Step 5: Commit** — `feat(neuron): HunyuanVideo VAE decoder application (diffusers' autoencoder, eager, tiled in space and time)`.

---

### Task 7: The CLI on the neuron backend

**Files:**
- Modify: `difflet/cli/orchestrators/hunyuan_video.py` (`compile`/`generate` :81/:93 branch on `torchrun_backend()`; new `_compile_torchrun`, `_generate_torchrun`, `_stage_generate_neuron`, `_stage_vae_neuron`; `_run_stage_internal` :113 dispatches `generate`/`vae` on neuron; `_stage_cache_inputs` :354 wraps with `torchrun_stage_keys`)
- Test: `tests/unit/cli/test_orchestrator_hunyuan_neuron.py`; modify `tests/unit/cli/test_neuron_stage_cache_keys.py` (add HunyuanVideo)

**Interfaces:**
- Consumes: `run_generate_stage`, `run_vae_stage`, `refuse_dit_overflow`, `reject_unsupported_torchrun_options`, `torchrun_stage_keys`, `root_compile_cache_at_cache_dir`, `torchrun_parallel` (`difflet/cli/orchestrators/torchrun.py`); `hbm.dit_fits` (Task 1); the application (Task 5); the VAE (Task 6).
- Produces: on neuron, `difflet generate --model hunyuan_video` runs two launches — `generate` on `tp × cp` cores (writes `latents.pt`), then `vae` on one core (writes the mp4) — following `difflet/cli/orchestrators/ltx_2.py:450-647` stage for stage. The AoT path (`clip`, `llama`, `generate`) is untouched. `--host-vae` decodes with diffusers on the host instead of the vae launch, if the LTX-2 orchestrator offers it (mirror its flag set; add nothing it lacks).

- [ ] **Step 1: Write the failing tests** (monkeypatch `runner.run_stage` and `torchrun_backend`, as `tests/unit/cli/test_orchestrator_ltx2.py` does): stage chain is `["generate", "vae"]` with core counts `[tp*cp, 1]`; `--sp`, `--cp-mode`, `--exec-mode` and the TeaCache flags are forwarded to the generate stage; tp1 exits 2 through `refuse_dit_overflow` before any stage runs; `--cfg-parallel` exits non-zero with the guidance-distilled message; the AoT path's stage chain is unchanged (`["clip", "llama", "generate"]`); the cache keys of the generate stage include `backend` and `exec_mode`.
- [ ] **Step 2: Run to verify they fail.** **Step 3: Implement.** **Step 4: Run `tests/unit/cli` to verify they pass.**
- [ ] **Step 5: Commit** — `feat(cli): HunyuanVideo on the neuron backend: torchrun generate stage (host encode), one-core vae stage`.

---

### Task 8: On-device validation and the evidence report

**Files:**
- Create: `tests/manual/check_neuron_hv1_dit.py` (flags as `check_neuron_hv15_dit.py`: `--model-dir`, `--make-oracle PATH | --oracle PATH|none`, `--iters`, `--exec-mode`, `--tp-degree`, `--cp-mode`, `--sp`, `--frames`, `--valid-text`)
- Create: `tests/manual/check_neuron_hv1_e2e.py` (`--make-ref` / `--check`, `--tp-degree`; diffusers `HunyuanVideoPipeline`, 3 steps, seed 42, latents compared)
- Create: `benchmark/trn2/hunyuan_video_neuron.md`
- Modify: `.claude/skills/difflet-device-verify/references/feature-catalog.md` (HunyuanVideo's neuron row)

**Interfaces:**
- Consumes: everything above. The oracle is fp32 diffusers on the host CPU at 320×512, 5 frames, 256-token text with `--valid-text` valid (default 20): ~52 GB host RAM.

- [ ] **Step 1: Make the oracles** (host, fp32): `check_neuron_hv1_dit.py --make-oracle $JOB/tmp/hv1_dit_oracle.pt --valid-text 20` and again with `--valid-text 3`; `check_neuron_hv1_e2e.py --make-ref`.
- [ ] **Step 2: DiT parity on device**, each passing the 1.15× gate, in this order (stop and debug at the first failure, one commit per bug): TP2 eager; TP2 compile; TP4 compile; TP2+SP; TP2×CP2 gather_kv; TP2×CP2 ulysses; TP2×CP2 ring; TP2 at `--valid-text 3`.
- [ ] **Step 3: End-to-end parity** — `check_neuron_hv1_e2e.py --check --tp-degree 2` passes the gate.
- [ ] **Step 4: CLI runs** — `difflet generate --model hunyuan_video --backend neuron` at TP2, TP4 and TP2×CP2 ulysses, 320×512×61, 50 steps: a playable mp4 each; record per-step latency, compile and load seconds, and peak HBM in `benchmark/trn2/hunyuan_video_neuron.md` beside the AoT numbers from `benchmark/trn2/` for the same shape; then 720×1280×129 at the fastest layout that fits.
- [ ] **Step 5: Run the full unit suite** — `pytest tests/unit -q -x -n 8`.
- [ ] **Step 6: Commit** — `docs(benchmark): HunyuanVideo on the neuron backend: parity per layout and native vs AoT on trn2`.

Use the `difflet-device-verify` skill for Steps 2–4 (its evidence-doc format, disk and supervision rules, one-bug-one-commit protocol).
