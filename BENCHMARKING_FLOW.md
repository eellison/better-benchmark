# Benchmarking Flow

## Lock-Safe Single-Repro Benchmarking

Every GPU benchmark must go through one of:

```bash
# Primary: bench.py with GPU lock
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python scripts/bench.py --device-kind H100 --hardware H100 path/to/repro.py

# Arbitrary commands: with_gpu_lock.py wrapper
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python scripts/with_gpu_lock.py --device-kind H100 -- <gpu-command>
```

Do NOT run raw Python benchmark commands directly — the GPU lock is advisory
and only works when all processes opt in.

### `bench.py` flags

| Flag | Effect |
|------|--------|
| `--gpu 0` | Lock a specific GPU by index |
| `--device-kind H100` | Lock any free matching GPU, sets `CUDA_VISIBLE_DEVICES` |
| `--hardware H100` | Controls `meta.json` perf bucket when `--update-meta` is used |
| `--no-gpu-lock` | Disables locking (avoid in shared runs) |

### Path behavior

- Accepts a single `repro.py` file or a directory containing `**/repro.py`.
- Normalizes accidental surrounding whitespace if the stripped path exists.
- Raises `FileNotFoundError` for nonexistent paths.
- Raises clear error for directories with no `repro.py` files.

## Batch Benchmarking

```bash
# Queue-driven batch
python scripts/bench_queue.py --queue-hardware H100 --limit 20

# Regenerate queue after batch
python scripts/bench_queue.py --queue-hardware H100 --regenerate-after
```

`bench_queue.py` delegates each item to `bench.py`, so it inherits lock behavior.
`--queue-hardware H100` defaults both `--device-kind` and `--hardware` to H100.

## SOL Investigation

```bash
# Benchmark + investigation of a repro
python scripts/sol_gap.py benchmark path/to/repro.py --device-kind H100
python scripts/sol_gap.py investigate path/to/repro.py --device-kind H100
```

Investigation compiles under `TORCH_LOGS=output_code` and writes generated
Triton code next to the repro.

## Canonical Repro Set

```bash
# Re-build canonical set from all aten_repros
python canonicalize_repros.py

# Query: which patterns are shared by most models
python query_repros.py shared --top 20

# Query: which models share a specific pattern
python query_repros.py models <pattern_hash>

# Benchmark a canonical repro (default shape)
python repros/canonical/<name>/repro.py

# Benchmark all shapes of a pattern
python repros/canonical/<name>/repro.py --all-shapes

# Benchmark a specific shape
python repros/canonical/<name>/repro.py --shape distillgpt2_inference_23f019ef
```

## Extraction (New Repros)

### Option A: Capture hook (preferred — works with any script)

Register the hook, then run your model normally:

```python
from capture_hook import install_capture_hook
install_capture_hook(output_dir="/tmp/captures/my_model", label="my_model")

# Now just run your model — no changes needed:
model = MyModel().cuda()
compiled = torch.compile(model)
compiled(inputs)
```

Or via environment variable (auto-installs on import):

```bash
REPRO_CAPTURE_DIR=/tmp/captures/my_model python my_training_script.py
```

Then merge captures into the canonical set:

```bash
python merge_captures.py /tmp/captures/my_model --canonical-dir repros/ --model-name my_model_inference
```

### Option B: extract_reductions.py (batch extraction from dynamo suite)

```bash
# Extract from a HuggingFace model, updating canonical set
python extract_reductions.py dynamo:ModelName --mode aten --inference-only --canonical-dir repros/

# Extract all dynamo models
python extract_reductions.py dynamo:all --mode aten --inference-only --canonical-dir repros/
```

## Legacy Scripts (Not Lock-Safe)

These do NOT use the GPU lock and should not be used in multi-agent runs:

- `scripts/benchmark_all.py`
- `scripts/investigate_kernel.py`

Wrap them with `with_gpu_lock.py` if you must use them.

## Agent Batch Protocol

1. Acquire GPU lock via `bench.py` or `with_gpu_lock.py`.
2. Run benchmark with `--device-kind H100 --hardware H100`.
3. Use fresh `TORCHINDUCTOR_CACHE_DIR` for codegen experiments.
4. H100 and B200 metadata are separate — always specify `--hardware`.
5. Do not draw conclusions from B200 metadata when local codegen differs.
