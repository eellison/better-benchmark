# Expanding The Benchmark Corpus

The source of truth is the merged repro root:

- `repros/canonical/<pattern>/repro.py`
- `repros/canonical/<pattern>/meta.json`
- `repros/canonical/<pattern>/shapes.txt`
- `repros/models/.../manifest.json`
- optional saved `full_graph_*.py` files under `repros/models/...`

Raw capture directories are temporary state. Do not commit `captures/` trees or
run summaries as corpus data.

## Preferred Capture Loop

Use `temporary_capture_for_merge()` for new model-backed capture sources. It
creates a temporary raw capture directory, gives that path to the capture hook,
and removes the raw state when the block exits. Call `capture.merge()` only
after the model run succeeds.

```python
from pathlib import Path

import torch

from capture_hook import install_capture_hook, uninstall_capture_hook
from merge_captures import temporary_capture_for_merge

REPRO_ROOT = Path("/tmp/scratch_space/better_benchmark/repros")
label = "my_model_infer"
model_dir = REPRO_ROOT / "models" / "other" / "infer" / label

with temporary_capture_for_merge(
    REPRO_ROOT,
    label,
    suite="other",
    mode="infer",
) as capture:
    try:
        install_capture_hook(
            output_dir=str(capture.capture_dir),
            label=label,
            graph_dir=str(model_dir),
        )

        model = MyModel().cuda()
        compiled = torch.compile(model)
        compiled(inputs)
    finally:
        uninstall_capture_hook()

    capture.merge()
```

After this, copy or commit the merged repro root contents. The temporary raw
capture directory is gone.

## Existing Capture Sources

Use existing model sources before adding synthetic patterns:

```bash
python scripts/recapture_hf.py --mode both
python scripts/recapture_timm.py --mode both
python scripts/recapture_torchbench.py --mode both
```

For already saved full graphs, use:

```bash
python scripts/recapture_full_graphs.py repros/models --canonical-root repros
```

The recapture scripts write generated output under
`/tmp/scratch_space/better_benchmark/repros` by default. Review the generated
`canonical/` and `models/` changes, validate them, then copy the intended corpus
changes into the repository.

## Merge Existing Captures

If you already have a raw capture directory from `capture_hook.py`, merge it
explicitly:

```bash
python merge_captures.py /tmp/captures/my_model \
  --canonical-dir repros \
  --model-name my_model_infer \
  --suite other \
  --mode infer
```

Prefer the temporary helper for new scripts so the raw capture directory cannot
be confused with corpus output.

## Validation Checklist

Before opening a PR with new repros:

```bash
python scripts/report_repro_corpus.py --canonical-dir repros/canonical --models-dir repros/models
python scripts/validate_eager.py repros/canonical --all-shapes --gpus 0,1 --max-workers 8
python scripts/validate_repros.py --canonical-dir repros/canonical --quick
pytest -q scripts/test_merge_captures.py scripts/test_repro_corpus_invariants.py
```

For a smaller confidence run, validate only the newly added canonical dirs and
then run `report_repro_corpus.py`.
