# Benchmarking Flow

This document describes the current benchmark and validation loop for canonical
repros. Keep it tied to scripts that exist in this repository.

## Benchmark One Repro

Use `scripts/bench.py` for a single `repro.py` or a directory of repros:

```bash
python scripts/bench.py repros/canonical/<pattern>/repro.py --device-kind B200 --hardware B200
python scripts/bench.py repros/canonical --device-kind B200 --hardware B200
```

Useful flags:

| Flag | Purpose |
| --- | --- |
| `--device-kind B200` | Pick a matching GPU with the repo GPU lock. |
| `--gpu 0` | Pin to one visible GPU index instead. |
| `--hardware B200` | Select the metadata bucket used with `--update-meta`. |
| `--update-meta` | Write measured perf back to repro metadata. |
| `--no-gpu-lock` | Disable the advisory lock; avoid this in shared runs. |

## Benchmark A Sweep

Use `scripts/bench_parallel.py` for multi-repro sweeps:

```bash
python scripts/bench_parallel.py repros/canonical \
  --device-kind B200 \
  --gpus 0,1 \
  --workers-per-gpu 2 \
  --output results_b200.json
```

The parallel runner uses persistent workers and the GPU lock so compile/setup
work can overlap while timing remains serialized per GPU. For measurement
sensitivity checks, use `--strict-gpu-lock`.

To benchmark the frozen set:

```bash
python scripts/bench_parallel.py \
  --benchmark-set benchmarks/v1.json \
  --canonical-dir repros/canonical \
  --device-kind B200 \
  --gpus 0,1 \
  --output results_v1_b200.json
```

## Validate Repros

Run eager validation before trusting a new corpus or after bounds changes:

```bash
python scripts/validate_eager.py repros/canonical \
  --all-shapes \
  --gpus 0,1 \
  --max-workers 8 \
  --output eager_validation.json
```

For targeted validation and corpus checks:

```bash
python scripts/validate_repros.py --canonical-dir repros/canonical --quick
python scripts/report_repro_corpus.py --canonical-dir repros/canonical --models-dir repros/models
python scripts/test_adversarial.py
```

## Compare Results

Use `scripts/bench_report.py` to compare sweep outputs and generate a report:

```bash
python scripts/bench_report.py \
  --compare results_before.json results_after.json \
  --output-md report.md
```

Keep hardware labels separate. Do not compare B200 metadata with H100 metadata
unless the report is explicitly about cross-hardware behavior.
