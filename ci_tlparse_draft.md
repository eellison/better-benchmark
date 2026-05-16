# Draft: Add TORCH_TRACE / tlparse to vLLM and inductor benchmark CI

## Motivation
Structured trace logs from TORCH_TRACE provide detailed inductor compilation
info (graphs, guards, scheduling decisions). For ~6.6MB raw per model compile,
they compress to ~300KB with zstd (21x ratio). Full benchmark suite would be
~12MB compressed — negligible storage cost.

## Changes needed

### 1. `_vllm-benchmark.yml` — add TORCH_TRACE collection

Add to "Run vLLM benchmark" step env:
```yaml
      TORCH_TRACE: ${{ runner.temp }}/torch_traces
```

Add new step after "Run vLLM benchmark":
```yaml
      - name: Collect and compress TORCH_TRACE logs
        if: always()
        run: |
          set -eux
          TRACE_DIR="${RUNNER_TEMP}/torch_traces"
          if [ -d "$TRACE_DIR" ] && [ "$(ls -A $TRACE_DIR 2>/dev/null)" ]; then
            echo "Compressing $(du -sh $TRACE_DIR | cut -f1) of trace logs..."
            # Install tlparse for HTML report
            pip install tlparse 2>/dev/null || true
            if command -v tlparse &>/dev/null; then
              mkdir -p vllm-project/vllm/benchmarks/results/tlparse_output
              tlparse -o vllm-project/vllm/benchmarks/results/tlparse_output/ \
                --no-browser --overwrite "$TRACE_DIR" 2>&1 || true
            fi
            # Also save compressed raw traces
            mkdir -p vllm-project/vllm/benchmarks/results/torch_traces
            for f in "$TRACE_DIR"/*.log; do
              gzip -c "$f" > "vllm-project/vllm/benchmarks/results/torch_traces/$(basename $f).gz"
            done
            echo "Compressed traces: $(du -sh vllm-project/vllm/benchmarks/results/torch_traces/ | cut -f1)"
          else
            echo "No TORCH_TRACE output found"
          fi
```

The `upload-benchmark-results` action at the end already uploads everything in
`vllm-project/vllm/benchmarks/results`, so the traces will be included
automatically.

### 2. Inductor perf benchmark — enable by default

The inductor perf CI already has `enable-torch-trace` support but it's disabled
by default. In the nightly workflow that calls `_linux-test.yml`, add:
```yaml
      enable-torch-trace: "1"
```

The existing `collect_tlparse_output` function in `.ci/pytorch/test.sh` handles
the rest. The HTML output goes into `test-reports/tlparse_output/` which is
already in the artifact upload list.

### 3. Size estimate

| Scenario | Raw | gzip | zstd |
|----------|-----|------|------|
| Single BERT fwd+bwd | 6.6 MB | 595 KB | 309 KB |
| ~40 HF models (inductor perf) | ~250 MB | ~25 MB | ~12 MB |
| ~15 vLLM models | ~100 MB | ~10 MB | ~5 MB |

S3 cost: negligible (~$0.02/month for all runs).
