#!/bin/bash
# Full pipeline: run benchmark models → extract repros → SOL benchmark → analysis
# Usage: ./run_full_pipeline.sh [suite]
#   suite: huggingface, timm_models, torchbench (default: huggingface)

set -euo pipefail

PYTORCH_DIR="/tmp/pytorch-work"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_BASE="/tmp/benchmark_traces"
SUITE="${1:-huggingface}"

cd "$PYTORCH_DIR"

echo "========================================"
echo " Full Pipeline: $SUITE"
echo " $(date)"
echo "========================================"

# Phase 1: Run models with extraction hook to get standalone repros
echo ""
echo "=== Phase 1: Extract reduction repros from $SUITE ==="
REPRO_DIR="$OUTPUT_BASE/repros/$SUITE"
mkdir -p "$REPRO_DIR"

MODEL_LIST="benchmarks/dynamo/${SUITE}_models_list.txt"
if [ "$SUITE" = "huggingface" ]; then
    MODEL_LIST="benchmarks/dynamo/huggingface_models_list.txt"
fi

extract_model() {
    local model="$1"
    local mode="$2"  # inference or training
    local model_repro_dir="$REPRO_DIR/${model}_${mode}"

    if [ -d "$model_repro_dir" ] && [ "$(find "$model_repro_dir" -name '*.py' | wc -l)" -gt 0 ]; then
        echo "  [skip] $model ($mode) — already extracted"
        return 0
    fi

    mkdir -p "$model_repro_dir"

    local mode_flag="--inference"
    if [ "$mode" = "training" ]; then
        mode_flag="--training --amp"
    else
        mode_flag="--inference --bfloat16"
    fi

    echo "  [extract] $model ($mode)..."
    timeout 600 python3 -c "
import sys, os
sys.path.insert(0, '$SCRIPT_DIR')
os.environ['EXTRACTION_OUTPUT_DIR'] = '$model_repro_dir'

from extract_reductions import ReductionExtractor
import torch
import torch._inductor.config as cfg
from torch._inductor.scheduler import Scheduler

cfg.force_disable_caches = True

extractor = ReductionExtractor('$model_repro_dir')

# Monkey-patch the scheduler to capture reductions
_orig_codegen = Scheduler.codegen
def _patched_codegen(self):
    for node in self.nodes:
        try:
            extractor.capture_node(node)
        except Exception:
            pass
    return _orig_codegen(self)
Scheduler.codegen = _patched_codegen

# Now run the model
import importlib.util
spec = importlib.util.spec_from_file_location('bench', 'benchmarks/dynamo/${SUITE}.py')
mod = importlib.util.module_from_spec(spec)
sys.argv = ['bench', $mode_flag, '--inductor', '--only', '$model', '--performance', '--disable-cudagraphs']
try:
    spec.loader.exec_module(mod)
except SystemExit:
    pass

import json
with open('$model_repro_dir/extraction_summary.json', 'w') as f:
    json.dump({'model': '$model', 'mode': '$mode', 'captured': len(extractor.captured)}, f)
" 2>"$model_repro_dir/extract_stderr.log" || {
        echo "    FAILED: $model ($mode)"
        return 0
    }
    local count=$(find "$model_repro_dir" -name "region_*.py" | wc -l)
    echo "    extracted $count regions"
}

# Phase 1a: Run inference for all models
echo ""
echo "--- Inference ---"
while IFS=',' read -r name _rest; do
    [ -z "$name" ] && continue
    extract_model "$name" "inference"
done < "$MODEL_LIST"

# Phase 1b: Run training for all models
echo ""
echo "--- Training ---"
while IFS=',' read -r name _rest; do
    [ -z "$name" ] && continue
    extract_model "$name" "training"
done < "$MODEL_LIST"

echo ""
echo "=== Phase 1 Complete ==="
echo "Total repros: $(find "$REPRO_DIR" -name 'region_*.py' | wc -l)"

# Phase 2: SOL benchmark all extracted repros
echo ""
echo "=== Phase 2: SOL Benchmark ==="
SOL_RESULTS="$OUTPUT_BASE/sol_results_${SUITE}.json"

# Collect all repro files
find "$REPRO_DIR" -name "region_*.py" > "$OUTPUT_BASE/repro_filelist_${SUITE}.txt"
REPRO_COUNT=$(wc -l < "$OUTPUT_BASE/repro_filelist_${SUITE}.txt")

if [ "$REPRO_COUNT" -eq 0 ]; then
    echo "No repros found, skipping SOL benchmark"
else
    echo "Benchmarking $REPRO_COUNT repros..."
    python3 "$SCRIPT_DIR/benchmark_sol_parallel.py" \
        --filelist "$OUTPUT_BASE/repro_filelist_${SUITE}.txt" \
        --output "$SOL_RESULTS" \
        --warmup 5 --iter 20 \
        2>"$OUTPUT_BASE/sol_benchmark_stderr_${SUITE}.log" || {
            echo "SOL benchmark had errors (see sol_benchmark_stderr_${SUITE}.log)"
        }

    if [ -f "$SOL_RESULTS" ]; then
        echo "Results: $SOL_RESULTS"
        # Quick summary
        python3 -c "
import json
with open('$SOL_RESULTS') as f:
    results = json.load(f)
total = len(results)
gaps = [r['graph_gap'] for r in results if r.get('graph_gap')]
if gaps:
    gaps.sort(reverse=True)
    print(f'  Total kernels: {total}')
    print(f'  With graph_gap: {len(gaps)}')
    print(f'  >= 2x gap: {sum(1 for g in gaps if g >= 2)}')
    print(f'  >= 1.5x gap: {sum(1 for g in gaps if g >= 1.5)}')
    print(f'  Top 5 gaps: {gaps[:5]}')
else:
    print(f'  Total: {total}, no graph_gap data')
"
    fi
fi

# Phase 3: Analysis summary
echo ""
echo "=== Phase 3: Analysis ==="
python3 -c "
import json, os
from collections import defaultdict

sol_file = '$SOL_RESULTS'
if not os.path.exists(sol_file):
    print('No SOL results to analyze')
    exit()

with open(sol_file) as f:
    results = json.load(f)

# Categorize by gap severity
severe = []  # >= 2x
moderate = []  # 1.5-2x
ok = []  # < 1.5x

for r in results:
    gap = r.get('graph_gap') or r.get('gap', 1.0)
    if gap >= 2.0:
        severe.append(r)
    elif gap >= 1.5:
        moderate.append(r)
    else:
        ok.append(r)

print(f'Gap distribution:')
print(f'  >= 2.0x (severe):   {len(severe)}')
print(f'  1.5-2.0x (moderate): {len(moderate)}')
print(f'  < 1.5x (ok):        {len(ok)}')
print()

# Save detailed analysis
analysis = {
    'suite': '$SUITE',
    'total_kernels': len(results),
    'severe_count': len(severe),
    'moderate_count': len(moderate),
    'ok_count': len(ok),
    'severe_kernels': sorted(severe, key=lambda x: x.get('graph_gap', x.get('gap', 0)), reverse=True)[:20],
    'moderate_kernels': sorted(moderate, key=lambda x: x.get('graph_gap', x.get('gap', 0)), reverse=True),
}

analysis_file = '$OUTPUT_BASE/analysis_${SUITE}.json'
with open(analysis_file, 'w') as f:
    json.dump(analysis, f, indent=2)
print(f'Analysis saved to: {analysis_file}')

# Print top offenders
if severe:
    print()
    print('Top 10 worst kernels:')
    for r in sorted(severe, key=lambda x: x.get('graph_gap', x.get('gap', 0)), reverse=True)[:10]:
        gap = r.get('graph_gap', r.get('gap'))
        path = r.get('file', r.get('path', '?'))
        model = os.path.basename(os.path.dirname(path)) if path != '?' else '?'
        print(f'  {gap:.2f}x  {model}  {os.path.basename(path)}')
"

echo ""
echo "========================================"
echo " Pipeline Complete: $SUITE"
echo " $(date)"
echo " Output: $OUTPUT_BASE/"
echo "========================================"
