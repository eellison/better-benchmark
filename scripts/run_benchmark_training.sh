#!/bin/bash
# Run benchmark suite models in TRAINING mode with TORCH_COMPILE_DEBUG.
set -euo pipefail

PYTORCH_DIR="/tmp/pytorch-work"
OUTPUT_DIR="/tmp/benchmark_traces_training"
SUITE="${1:-huggingface}"
MODEL="${2:-all}"

cd "$PYTORCH_DIR"
mkdir -p "$OUTPUT_DIR"

run_model() {
    local suite="$1"
    local model="$2"
    local model_dir="$OUTPUT_DIR/$suite/$model"
    mkdir -p "$model_dir"

    echo "=== Running $suite/$model (training) ==="
    TORCH_COMPILE_DEBUG=1 \
    TORCH_COMPILE_DEBUG_DIR="$model_dir" \
    TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
    python3 benchmarks/dynamo/"$suite".py \
        --training --amp --inductor \
        --only "$model" \
        --performance --disable-cudagraphs \
        --output "$model_dir/perf.csv" \
        2>"$model_dir/stderr.log" || {
            echo "  FAILED: $model (see $model_dir/stderr.log)"
            return 0
        }
    echo "  Done: $model"
}

if [ "$MODEL" != "all" ]; then
    run_model "$SUITE" "$MODEL"
else
    MODEL_LIST="benchmarks/dynamo/${SUITE}_models_list.txt"
    if [ "$SUITE" = "huggingface" ]; then
        MODEL_LIST="benchmarks/dynamo/huggingface_models_list.txt"
    fi

    while IFS=',' read -r name _rest; do
        [ -z "$name" ] && continue
        run_model "$SUITE" "$name"
    done < "$MODEL_LIST"
fi

echo ""
echo "=== Summary (training) ==="
echo "Output: $OUTPUT_DIR/$SUITE/"
echo "FX graphs: $(find "$OUTPUT_DIR/$SUITE" -name 'fx_graph_runnable.py' 2>/dev/null | wc -l)"
echo "Output code: $(find "$OUTPUT_DIR/$SUITE" -name 'output_code.py' 2>/dev/null | wc -l)"
echo "Total size: $(du -sh "$OUTPUT_DIR/$SUITE" 2>/dev/null | cut -f1)"
