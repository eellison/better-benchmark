#!/bin/bash
# Run benchmark models in parallel across 2 GPUs, both inference and training.
# GPU 0: inference for remaining models (second half of list)
# GPU 1: inference for remaining models (first batch that GPU0's job hasn't reached)
# Then training on both GPUs.
set -euo pipefail

PYTORCH_DIR="/tmp/pytorch-work"
OUTPUT_DIR="/tmp/benchmark_traces"
TRAINING_OUTPUT_DIR="/tmp/benchmark_traces_training"
SUITE="${1:-huggingface}"

cd "$PYTORCH_DIR"
mkdir -p "$OUTPUT_DIR/$SUITE" "$TRAINING_OUTPUT_DIR/$SUITE"

# Read all models
mapfile -t ALL_MODELS < <(awk -F',' '{print $1}' "benchmarks/dynamo/${SUITE}_models_list.txt" | grep -v '^$')

echo "Total models: ${#ALL_MODELS[@]}"

run_inference() {
    local gpu="$1"
    local model="$2"
    local model_dir="$OUTPUT_DIR/$SUITE/$model"

    # Skip if already done (has output_code.py)
    if find "$model_dir" -name 'output_code.py' 2>/dev/null | grep -q .; then
        echo "  [GPU$gpu] SKIP inference $model (already done)"
        return 0
    fi

    mkdir -p "$model_dir"
    echo "  [GPU$gpu] inference $model ..."
    CUDA_VISIBLE_DEVICES=$gpu \
    TORCH_COMPILE_DEBUG=1 \
    TORCH_COMPILE_DEBUG_DIR="$model_dir" \
    TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
    timeout 600 python3 benchmarks/dynamo/"$SUITE".py \
        --inference --bfloat16 --inductor \
        --only "$model" \
        --performance --disable-cudagraphs \
        --output "$model_dir/perf.csv" \
        2>"$model_dir/stderr.log" || {
            echo "  [GPU$gpu] FAILED inference $model"
            return 0
        }
    echo "  [GPU$gpu] DONE inference $model"
}

run_training() {
    local gpu="$1"
    local model="$2"
    local model_dir="$TRAINING_OUTPUT_DIR/$SUITE/$model"

    # Skip if already done
    if find "$model_dir" -name 'output_code.py' 2>/dev/null | grep -q .; then
        echo "  [GPU$gpu] SKIP training $model (already done)"
        return 0
    fi

    mkdir -p "$model_dir"
    echo "  [GPU$gpu] training $model ..."
    CUDA_VISIBLE_DEVICES=$gpu \
    TORCH_COMPILE_DEBUG=1 \
    TORCH_COMPILE_DEBUG_DIR="$model_dir" \
    TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
    timeout 600 python3 benchmarks/dynamo/"$SUITE".py \
        --training --amp --inductor \
        --only "$model" \
        --performance --disable-cudagraphs \
        --output "$model_dir/perf.csv" \
        2>"$model_dir/stderr.log" || {
            echo "  [GPU$gpu] FAILED training $model"
            return 0
        }
    echo "  [GPU$gpu] DONE training $model"
}

# Split models into two halves
HALF=$(( ${#ALL_MODELS[@]} / 2 ))

echo ""
echo "=== Phase 1: Inference (parallel on 2 GPUs) ==="
echo "GPU 0: models 0..$(($HALF-1)), GPU 1: models $HALF..$(( ${#ALL_MODELS[@]} - 1 ))"

# Run GPU 0 batch in background
(
    for i in $(seq 0 $(($HALF - 1))); do
        run_inference 0 "${ALL_MODELS[$i]}"
    done
) &
PID_GPU0=$!

# Run GPU 1 batch in foreground-ish
(
    for i in $(seq $HALF $(( ${#ALL_MODELS[@]} - 1 ))); do
        run_inference 1 "${ALL_MODELS[$i]}"
    done
) &
PID_GPU1=$!

wait $PID_GPU0 $PID_GPU1
echo "=== Inference complete ==="
echo "Output code files: $(find "$OUTPUT_DIR/$SUITE" -name 'output_code.py' | wc -l)"

echo ""
echo "=== Phase 2: Training (parallel on 2 GPUs) ==="

# Run GPU 0 batch in background
(
    for i in $(seq 0 $(($HALF - 1))); do
        run_training 0 "${ALL_MODELS[$i]}"
    done
) &
PID_GPU0=$!

# Run GPU 1 batch
(
    for i in $(seq $HALF $(( ${#ALL_MODELS[@]} - 1 ))); do
        run_training 1 "${ALL_MODELS[$i]}"
    done
) &
PID_GPU1=$!

wait $PID_GPU0 $PID_GPU1
echo "=== Training complete ==="
echo "Output code files: $(find "$TRAINING_OUTPUT_DIR/$SUITE" -name 'output_code.py' | wc -l)"

echo ""
echo "=== Summary ==="
echo "Inference output_code: $(find "$OUTPUT_DIR/$SUITE" -name 'output_code.py' | wc -l)"
echo "Training output_code: $(find "$TRAINING_OUTPUT_DIR/$SUITE" -name 'output_code.py' | wc -l)"
echo "Total size: $(du -sh "$OUTPUT_DIR/$SUITE" 2>/dev/null | cut -f1) (inference), $(du -sh "$TRAINING_OUTPUT_DIR/$SUITE" 2>/dev/null | cut -f1) (training)"
