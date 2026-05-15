"""Extract individual Triton kernels from TORCH_COMPILE_DEBUG output_code.py files.

Parses output_code.py files, extracts each kernel as a standalone benchmarkable script
that can be run through our SOL pipeline.

Usage:
    python extract_kernels_from_debug.py /tmp/benchmark_traces/huggingface/ --output /tmp/kernel_repros/
"""
import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path


def find_output_code_files(base_dir):
    """Find all output_code.py files under the debug directory."""
    results = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f == "output_code.py":
                results.append(os.path.join(root, f))
    return sorted(results)


def parse_kernel_info(output_code_path):
    """Parse an output_code.py file and extract kernel information.

    Returns list of dicts with kernel name, source, metadata.
    """
    with open(output_code_path) as f:
        content = f.read()

    kernels = []

    # Find all triton kernel definitions (async_compile.triton calls)
    # Pattern: varname = async_compile.triton('kernel_name', '''...''', ...)
    pattern = r"(\w+)\s*=\s*async_compile\.triton\('([^']+)',\s*'''(.*?)'''(?:,\s*device_str='[^']*')?\)"
    for match in re.finditer(pattern, content, re.DOTALL):
        var_name = match.group(1)
        kernel_name = match.group(2)
        kernel_source = match.group(3)

        # Extract metadata from the kernel
        meta = {}

        # Get size_hints
        size_hints_match = re.search(r"size_hints=(\{[^}]+\})", kernel_source)
        if size_hints_match:
            try:
                meta["size_hints"] = eval(size_hints_match.group(1))
            except Exception:
                meta["size_hints_raw"] = size_hints_match.group(1)

        # Get reduction_hint
        hint_match = re.search(r"reduction_hint=ReductionHint\.(\w+)", kernel_source)
        if hint_match:
            meta["reduction_hint"] = hint_match.group(1)

        # Get kernel type (persistent_reduction, reduction, pointwise)
        type_match = re.search(r"@triton_heuristics\.(\w+)", kernel_source)
        if type_match:
            meta["kernel_type"] = type_match.group(1)

        # Get num_load, num_store, num_reduction from inductor_meta
        for field in ["num_load", "num_store", "num_reduction"]:
            field_match = re.search(rf"'{field}':\s*(\d+)", kernel_source)
            if field_match:
                meta[field] = int(field_match.group(1))

        # Get tiling_scores
        tiling_match = re.search(r"'tiling_scores':\s*(\{[^}]+\})", kernel_source)
        if tiling_match:
            try:
                meta["tiling_scores"] = eval(tiling_match.group(1))
            except Exception:
                pass

        kernels.append({
            "var_name": var_name,
            "kernel_name": kernel_name,
            "source": kernel_source,
            "meta": meta,
        })

    # Find the Runner.call method to understand data flow
    call_match = re.search(r"def call\(self, args\):(.*?)(?=\n    def |\nrunner = )", content, re.DOTALL)
    call_body = call_match.group(1) if call_match else ""

    # Find get_args for input shapes
    args_match = re.search(r"def get_args\(\):(.*?)(?=\ndef |\nif __name__)", content, re.DOTALL)
    args_body = args_match.group(1) if args_match else ""

    return {
        "kernels": kernels,
        "call_body": call_body,
        "args_body": args_body,
        "full_source": content,
        "path": output_code_path,
    }


def get_model_name_from_path(output_code_path):
    """Extract model name from the debug directory path."""
    parts = Path(output_code_path).parts
    # Pattern: .../model_name/torch_compile_debug/run_.../torchinductor/.../output_code.py
    for i, p in enumerate(parts):
        if p == "torch_compile_debug" and i > 0:
            return parts[i - 1]
    return "unknown"


def create_summary(base_dir, output_dir):
    """Create a summary of all kernels found."""
    output_code_files = find_output_code_files(base_dir)

    all_kernels = []
    for path in output_code_files:
        model = get_model_name_from_path(path)
        parsed = parse_kernel_info(path)
        for k in parsed["kernels"]:
            k["model"] = model
            k["output_code_path"] = path
            all_kernels.append(k)

    # Summary statistics
    kernel_types = {}
    reduction_hints = {}
    for k in all_kernels:
        kt = k["meta"].get("kernel_type", "unknown")
        kernel_types[kt] = kernel_types.get(kt, 0) + 1
        rh = k["meta"].get("reduction_hint", "none")
        reduction_hints[rh] = reduction_hints.get(rh, 0) + 1

    summary = {
        "total_output_code_files": len(output_code_files),
        "total_kernels": len(all_kernels),
        "kernel_types": kernel_types,
        "reduction_hints": reduction_hints,
        "models": list(set(k["model"] for k in all_kernels)),
    }

    os.makedirs(output_dir, exist_ok=True)

    # Save summary
    with open(os.path.join(output_dir, "kernel_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save per-kernel metadata (without full source to keep it small)
    kernel_index = []
    for k in all_kernels:
        kernel_index.append({
            "model": k["model"],
            "kernel_name": k["kernel_name"],
            "var_name": k["var_name"],
            "meta": k["meta"],
            "output_code_path": k["output_code_path"],
        })

    with open(os.path.join(output_dir, "kernel_index.json"), "w") as f:
        json.dump(kernel_index, f, indent=2)

    return summary, all_kernels


def main():
    parser = argparse.ArgumentParser(description="Extract kernels from TORCH_COMPILE_DEBUG output")
    parser.add_argument("base_dir", help="Base directory with model debug output")
    parser.add_argument("--output", "-o", default="/tmp/kernel_analysis", help="Output directory")
    parser.add_argument("--summary-only", action="store_true", help="Only generate summary, don't extract standalone scripts")
    args = parser.parse_args()

    print(f"Scanning {args.base_dir} for output_code.py files...")
    summary, all_kernels = create_summary(args.base_dir, args.output)

    print(f"\nResults:")
    print(f"  Output code files: {summary['total_output_code_files']}")
    print(f"  Total kernels: {summary['total_kernels']}")
    print(f"  Kernel types: {summary['kernel_types']}")
    print(f"  Reduction hints: {summary['reduction_hints']}")
    print(f"  Models: {len(summary['models'])}")
    print(f"\nSaved to: {args.output}/")


if __name__ == "__main__":
    main()
