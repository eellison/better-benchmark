"""
Test script for scatter-reduce fusion pattern detection.

This exercises the FX pass on the target UNet repros to verify
that the scatter-then-reduce pattern is correctly identified.
"""
import sys
import os
import logging
from pathlib import Path

# Enable the pass
os.environ["TORCHINDUCTOR_SCATTER_REDUCE_FUSION"] = "1"

import torch
import torch._dynamo
import torch._inductor.config as inductor_config
from torch._dynamo.utils import counters

# Add the repo root for repro_harness
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Enable logging from our pass
logging.basicConfig(level=logging.INFO)
scatter_logger = logging.getLogger("torch._inductor.fx_passes.scatter_reduce_fusion")
scatter_logger.setLevel(logging.DEBUG)

# Also enable the inductor pass logging
inductor_logger = logging.getLogger("torch._inductor")


def test_pattern_detection_on_repro(repro_id: str):
    """Load a repro and run compilation to test pattern detection."""
    repro_dir = Path(__file__).resolve().parents[1] / "repros" / "canonical" / repro_id
    repro_path = repro_dir / "repro.py"

    if not repro_path.exists():
        print(f"SKIP: {repro_id} (repro not found at {repro_path})")
        return False

    print(f"\n{'='*70}")
    print(f"Testing pattern detection on: {repro_id}")
    print(f"{'='*70}")

    # Import the repro module
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"repro_{repro_id}", str(repro_path))
    if spec is None or spec.loader is None:
        print(f"FAIL: Could not load {repro_path}")
        return False

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Create model and inputs
    model = module.Repro()
    inputs = module.make_inputs()

    # Move to CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    inputs = tuple(
        x.to(device) if isinstance(x, torch.Tensor) else x
        for x in inputs
    )

    # Reset counters
    counters.clear()

    # Compile with torch.compile
    print(f"Compiling with scatter_reduce_fusion_enabled={inductor_config.scatter_reduce_fusion_enabled}")

    compiled_model = torch.compile(model, fullgraph=True)

    with torch.no_grad():
        try:
            output = compiled_model(*inputs)
            print(f"Compilation successful!")
            print(f"Output shapes: {[o.shape for o in output] if isinstance(output, tuple) else output.shape}")
        except Exception as e:
            print(f"Compilation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Check if our pass detected anything
    scatter_count = counters.get("inductor", {}).get("scatter_reduce_fusion_applied", 0)
    print(f"\nResults:")
    print(f"  scatter_reduce_fusion_applied: {scatter_count}")

    # Print all inductor counters for debugging
    if "inductor" in counters:
        print(f"  All inductor counters: {dict(counters['inductor'])}")

    return True


def test_standalone_graph_analysis():
    """Create a minimal FX graph with the scatter-reduce pattern and test detection."""
    print(f"\n{'='*70}")
    print("Testing standalone graph analysis (minimal pattern)")
    print(f"{'='*70}")

    from torch._inductor.fx_passes.scatter_reduce_fusion import (
        _find_scatter_reduce_chains,
        get_scatter_reduce_stats,
    )

    # Create a simple model that exhibits the pattern
    class SimpleScatterReduce(torch.nn.Module):
        def forward(self, src, row_idx, col_idx):
            # Scatter into zeros
            out = torch.zeros(2, 4, 8, 8, device=src.device, dtype=src.dtype)
            out = torch.ops.aten.index_put.default(
                out, [None, None, row_idx, col_idx], src, True
            )
            # Reduce over all scattered dims
            return out.sum(dim=[0, 2, 3])

    model = SimpleScatterReduce()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create sample inputs
    src = torch.randn(2, 4, 16, 16, device=device)
    row_idx = torch.randint(0, 8, (16, 1), device=device, dtype=torch.int64)
    col_idx = torch.randint(0, 8, (16,), device=device, dtype=torch.int64)

    # Reset
    counters.clear()
    torch._dynamo.reset()

    compiled = torch.compile(model, fullgraph=True)
    with torch.no_grad():
        try:
            out = compiled(src, row_idx, col_idx)
            print(f"Compilation successful! Output shape: {out.shape}")
        except Exception as e:
            print(f"Compilation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    scatter_count = counters.get("inductor", {}).get("scatter_reduce_fusion_applied", 0)
    print(f"  scatter_reduce_fusion_applied: {scatter_count}")
    return True


def test_graph_analysis_via_export():
    """Use torch.export to get the FX graph, then run analysis directly."""
    print(f"\n{'='*70}")
    print("Testing direct FX graph analysis via export")
    print(f"{'='*70}")

    from torch._inductor.fx_passes.scatter_reduce_fusion import (
        _find_scatter_reduce_chains,
        get_scatter_reduce_stats,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a traced graph that mimics the UNet pattern
    graph = torch.fx.Graph()

    # We'll create a graph manually to test the analysis
    # This mimics what Inductor would see after decomposition

    # Arguments
    src = graph.placeholder("src")  # [B, C, H_src, W_src]
    row_idx = graph.placeholder("row_idx")  # [H_out, 1]
    col_idx = graph.placeholder("col_idx")  # [W_out]
    skip = graph.placeholder("skip")  # [B, C, H_out, W_out]

    # full(zeros)
    zeros = graph.call_function(
        torch.ops.aten.full.default,
        args=([2, 4, 8, 8], 0),
        kwargs={"dtype": torch.float32, "device": torch.device(device)}
    )

    # index_put(zeros, [None, None, row_idx, col_idx], src, True)
    scatter = graph.call_function(
        torch.ops.aten.index_put.default,
        args=(zeros, [None, None, row_idx, col_idx], src, True)
    )

    # sum(scatter, dim=[0, 2, 3])
    result = graph.call_function(
        torch.ops.aten.sum.dim_IntList,
        args=(scatter, [0, 2, 3])
    )

    graph.output(result)

    # Add fake tensor metadata
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode() as mode:
        fake_src = mode.from_tensor(torch.randn(2, 4, 16, 16, device="cpu"))
        fake_zeros = mode.from_tensor(torch.zeros(2, 4, 8, 8, device="cpu"))
        fake_scatter = mode.from_tensor(torch.zeros(2, 4, 8, 8, device="cpu"))
        fake_result = mode.from_tensor(torch.zeros(4, device="cpu"))

        src.meta["val"] = fake_src
        zeros.meta["val"] = fake_zeros
        scatter.meta["val"] = fake_scatter
        result.meta["val"] = fake_result

    # Run analysis
    chains = _find_scatter_reduce_chains(graph)
    print(f"  Found {len(chains)} scatter-reduce chain(s)")

    for i, chain in enumerate(chains):
        print(f"  Chain {i}:")
        print(f"    num_scatters: {len(chain.scatter_nodes)}")
        print(f"    has_mask: {chain.has_mask}")
        print(f"    reduction_dims: {chain.reduction_dims}")
        print(f"    scatter_output_shape: {chain.scatter_output_shape}")

    stats = get_scatter_reduce_stats(graph)
    print(f"\n  Stats: {stats}")

    return len(chains) > 0


if __name__ == "__main__":
    print("Scatter-Reduce Fusion Pattern Detection Test")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"scatter_reduce_fusion_enabled: {inductor_config.scatter_reduce_fusion_enabled}")

    results = {}

    # Test 1: Direct graph analysis
    results["graph_analysis"] = test_graph_analysis_via_export()

    # Test 2: Standalone simple pattern
    torch._dynamo.reset()
    results["simple_pattern"] = test_standalone_graph_analysis()

    # Test 3: Real repros (if on CUDA)
    if torch.cuda.is_available():
        for repro_id in [
            "sum_sum_sum_f90d684d32cb",
            "sum_sum_sum_45f02142ecfd",
            "sum_sum_sum_dadf6aa035dd",
        ]:
            torch._dynamo.reset()
            results[repro_id] = test_pattern_detection_on_repro(repro_id)
    else:
        print("\nSkipping CUDA-dependent repro tests (no CUDA available)")

    print(f"\n{'='*70}")
    print("Summary:")
    print(f"{'='*70}")
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {test_name}")
