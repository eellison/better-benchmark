"""
Capture fused reduction scheduler nodes from inductor compilation.

Uses config._post_fusion_custom_pass to intercept scheduler nodes after fusion,
extract reduction metadata, and serialize standalone FX graph repros.
"""

import collections
import hashlib
import json
import os
import sys
import textwrap
import time
from pathlib import Path

import torch
import torch._inductor.config as inductor_config
from torch._inductor.scheduler import (
    BaseSchedulerNode,
    FusedSchedulerNode,
    SchedulerNode,
)
from torch._inductor.ir import ComputedBuffer, Reduction

CAPTURED_REDUCTIONS: list[dict] = []
SEEN_HASHES: set[str] = set()


def _get_reduction_info(snode: SchedulerNode) -> dict | None:
    """Extract reduction metadata from a single SchedulerNode."""
    if not snode.is_reduction():
        return None
    node = snode.node
    if not isinstance(node, ComputedBuffer):
        return None
    data = node.data
    if not isinstance(data, Reduction):
        return None

    ranges = [str(r) for r in data.ranges]
    reduction_ranges = [str(r) for r in data.reduction_ranges]
    reduction_type = data.reduction_type
    dtype = str(data.dtype)
    src_dtype = str(data.src_dtype)
    device = str(data.device)

    # split reduction info
    split_size = int(node._split_size) if node._split_size is not None else None
    original_ranges = (
        [str(r) for r in node._original_ranges] if node._original_ranges else None
    )
    original_reduction_ranges = (
        [str(r) for r in node._original_reduction_ranges]
        if node._original_reduction_ranges
        else None
    )

    # origin FX nodes
    origins = []
    try:
        for o in node.get_origins():
            origins.append(
                {
                    "op": o.op,
                    "target": str(o.target),
                    "name": o.name,
                }
            )
    except Exception:
        pass

    return {
        "name": snode.get_name(),
        "ranges": ranges,
        "reduction_ranges": reduction_ranges,
        "reduction_type": reduction_type,
        "dtype": dtype,
        "src_dtype": src_dtype,
        "device": device,
        "split_size": split_size,
        "original_ranges": original_ranges,
        "original_reduction_ranges": original_reduction_ranges,
        "origins": origins,
        "group": str(snode.group),
    }


def _hash_reduction(info: dict) -> str:
    """Hash reduction by its structural properties for dedup."""
    key = (
        tuple(info["ranges"]),
        tuple(info["reduction_ranges"]),
        info["reduction_type"],
        info["dtype"],
        info["src_dtype"],
    )
    return hashlib.md5(str(key).encode()).hexdigest()[:12]


def post_fusion_capture(nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """Post-fusion custom pass that captures reduction scheduler nodes."""
    for node in nodes:
        reduction_infos = []

        if isinstance(node, FusedSchedulerNode):
            has_reduction = False
            for snode in node.snodes:
                if isinstance(snode, SchedulerNode):
                    info = _get_reduction_info(snode)
                    if info:
                        has_reduction = True
                        reduction_infos.append(info)
            if not has_reduction:
                continue
        elif isinstance(node, SchedulerNode):
            info = _get_reduction_info(node)
            if not info:
                continue
            reduction_infos.append(info)
        else:
            continue

        # Build a combined entry for this fused group
        entry = {
            "node_type": type(node).__name__,
            "node_name": node.get_name(),
            "is_fused": isinstance(node, FusedSchedulerNode),
            "num_subnodes": (
                len(node.snodes) if isinstance(node, FusedSchedulerNode) else 1
            ),
            "reductions": reduction_infos,
        }

        # Dedup by hashing the reduction signatures
        combined_hash = hashlib.md5(
            json.dumps(reduction_infos, sort_keys=True).encode()
        ).hexdigest()[:16]
        if combined_hash not in SEEN_HASHES:
            SEEN_HASHES.add(combined_hash)
            entry["hash"] = combined_hash
            CAPTURED_REDUCTIONS.append(entry)

    return nodes


def setup_capture():
    """Install the post-fusion capture hook and disable split reductions."""
    inductor_config._post_fusion_custom_pass = post_fusion_capture
    inductor_config.split_reductions = False


def print_summary():
    """Print a summary of captured reductions."""
    print(f"\n{'='*80}")
    print(f"CAPTURED {len(CAPTURED_REDUCTIONS)} UNIQUE REDUCTION GROUPS")
    print(f"{'='*80}")

    for i, entry in enumerate(CAPTURED_REDUCTIONS):
        print(f"\n--- Reduction Group {i} [{entry['hash']}] ---")
        print(f"  Node: {entry['node_name']} ({entry['node_type']})")
        print(f"  Fused: {entry['is_fused']} ({entry['num_subnodes']} subnodes)")
        for j, red in enumerate(entry["reductions"]):
            print(f"  Reduction {j}:")
            print(f"    type={red['reduction_type']}, dtype={red['dtype']}")
            print(f"    ranges={red['ranges']}")
            print(f"    reduction_ranges={red['reduction_ranges']}")
            if red["split_size"]:
                print(f"    split_size={red['split_size']}")
                print(f"    original_ranges={red['original_ranges']}")
                print(f"    original_reduction_ranges={red['original_reduction_ranges']}")
            if red["origins"]:
                origin_ops = [f"{o['target']}" for o in red["origins"]]
                print(f"    origins: {origin_ops}")


def save_results(output_dir: str):
    """Save captured reductions to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "captured_reductions.json")
    with open(path, "w") as f:
        json.dump(CAPTURED_REDUCTIONS, f, indent=2)
    print(f"\nSaved {len(CAPTURED_REDUCTIONS)} reduction groups to {path}")


# --- Test models ---


def make_simple_reduction_model():
    """A simple model with several reduction patterns."""

    class ReduceModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = torch.nn.LayerNorm(768)
            self.linear = torch.nn.Linear(768, 768)

        def forward(self, x):
            x = self.ln(x)
            x = self.linear(x)
            x = x.softmax(dim=-1)
            return x

    return ReduceModel().cuda()


def make_rms_norm_model():
    """RMSNorm-style reduction."""

    class RMSNorm(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(dim))
            self.eps = 1e-6

        def forward(self, x):
            norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return x * norm * self.weight

    return RMSNorm(768).cuda()


def make_cross_entropy_model():
    """Cross entropy with reduction."""

    class CEModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(768, 32000)

        def forward(self, x, targets):
            logits = self.linear(x)
            return torch.nn.functional.cross_entropy(logits, targets)

    return CEModel().cuda()


def run_probe(model_name="simple", batch_size=32, seq_len=512, dim=768):
    """Run the probe on a test model."""
    setup_capture()
    CAPTURED_REDUCTIONS.clear()
    SEEN_HASHES.clear()

    print(f"Compiling {model_name} model (batch={batch_size}, seq={seq_len}, dim={dim})...")

    if model_name == "simple":
        model = make_simple_reduction_model()
        x = torch.randn(batch_size, seq_len, dim, device="cuda")
        compiled = torch.compile(model)
        with torch.no_grad():
            out = compiled(x)
    elif model_name == "rmsnorm":
        model = make_rms_norm_model()
        x = torch.randn(batch_size, seq_len, dim, device="cuda")
        compiled = torch.compile(model)
        with torch.no_grad():
            out = compiled(x)
    elif model_name == "cross_entropy":
        model = make_cross_entropy_model()
        x = torch.randn(batch_size * seq_len, dim, device="cuda")
        targets = torch.randint(0, 32000, (batch_size * seq_len,), device="cuda")
        compiled = torch.compile(model)
        with torch.no_grad():
            out = compiled(x, targets)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print_summary()
    return CAPTURED_REDUCTIONS


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "simple"
    results = run_probe(model_name)
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    save_results(output_dir)
