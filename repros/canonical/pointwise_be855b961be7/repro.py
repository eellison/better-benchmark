"""
Standalone repro captured via capture_hook.
Label: torchbench_dlrm_train
Pattern hash: be855b961be7
Shape hash: 5b3edc3a
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([2048, 100], f32), T([36], i64), T([36], i64))"

class Repro(torch.nn.Module):
    def forward(self, mm_5: "f32[2048, 100]", _tensor_constant0: "i64[36]", _tensor_constant1: "i64[36]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:337 in interact_features, code: R = torch.cat([x] + [Zflat], dim=1)
        slice_tensor: "f32[2048, 36]" = torch.ops.aten.slice.Tensor(mm_5, 1, 64, 100);  mm_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:335 in interact_features, code: Zflat = Z[:, li, lj]
        full_default: "f32[2048, 9, 9]" = torch.ops.aten.full.default([2048, 9, 9], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:329 in interact_features, code: li = torch.tensor(
        lift_fresh_copy_default: "i64[36]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:332 in interact_features, code: lj = torch.tensor(
        lift_fresh_copy_default_1: "i64[36]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dlrm/dlrm_s_pytorch.py:335 in interact_features, code: Zflat = Z[:, li, lj]
        index_put_default: "f32[2048, 9, 9]" = torch.ops.aten.index_put.default(full_default, [None, lift_fresh_copy_default, lift_fresh_copy_default_1], slice_tensor, True);  full_default = lift_fresh_copy_default = lift_fresh_copy_default_1 = slice_tensor = None
        return index_put_default



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
