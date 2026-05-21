"""
Standalone repro captured via capture_hook.
Label: torchbench_moco_infer
Pattern hash: ade13fba7659
Shape hash: 5d885fec
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
_shapes_config = "(T([32], i64, gen=Index(32)), T([32, 128], f32), S([32, 128, 1]))"

class Repro(torch.nn.Module):
    def forward(self, arg268_1: "i64[32]", wait_tensor: "f32[32, 128]", _shape_param_0):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:120 in _batch_unshuffle_ddp, code: idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        reshape_default: "i64[1, 32]" = torch.ops.aten.reshape.default(arg268_1, [1, -1]);  arg268_1 = None
        select_int: "i64[32]" = torch.ops.aten.select.int(reshape_default, 0, 0);  reshape_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:122 in _batch_unshuffle_ddp, code: return x_gather[idx_this]
        index_tensor: "f32[32, 128]" = torch.ops.aten.index.Tensor(wait_tensor, [select_int]);  wait_tensor = select_int = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:153 in torch_dynamo_resume_in_forward_at_142, code: l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        reshape_default_1: "f32[32, 128, 1]" = torch.ops.aten.reshape.default(index_tensor, _shape_param_0);  index_tensor = _shape_param_0 = None
        return reshape_default_1



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
