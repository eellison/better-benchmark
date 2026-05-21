"""
Standalone repro captured via capture_hook.
Label: torchbench_moco_infer
Pattern hash: a436db6611f3
Shape hash: ffd9ed70
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
_shapes_config = "(T([32, 128], f32), T([32], i64, gen=Index(32)), T([32, 128], f32), T([128, 32000], f32), S([32, 1, 128]), S([32, 128, 1]), S([1, 32, 128]), S([1, 128, 32000]))"

class Repro(torch.nn.Module):
    def forward(self, arg269_1: "f32[32, 128]", arg268_1: "i64[32]", wait_tensor: "f32[32, 128]", arg270_1: "f32[128, 32000]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:153 in torch_dynamo_resume_in_forward_at_142, code: l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        reshape_default: "f32[32, 1, 128]" = torch.ops.aten.reshape.default(arg269_1, _shape_param_0);  _shape_param_0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:120 in _batch_unshuffle_ddp, code: idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        reshape_default_1: "i64[1, 32]" = torch.ops.aten.reshape.default(arg268_1, [1, -1]);  arg268_1 = None
        select_int: "i64[32]" = torch.ops.aten.select.int(reshape_default_1, 0, 0);  reshape_default_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:122 in _batch_unshuffle_ddp, code: return x_gather[idx_this]
        index_tensor: "f32[32, 128]" = torch.ops.aten.index.Tensor(wait_tensor, [select_int]);  wait_tensor = select_int = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:153 in torch_dynamo_resume_in_forward_at_142, code: l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        reshape_default_2: "f32[32, 128, 1]" = torch.ops.aten.reshape.default(index_tensor, _shape_param_1);  index_tensor = _shape_param_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:155 in torch_dynamo_resume_in_forward_at_142, code: l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        unsqueeze_default: "f32[32, 128, 1]" = torch.ops.aten.unsqueeze.default(arg269_1, 2);  arg269_1 = None
        reshape_default_3: "f32[1, 32, 128]" = torch.ops.aten.reshape.default(unsqueeze_default, _shape_param_2);  unsqueeze_default = _shape_param_2 = None
        squeeze_dim: "f32[32, 128]" = torch.ops.aten.squeeze.dim(reshape_default_3, 0);  reshape_default_3 = None
        unsqueeze_default_1: "f32[128, 32000, 1]" = torch.ops.aten.unsqueeze.default(arg270_1, 2);  arg270_1 = None
        reshape_default_4: "f32[1, 128, 32000]" = torch.ops.aten.reshape.default(unsqueeze_default_1, _shape_param_3);  unsqueeze_default_1 = _shape_param_3 = None
        squeeze_dim_1: "f32[128, 32000]" = torch.ops.aten.squeeze.dim(reshape_default_4, 0);  reshape_default_4 = None
        return (reshape_default, reshape_default_2, squeeze_dim, squeeze_dim_1)



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
