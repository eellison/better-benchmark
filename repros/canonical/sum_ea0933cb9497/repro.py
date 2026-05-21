"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: ea0933cb9497
Shape hash: ffc7b387
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
_shapes_config = "(T([512, 1280], b8), T([512, 1280], f32), T([512, 1280, 1, 1], b8), S([512, 1280, 1, 1]))"

class Repro(torch.nn.Module):
    def forward(self, gt: "b8[512, 1280]", mm: "f32[512, 1280]", le: "b8[512, 1280, 1, 1]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:836 in forward_head, code: x = F.dropout(x, p=self.drop_rate, training=self.training)
        convert_element_type_default: "f32[512, 1280]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor: "f32[512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.25);  convert_element_type_default = None
        mul_tensor_1: "f32[512, 1280]" = torch.ops.aten.mul.Tensor(mm, mul_tensor);  mm = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:834 in forward_head, code: x = self.flatten(x)
        reshape_default: "f32[512, 1280, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_0);  mul_tensor_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:833 in forward_head, code: x = self.act2(x)
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[512, 1280, 1, 1]" = torch.ops.aten.where.self(le, full_default, reshape_default);  le = full_default = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:832 in forward_head, code: x = self.conv_head(x)
        sum_dim_int_list: "f32[1280]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3]);  where_self = None
        return sum_dim_int_list



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
