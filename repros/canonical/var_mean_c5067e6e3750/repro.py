"""
Standalone repro captured via capture_hook.
Label: timm_deit_base_distilled_patch16_224_train
Pattern hash: c5067e6e3750
Shape hash: 575fc6d7
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
_shapes_config = "(T([25344, 768], f32), T([128, 198, 768], f32), T([768], f32), T([768], f32), S([128, 198, 768]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_47: "f32[25344, 768]", add_80: "f32[128, 198, 768]", primals_151: "f32[768]", primals_152: "f32[768]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[128, 198, 768]" = torch.ops.aten.reshape.default(addmm_47, _shape_param_0);  addmm_47 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_tensor: "f32[128, 198, 768]" = torch.ops.aten.add.Tensor(add_80, reshape_default);  add_80 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 198, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 198, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[128, 198, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt_default: "f32[128, 198, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[128, 198, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_1: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_151);  mul_tensor = primals_151 = None
        add_tensor_2: "f32[128, 198, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_152);  mul_tensor_1 = primals_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:114 in forward_head, code: x, x_dist = x[:, 0], x[:, 1]
        select_int: "f32[128, 768]" = torch.ops.aten.select.int(add_tensor_2, 1, 0)
        select_int_1: "f32[128, 768]" = torch.ops.aten.select.int(add_tensor_2, 1, 1);  add_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div_tensor: "f32[128, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_default, 768);  rsqrt_default = None
        return (select_int, select_int_1, div_tensor)



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
