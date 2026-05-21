"""
Standalone repro captured via capture_hook.
Label: torchbench_timm_vision_transformer_infer
Pattern hash: 293c509399b8
Shape hash: 8c137207
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
_shapes_config = "(T([6304, 384], f16), T([32, 197, 384], f16), T([384], f16), T([384], f16), S([32, 197, 384]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_47: "f16[6304, 384]", add_80: "f16[32, 197, 384]", arg149_1: "f16[384]", arg150_1: "f16[384]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f16[32, 197, 384]" = torch.ops.aten.reshape.default(addmm_47, _shape_param_0);  addmm_47 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_tensor: "f16[32, 197, 384]" = torch.ops.aten.add.Tensor(add_80, reshape_default);  add_80 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_default: "f32[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 197, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 197, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[32, 197, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1);  convert_element_type_default = getitem_1 = None
        add_tensor_1: "f32[32, 197, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt_default: "f32[32, 197, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(mul_tensor, arg149_1);  mul_tensor = arg149_1 = None
        add_tensor_2: "f32[32, 197, 384]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg150_1);  mul_tensor_1 = arg150_1 = None
        convert_element_type_default_1: "f16[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float16);  add_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:696 in global_pool_nlc, code: x = x[:, 0]  # class token
        select_int: "f16[32, 384]" = torch.ops.aten.select.int(convert_element_type_default_1, 1, 0);  convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1291 in forward_head, code: x = self.head_drop(x)
        clone_default: "f16[32, 384]" = torch.ops.aten.clone.default(select_int);  select_int = None
        return clone_default



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
