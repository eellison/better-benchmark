"""
Standalone repro captured via capture_hook.
Label: torchbench_timm_nfnet_infer
Pattern hash: ab62ae1f1452
Shape hash: b6a5d428
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
_shapes_config = "(T([1536, 768, 1, 1], f16), T([128, 768, 6, 6], f16), T([1536, 1, 1, 1], f16), S([1, 1536, -1]), S([1536, 768, 1, 1]))"

class Repro(torch.nn.Module):
    def forward(self, arg221_1: "f16[1536, 768, 1, 1]", convolution_76: "f16[128, 768, 6, 6]", arg222_1: "f16[1536, 1, 1, 1]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default: "f16[1, 1536, 768]" = torch.ops.aten.reshape.default(arg221_1, _shape_param_0);  arg221_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        convert_element_type_default: "f32[1, 1536, 768]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [0, 2], correction = 0, keepdim = True);  convert_element_type_default = None
        getitem: "f32[1, 1536, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 1536, 1]" = var_mean_correction[1];  var_mean_correction = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        convert_element_type_default_1: "f32[128, 768, 6, 6]" = torch.ops.prims.convert_element_type.default(convolution_76, torch.float32);  convolution_76 = None
        mul_tensor: "f32[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 0.5)
        mul_tensor_1: "f32[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 0.7071067811865476);  convert_element_type_default_1 = None
        erf_default: "f32[128, 768, 6, 6]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[128, 768, 6, 6]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None
        convert_element_type_default_2: "f16[128, 768, 6, 6]" = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.float16);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_tensor_3: "f16[128, 768, 6, 6]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 1.7015043497085571);  convert_element_type_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(reshape_default, getitem_1);  reshape_default = getitem_1 = None
        add_tensor_1: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_4: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_5: "f16[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg222_1, 0.03608439182435161);  arg222_1 = None
        reshape_default_1: "f16[1536]" = torch.ops.aten.reshape.default(mul_tensor_5, [-1]);  mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        unsqueeze_default: "f16[1536, 1]" = torch.ops.aten.unsqueeze.default(reshape_default_1, -1);  reshape_default_1 = None
        mul_tensor_6: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_4, unsqueeze_default);  mul_tensor_4 = unsqueeze_default = None
        convert_element_type_default_3: "f16[1, 1536, 768]" = torch.ops.prims.convert_element_type.default(mul_tensor_6, torch.float16);  mul_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_2: "f16[1536, 768, 1, 1]" = torch.ops.aten.reshape.default(convert_element_type_default_3, _shape_param_1);  convert_element_type_default_3 = _shape_param_1 = None
        return (mul_tensor_3, reshape_default_2)



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
