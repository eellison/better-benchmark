"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_train
Pattern hash: c2f21bf79643
Shape hash: 7437755c
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
_shapes_config = "(T([768, 128, 3, 3], f32, stride=(1152, 1, 384, 128)), T([768, 1, 1, 1], f32), S([1, 768, 1152]), S([768, 128, 3, 3]))"

class Repro(torch.nn.Module):
    def forward(self, primals_219: "f32[768, 128, 3, 3]", primals_220: "f32[768, 1, 1, 1]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(primals_219, memory_format = torch.contiguous_format);  primals_219 = None
        reshape_default: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_220, 0.02946278254943948);  primals_220 = None
        reshape_default_1: "f32[768]" = torch.ops.aten.reshape.default(mul_tensor, [-1]);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default, [0, 2], correction = 0, keepdim = True)
        getitem: "f32[1, 768, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 768, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 768, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 768, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default, getitem_1);  reshape_default = getitem_1 = None
        mul_tensor_1: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(reshape_default_1, -1);  reshape_default_1 = None
        mul_tensor_2: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default);  mul_tensor_1 = unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_2: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_2, _shape_param_1);  mul_tensor_2 = _shape_param_1 = None
        return reshape_default_2



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
