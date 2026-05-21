"""
Standalone repro captured via capture_hook.
Label: torchbench_nanogpt_infer
Pattern hash: e371243e6081
Shape hash: ee4f09b9
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
_shapes_config = "(T([64, 768], f32), T([1, 64, 768], f32), T([768], f32), T([768], f32), T([3072, 768], f32), S([1, 64, 768]), S([64, 768]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_45: "f32[64, 768]", add_88: "f32[1, 64, 768]", arg141_1: "f32[768]", arg142_1: "f32[768]", arg143_1: "f32[3072, 768]", _shape_param_0, _shape_param_1):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:117 in forward, code: y = self.resid_dropout(self.c_proj(y))
        reshape_default: "f32[1, 64, 768]" = torch.ops.aten.reshape.default(addmm_45, _shape_param_0);  addmm_45 = _shape_param_0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:145 in forward, code: x = x + self.attn(self.ln_1(x))
        add_tensor: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_88, reshape_default);  add_88 = reshape_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:44 in forward, code: return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 64, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 64, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        add_tensor_1: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg141_1);  mul_tensor = arg141_1 = None
        add_tensor_2: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg142_1);  mul_tensor_1 = arg142_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/nanogpt/model.py:129 in forward, code: x = self.c_fc(x)
        reshape_default_1: "f32[64, 768]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        permute_default: "f32[768, 3072]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        return (reshape_default_1, permute_default)



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
