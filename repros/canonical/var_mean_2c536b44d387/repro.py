"""
Standalone repro captured via capture_hook.
Label: hf_XLNetLMHeadModel_infer
Pattern hash: 2c536b44d387
Shape hash: cc7292e1
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([8192, 1024], f32), T([512, 16, 1024], f32), T([1024], f32), T([1024], f32), T([4096, 1024], f32), S([512, 16, 1, 1, 1024]), S([512, 16, 1024]), S([8192, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, mm_default: "f32[8192, 1024]", add_252: "f32[512, 16, 1024]", arg354_1: "f32[1024]", arg355_1: "f32[1024]", arg356_1: "f32[4096, 1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_default: "f32[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_default, 0);  mm_default = None
        reshape_default: "f32[512, 16, 1, 1, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default, _shape_param_0);  unsqueeze_default = _shape_param_0 = None
        permute_default: "f32[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(reshape_default, [0, 1, 4, 2, 3]);  reshape_default = None
        reshape_default_1: "f32[512, 16, 1024]" = torch.ops.aten.reshape.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_tensor: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(reshape_default_1, add_252);  reshape_default_1 = add_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[512, 16, 1]" = var_mean_correction[0]
        getitem_1: "f32[512, 16, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        add_tensor_1: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, arg354_1);  mul_tensor = arg354_1 = None
        add_tensor_2: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg355_1);  mul_tensor_1 = arg355_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        reshape_default_2: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_2);  add_tensor_2 = _shape_param_2 = None
        permute_default_1: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg356_1, [1, 0]);  arg356_1 = None
        return (reshape_default_2, permute_default_1)


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
