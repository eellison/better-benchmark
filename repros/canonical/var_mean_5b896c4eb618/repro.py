"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_infer
Pattern hash: 5b896c4eb618
Shape hash: f5c3276b
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([32768, 512], f32), T([512], f32), T([512], f32), T([30522, 128], f32), T([384, 30522], f32), S([256, 128, 512]), S([32768, 512]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_361: "f32[32768, 512]", arg1115_1: "f32[512]", arg1116_1: "f32[512]", arg2_1: "f32[30522, 128]", arg1117_1: "f32[384, 30522]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:489 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_361, _shape_param_0);  addmm_361 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:490 in forward, code: hidden_states = self.transform_act_fn(hidden_states)
        relu_default: "f32[256, 128, 512]" = torch.ops.aten.relu.default(reshape_default);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:491 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(relu_default, [2], correction = 0, keepdim = True)
        getitem: "f32[256, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[256, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[256, 128, 512]" = torch.ops.aten.sub.Tensor(relu_default, getitem_1);  relu_default = getitem_1 = None
        add_tensor: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, arg1115_1);  mul_tensor = arg1115_1 = None
        add_tensor_1: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg1116_1);  mul_tensor_1 = arg1116_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:507 in forward, code: hidden_states = hidden_states.matmul(torch.cat([self.decoder.weight.t(), self.dense.weight], dim=0))
        reshape_default_1: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_1);  add_tensor_1 = _shape_param_1 = None
        permute_default: "f32[128, 30522]" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        cat_default: "f32[512, 30522]" = torch.ops.aten.cat.default([permute_default, arg1117_1]);  permute_default = arg1117_1 = None
        constant_pad_nd_default: "f32[512, 30524]" = torch.ops.aten.constant_pad_nd.default(cat_default, [0, 2, 0, 0]);  cat_default = None
        return (reshape_default_1, constant_pad_nd_default)


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
