"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_infer
Pattern hash: 3be0fc282e4b
Shape hash: 69c15cc0
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([8192, 768], f32), T([50265], f32), T([768], f32), T([768], f32), T([50265, 768], f32), S([8, 1024, 768]), S([8192, 768]))"

class Repro(torch.nn.Module):
    def forward(self, addmm: "f32[8192, 768]", arg6_1: "f32[50265]", arg3_1: "f32[768]", arg4_1: "f32[768]", arg5_1: "f32[50265, 768]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1276 in forward, code: x = self.dense(features)
        reshape_default: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_tensor: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(reshape_default, 0.5)
        mul_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(reshape_default, 0.7071067811865476);  reshape_default = None
        erf_default: "f32[8, 1024, 768]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1278 in forward, code: x = self.layer_norm(x)
        var_mean_correction = torch.ops.aten.var_mean.correction(mul_tensor_2, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1024, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 1024, 1]" = var_mean_correction[1];  var_mean_correction = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1281 in forward, code: x = self.decoder(x)
        full_default: "f32[3]" = torch.ops.aten.full.default([3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default: "f32[50268]" = torch.ops.aten.cat.default([arg6_1, full_default]);  arg6_1 = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1278 in forward, code: x = self.layer_norm(x)
        sub_tensor: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_2, getitem_1);  mul_tensor_2 = getitem_1 = None
        add_tensor_1: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_4: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_3, arg3_1);  mul_tensor_3 = arg3_1 = None
        add_tensor_2: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_tensor_4, arg4_1);  mul_tensor_4 = arg4_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1281 in forward, code: x = self.decoder(x)
        reshape_default_1: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        permute_default: "f32[768, 50265]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        constant_pad_nd_default: "f32[768, 50268]" = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 3, 0, 0]);  permute_default = None
        return (cat_default, reshape_default_1, constant_pad_nd_default)


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
