"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_train
Pattern hash: bd4890c0f5ed
Shape hash: 4ae91b05
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
_shapes_config = "(T([128, 768, 7, 7], f32, stride=(37632, 1, 5376, 768)), T([128, 768, 7, 7], f32, stride=(37632, 1, 5376, 768)), T([768], f32), T([768], f32), T([768], f32), T([768], f32), S([128, 768]))"

class Repro(torch.nn.Module):
    def forward(self, add_175: "f32[128, 768, 7, 7]", convolution_56: "f32[128, 768, 7, 7]", primals_201: "f32[768]", primals_202: "f32[768]", primals_203: "f32[768]", primals_204: "f32[768]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_tensor: "f32[128, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_175, convolution_56);  add_175 = convolution_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:468 in forward_features, code: x = self.norm(x)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 768, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 768, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt_default: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[128, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1)
        mul_tensor: "f32[128, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        squeeze_dims: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_dims_1: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_default, [0, 2, 3]);  rsqrt_default = None
        mul_tensor_1: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1)
        mul_tensor_2: "f32[768]" = torch.ops.aten.mul.Tensor(primals_201, 0.9)
        add_tensor_2: "f32[768]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        squeeze_dims_2: "f32[768]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_3: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_dims_2, 1.0001594642002871);  squeeze_dims_2 = None
        mul_tensor_4: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 0.1);  mul_tensor_3 = None
        mul_tensor_5: "f32[768]" = torch.ops.aten.mul.Tensor(primals_202, 0.9)
        add_tensor_3: "f32[768]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_203, -1);  primals_203 = None
        unsqueeze_default_1: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_6: "f32[128, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_204, -1);  primals_204 = None
        unsqueeze_default_3: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_4: "f32[128, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_6, unsqueeze_default_3);  mul_tensor_6 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_dim: "f32[128, 768, 1, 1]" = torch.ops.aten.mean.dim(add_tensor_4, [-1, -2], True);  add_tensor_4 = None
        as_strided_default: "f32[128, 768, 1, 1]" = torch.ops.aten.as_strided.default(mean_dim, [128, 768, 1, 1], [768, 1, 768, 768]);  mean_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        reshape_default: "f32[128, 768]" = torch.ops.aten.reshape.default(as_strided_default, _shape_param_0);  as_strided_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:468 in forward_features, code: x = self.norm(x)
        unsqueeze_default_4: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sub_tensor_1: "f32[128, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_tensor, unsqueeze_default_6);  add_tensor = unsqueeze_default_6 = None

        # No stacktrace found for following nodes
        copy__default: "f32[768]" = torch.ops.aten.copy_.default(primals_201, add_tensor_2);  primals_201 = add_tensor_2 = None
        copy__default_1: "f32[768]" = torch.ops.aten.copy_.default(primals_202, add_tensor_3);  primals_202 = add_tensor_3 = None
        return (sub_tensor_1, squeeze_dims_1, copy__default, copy__default_1, reshape_default)



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
