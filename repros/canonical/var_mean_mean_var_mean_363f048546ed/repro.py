"""
Standalone repro captured via capture_hook.
Label: timm_repvgg_a2_train
Pattern hash: 363f048546ed
Shape hash: 87b46cfc
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
_shapes_config = "(T([128, 1408, 7, 7], f32, stride=(68992, 1, 9856, 1408)), T([1408], f32), T([1408], f32), T([1408], f32), T([1408], f32), T([128, 1408, 7, 7], f32, stride=(68992, 1, 9856, 1408)), T([1408], f32), T([1408], f32), T([1408], f32), T([1408], f32), S([128, 1408]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_42: "f32[128, 1408, 7, 7]", primals_341: "f32[1408]", primals_342: "f32[1408]", primals_343: "f32[1408]", primals_344: "f32[1408]", convolution_43: "f32[128, 1408, 7, 7]", primals_347: "f32[1408]", primals_348: "f32[1408]", primals_349: "f32[1408]", primals_350: "f32[1408]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 1408, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 1408, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 1408, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt_default: "f32[1, 1408, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_1);  convolution_42 = None
        mul_tensor: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        squeeze_dims: "f32[1408]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        mul_tensor_1: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1);  squeeze_dims = None
        mul_tensor_2: "f32[1408]" = torch.ops.aten.mul.Tensor(primals_341, 0.9)
        add_tensor_1: "f32[1408]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        squeeze_dims_1: "f32[1408]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_3: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, 1.0001594642002871);  squeeze_dims_1 = None
        mul_tensor_4: "f32[1408]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 0.1);  mul_tensor_3 = None
        mul_tensor_5: "f32[1408]" = torch.ops.aten.mul.Tensor(primals_342, 0.9)
        add_tensor_2: "f32[1408]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(primals_343, -1);  primals_343 = None
        unsqueeze_default_1: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_6: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(primals_344, -1);  primals_344 = None
        unsqueeze_default_3: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[128, 1408, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_6, unsqueeze_default_3);  mul_tensor_6 = unsqueeze_default_3 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convolution_43, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 1408, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 1408, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_4: "f32[1, 1408, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
        rsqrt_default_1: "f32[1, 1408, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        sub_tensor_1: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_43, getitem_3);  convolution_43 = None
        mul_tensor_7: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        squeeze_dims_2: "f32[1408]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        mul_tensor_8: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_dims_2, 0.1);  squeeze_dims_2 = None
        mul_tensor_9: "f32[1408]" = torch.ops.aten.mul.Tensor(primals_347, 0.9)
        add_tensor_5: "f32[1408]" = torch.ops.aten.add.Tensor(mul_tensor_8, mul_tensor_9);  mul_tensor_8 = mul_tensor_9 = None
        squeeze_dims_3: "f32[1408]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_tensor_10: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, 1.0001594642002871);  squeeze_dims_3 = None
        mul_tensor_11: "f32[1408]" = torch.ops.aten.mul.Tensor(mul_tensor_10, 0.1);  mul_tensor_10 = None
        mul_tensor_12: "f32[1408]" = torch.ops.aten.mul.Tensor(primals_348, 0.9)
        add_tensor_6: "f32[1408]" = torch.ops.aten.add.Tensor(mul_tensor_11, mul_tensor_12);  mul_tensor_11 = mul_tensor_12 = None
        unsqueeze_default_4: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(primals_349, -1);  primals_349 = None
        unsqueeze_default_5: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_13: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_7, unsqueeze_default_5);  mul_tensor_7 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(primals_350, -1);  primals_350 = None
        unsqueeze_default_7: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_7: "f32[128, 1408, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_13, unsqueeze_default_7);  mul_tensor_13 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:748 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_tensor_8: "f32[128, 1408, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_3, add_tensor_7);  add_tensor_3 = add_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        relu_default: "f32[128, 1408, 7, 7]" = torch.ops.aten.relu.default(add_tensor_8);  add_tensor_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_dim: "f32[128, 1408, 1, 1]" = torch.ops.aten.mean.dim(relu_default, [-1, -2], True);  relu_default = None
        as_strided_default: "f32[128, 1408, 1, 1]" = torch.ops.aten.as_strided.default(mean_dim, [128, 1408, 1, 1], [1408, 1, 1408, 1408]);  mean_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        reshape_default: "f32[128, 1408]" = torch.ops.aten.reshape.default(as_strided_default, _shape_param_0);  as_strided_default = _shape_param_0 = None

        # No stacktrace found for following nodes
        copy__default: "f32[1408]" = torch.ops.aten.copy_.default(primals_341, add_tensor_1);  primals_341 = add_tensor_1 = None
        copy__default_1: "f32[1408]" = torch.ops.aten.copy_.default(primals_342, add_tensor_2);  primals_342 = add_tensor_2 = None
        copy__default_2: "f32[1408]" = torch.ops.aten.copy_.default(primals_347, add_tensor_5);  primals_347 = add_tensor_5 = None
        copy__default_3: "f32[1408]" = torch.ops.aten.copy_.default(primals_348, add_tensor_6);  primals_348 = add_tensor_6 = None
        return (copy__default, copy__default_1, reshape_default, copy__default_2, copy__default_3)



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
