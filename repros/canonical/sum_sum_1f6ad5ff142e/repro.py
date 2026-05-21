"""
Standalone repro captured via capture_hook.
Label: torchbench_timm_regnet_train
Pattern hash: 1f6ad5ff142e
Shape hash: f894fbbd
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
_shapes_config = "(T([32, 224, 56, 56], f32), T([32, 224, 1, 1], f32), T([32, 224, 1, 1], f32), T([32, 224, 56, 56], f32), T([], f32), T([1, 224, 1, 1], f32), T([32, 224, 56, 56], f32), T([1, 224, 1, 1], f32), T([224], f32), S([32, 224, 56, 56]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_406: "f32[32, 224, 56, 56]", sigmoid: "f32[32, 224, 1, 1]", getitem_412: "f32[32, 224, 1, 1]", relu_2: "f32[32, 224, 56, 56]", full_default: "f32[]", getitem_5: "f32[1, 224, 1, 1]", convolution_2: "f32[32, 224, 56, 56]", rsqrt_2: "f32[1, 224, 1, 1]", primals_18: "f32[224]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_tensor: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_406, sigmoid);  getitem_406 = sigmoid = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_default: "f32[32, 224, 56, 56]" = torch.ops.aten.expand.default(getitem_412, _shape_param_0);  getitem_412 = _shape_param_0 = None
        div_scalar: "f32[32, 224, 56, 56]" = torch.ops.aten.div.Scalar(expand_default, 3136);  expand_default = None
        add_tensor: "f32[32, 224, 56, 56]" = torch.ops.aten.add.Tensor(mul_tensor, div_scalar);  mul_tensor = div_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_scalar: "b8[32, 224, 56, 56]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_self: "f32[32, 224, 56, 56]" = torch.ops.aten.where.self(le_scalar, full_default, add_tensor);  le_scalar = full_default = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        unsqueeze_default: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_1: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        sum_dim_int_list: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[32, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_default_2);  convolution_2 = unsqueeze_default_2 = None
        mul_tensor_1: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 2, 3]);  mul_tensor_1 = None
        mul_tensor_2: "f32[224]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 9.964923469387754e-06);  sum_dim_int_list = None
        unsqueeze_default_3: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_tensor_2, 0);  mul_tensor_2 = None
        unsqueeze_default_4: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_3: "f32[224]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 9.964923469387754e-06);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[224]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_tensor_4: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_5: "f32[224]" = torch.ops.aten.mul.Tensor(mul_tensor_3, mul_tensor_4);  mul_tensor_3 = mul_tensor_4 = None
        unsqueeze_default_6: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_18);  squeeze_dims_1 = primals_18 = None
        unsqueeze_default_9: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_10: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 2);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 3);  unsqueeze_default_10 = None
        mul_tensor_7: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_8);  sub_tensor = unsqueeze_default_8 = None
        sub_tensor_1: "f32[32, 224, 56, 56]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_7);  where_self = mul_tensor_7 = None
        sub_tensor_2: "f32[32, 224, 56, 56]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_5);  sub_tensor_1 = unsqueeze_default_5 = None
        mul_tensor_8: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_11);  sub_tensor_2 = unsqueeze_default_11 = None
        return mul_tensor_8



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
