"""
Standalone repro captured via capture_hook.
Label: torchbench_timm_regnet_train
Pattern hash: 10cbbf3a30e3
Shape hash: 01720655
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
_shapes_config = "(T([32, 2240], f32), T([32, 2240, 7, 7], f32), T([1, 2240, 1, 1], f32), T([1, 2240, 1, 1], f32), T([2240], f32), T([2240], f32), T([32, 2240, 7, 7], f32), T([1, 2240, 1, 1], f32), T([1, 2240, 1, 1], f32), T([2240], f32), T([2240], f32), S([32, 2240, 1, 1]), S([32, 2240, 7, 7]))"

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[32, 2240]", convolution_98: "f32[32, 2240, 7, 7]", getitem_121: "f32[1, 2240, 1, 1]", rsqrt_60: "f32[1, 2240, 1, 1]", primals_442: "f32[2240]", primals_443: "f32[2240]", convolution_99: "f32[32, 2240, 7, 7]", getitem_123: "f32[1, 2240, 1, 1]", rsqrt_61: "f32[1, 2240, 1, 1]", primals_448: "f32[2240]", primals_449: "f32[2240]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        reshape_default: "f32[32, 2240, 1, 1]" = torch.ops.aten.reshape.default(mm, _shape_param_0);  mm = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        expand_default: "f32[32, 2240, 7, 7]" = torch.ops.aten.expand.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        div_scalar: "f32[32, 2240, 7, 7]" = torch.ops.aten.div.Scalar(expand_default, 49);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_tensor: "f32[32, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_98, getitem_121)
        mul_tensor: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_60);  sub_tensor = None
        unsqueeze_default: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_442, -1)
        unsqueeze_default_1: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_443, -1);  primals_443 = None
        unsqueeze_default_3: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[32, 2240, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        sub_tensor_1: "f32[32, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_99, getitem_123);  convolution_99 = getitem_123 = None
        mul_tensor_2: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_61);  sub_tensor_1 = rsqrt_61 = None
        unsqueeze_default_4: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_448, -1);  primals_448 = None
        unsqueeze_default_5: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_3: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_2, unsqueeze_default_5);  mul_tensor_2 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_449, -1);  primals_449 = None
        unsqueeze_default_7: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[32, 2240, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_3, unsqueeze_default_7);  mul_tensor_3 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:373 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_tensor_2: "f32[32, 2240, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor, add_tensor_1);  add_tensor = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        relu_default: "f32[32, 2240, 7, 7]" = torch.ops.aten.relu.default(add_tensor_2);  add_tensor_2 = None
        le_scalar: "b8[32, 2240, 7, 7]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[32, 2240, 7, 7]" = torch.ops.aten.where.self(le_scalar, full_default, div_scalar);  le_scalar = full_default = div_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims: "f32[2240]" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
        unsqueeze_default_8: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_9: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 2);  unsqueeze_default_8 = None
        unsqueeze_default_10: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 3);  unsqueeze_default_9 = None
        sum_dim_int_list: "f32[2240]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_2: "f32[32, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_default_10);  convolution_98 = unsqueeze_default_10 = None
        mul_tensor_4: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_2)
        sum_dim_int_list_1: "f32[2240]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 2, 3]);  mul_tensor_4 = None
        mul_tensor_5: "f32[2240]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.0006377551020408163);  sum_dim_int_list = None
        unsqueeze_default_11: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_12: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 2);  unsqueeze_default_11 = None
        unsqueeze_default_13: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 3);  unsqueeze_default_12 = None
        mul_tensor_6: "f32[2240]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.0006377551020408163);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[2240]" = torch.ops.aten.squeeze.dims(rsqrt_60, [0, 2, 3]);  rsqrt_60 = None
        mul_tensor_7: "f32[2240]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_8: "f32[2240]" = torch.ops.aten.mul.Tensor(mul_tensor_6, mul_tensor_7);  mul_tensor_6 = mul_tensor_7 = None
        unsqueeze_default_14: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(mul_tensor_8, 0);  mul_tensor_8 = None
        unsqueeze_default_15: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 2);  unsqueeze_default_14 = None
        unsqueeze_default_16: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_15, 3);  unsqueeze_default_15 = None
        mul_tensor_9: "f32[2240]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_442);  squeeze_dims_1 = primals_442 = None
        unsqueeze_default_17: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(mul_tensor_9, 0);  mul_tensor_9 = None
        unsqueeze_default_18: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_17, 2);  unsqueeze_default_17 = None
        unsqueeze_default_19: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, 3);  unsqueeze_default_18 = None
        mul_tensor_10: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_16);  sub_tensor_2 = unsqueeze_default_16 = None
        sub_tensor_3: "f32[32, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_10);  where_self = mul_tensor_10 = None
        sub_tensor_4: "f32[32, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(sub_tensor_3, unsqueeze_default_13);  sub_tensor_3 = unsqueeze_default_13 = None
        mul_tensor_11: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_4, unsqueeze_default_19);  sub_tensor_4 = unsqueeze_default_19 = None
        return mul_tensor_11



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
