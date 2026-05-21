"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_train
Pattern hash: 3e737b963595
Shape hash: 2420fe07
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
_shapes_config = "(T([128, 32, 112, 112], f32, stride=(401408, 1, 3584, 32)), T([128, 32, 112, 112], f32, stride=(401408, 1, 3584, 32)), T([128, 32, 112, 112], f32, stride=(401408, 1, 3584, 32)), T([1, 32, 1, 1], f32), T([32], f32), T([32], f32))"

class Repro(torch.nn.Module):
    def forward(self, relu: "f32[128, 32, 112, 112]", getitem_245: "f32[128, 32, 112, 112]", convolution: "f32[128, 32, 112, 112]", unsqueeze_438: "f32[1, 32, 1, 1]", squeeze_1: "f32[32]", primals_6: "f32[32]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:437 in forward_features, code: x = self.stem(x)
        le_scalar: "b8[128, 32, 112, 112]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[128, 32, 112, 112]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_245);  le_scalar = full_default = getitem_245 = None
        sum_dim_int_list: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_438);  convolution = unsqueeze_438 = None
        mul_tensor: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[32]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 6.228077168367346e-07);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[32]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 6.228077168367346e-07);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_tensor_4: "f32[32]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_6);  squeeze_1 = primals_6 = None
        unsqueeze_default_6: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        return mul_tensor_7



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
