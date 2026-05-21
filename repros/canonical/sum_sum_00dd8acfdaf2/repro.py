"""
Standalone repro captured via capture_hook.
Label: torchbench_phlippe_densenet_train
Pattern hash: 00dd8acfdaf2
Shape hash: fa1ddbd2
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
_shapes_config = "(T([128, 64, 32, 32], f32), T([128, 48, 32, 32], f32), T([], f32), T([128, 48, 32, 32], f32), T([128, 48, 32, 32], f32), T([1, 48, 1, 1], f32), T([48], f32), T([48], f32))"

class Repro(torch.nn.Module):
    def forward(self, add_281: "f32[128, 64, 32, 32]", relu_2: "f32[128, 48, 32, 32]", full_default: "f32[]", getitem_248: "f32[128, 48, 32, 32]", cat: "f32[128, 48, 32, 32]", unsqueeze_798: "f32[1, 48, 1, 1]", squeeze_7: "f32[48]", primals_19: "f32[48]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:33 in forward, code: out = torch.cat([out, x], dim=1)
        slice_tensor: "f32[128, 48, 32, 32]" = torch.ops.aten.slice.Tensor(add_281, 1, 16, 64);  add_281 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:32 in forward, code: out = self.net(x)
        le_scalar: "b8[128, 48, 32, 32]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_self: "f32[128, 48, 32, 32]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_248);  le_scalar = full_default = getitem_248 = None
        sum_dim_int_list: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[128, 48, 32, 32]" = torch.ops.aten.sub.Tensor(cat, unsqueeze_798);  cat = unsqueeze_798 = None
        mul_tensor: "f32[128, 48, 32, 32]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[48]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 7.62939453125e-06);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[48]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 7.62939453125e-06);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_tensor_4: "f32[48]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_19);  squeeze_7 = primals_19 = None
        unsqueeze_default_6: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[128, 48, 32, 32]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[128, 48, 32, 32]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[128, 48, 32, 32]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[128, 48, 32, 32]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        add_tensor: "f32[128, 48, 32, 32]" = torch.ops.aten.add.Tensor(slice_tensor, mul_tensor_7);  slice_tensor = mul_tensor_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:33 in forward, code: out = torch.cat([out, x], dim=1)
        slice_tensor_1: "f32[128, 16, 32, 32]" = torch.ops.aten.slice.Tensor(add_tensor, 1, 0, 16);  add_tensor = None
        return slice_tensor_1



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
