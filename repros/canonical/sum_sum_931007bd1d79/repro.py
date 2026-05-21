"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train_001
Pattern hash: 931007bd1d79
Shape hash: 2307b353
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
_shapes_config = "(T([401408, 128], f32), T([56], i64, gen=Index(56)), T([128], f32), T([128, 56, 56, 128], f32), T([128, 56, 56, 1], f32), T([128, 56, 56, 128], f32), S([8192, 49, 128]), S([8192, 7, 7, 128]), S([128, 8, 8, 7, 7, 128]), S([128, 56, 56, 128]), S([128, 3136, 128]), S([401408, 128]))"

class Repro(torch.nn.Module):
    def forward(self, mm_190: "f32[401408, 128]", arg196_1: "i64[56]", arg11_1: "f32[128]", arg192_1: "f32[128, 56, 56, 128]", arg554_1: "f32[128, 56, 56, 1]", view_750: "f32[128, 56, 56, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        view_default: "f32[8192, 49, 128]" = torch.ops.aten.view.default(mm_190, _shape_param_0);  mm_190 = _shape_param_0 = None
        view_default_1: "f32[8192, 7, 7, 128]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        view_default_2: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.view.default(view_default_1, _shape_param_2);  view_default_1 = _shape_param_2 = None
        permute_default: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.permute.default(view_default_2, [0, 1, 3, 2, 4, 5]);  view_default_2 = None
        clone_default: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_3: "f32[128, 56, 56, 128]" = torch.ops.aten.view.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        index_tensor: "f32[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(view_default_3, [None, None, arg196_1]);  view_default_3 = None
        index_tensor_1: "f32[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(index_tensor, [None, arg196_1]);  index_tensor = arg196_1 = None
        mul_tensor: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(index_tensor_1, arg11_1);  index_tensor_1 = arg11_1 = None
        mul_tensor_1: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, 128)
        sum_dim_int_list: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)
        mul_tensor_2: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, arg192_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [3], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(arg192_1, sum_dim_int_list_1);  arg192_1 = sum_dim_int_list_1 = None
        sub_tensor: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(arg554_1, sub_tensor_1);  arg554_1 = sub_tensor_1 = None
        add_tensor: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(view_750, mul_tensor_4);  view_750 = mul_tensor_4 = None
        view_default_4: "f32[128, 3136, 128]" = torch.ops.aten.view.default(add_tensor, _shape_param_4);  add_tensor = _shape_param_4 = None
        view_default_5: "f32[401408, 128]" = torch.ops.aten.view.default(view_default_4, _shape_param_5);  view_default_4 = _shape_param_5 = None
        return view_default_5



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
