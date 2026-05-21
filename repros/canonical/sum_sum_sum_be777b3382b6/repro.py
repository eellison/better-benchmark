"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train_001
Pattern hash: be777b3382b6
Shape hash: 44226644
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
_shapes_config = "(T([401408, 128], f32), T([128], f32), T([128, 3136, 128], f32), T([128, 3136, 1], f32), T([128, 3136, 128], f32), T([128, 1, 1, 1], b8), S([128, 3136, 128]), S([128, 56, 56, 128]), S([128, 8, 7, 8, 7, 128]), S([8192, 7, 7, 128]), S([8192, 49, 128]), S([401408, 128]), S([128]))"

class Repro(torch.nn.Module):
    def forward(self, mm_186: "f32[401408, 128]", arg15_1: "f32[128]", arg198_1: "f32[128, 3136, 128]", arg550_1: "f32[128, 3136, 1]", view_742: "f32[128, 3136, 128]", arg197_1: "b8[128, 1, 1, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        view_default: "f32[128, 3136, 128]" = torch.ops.aten.view.default(mm_186, _shape_param_0);  mm_186 = _shape_param_0 = None
        mul_tensor: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(view_default, arg15_1);  arg15_1 = None
        mul_tensor_1: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, 128)
        sum_dim_int_list: "f32[128, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, arg198_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(arg198_1, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[128, 3136, 128]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[128, 3136, 128]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(arg550_1, sub_tensor_1);  arg550_1 = sub_tensor_1 = None
        mul_tensor_5: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(view_default, arg198_1);  arg198_1 = None
        sum_dim_int_list_2: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_3: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_default, [0, 1]);  view_default = None
        add_tensor: "f32[128, 3136, 128]" = torch.ops.aten.add.Tensor(view_742, mul_tensor_4);  view_742 = mul_tensor_4 = None
        view_default_1: "f32[128, 56, 56, 128]" = torch.ops.aten.view.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None
        convert_element_type_default: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(arg197_1, torch.float32);  arg197_1 = None
        div_tensor: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.9956521736457944);  convert_element_type_default = None
        mul_tensor_6: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(view_default_1, div_tensor);  view_default_1 = div_tensor = None
        iota_default: "i64[56]" = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_1: "i64[56]" = torch.ops.aten.add.Tensor(iota_default, 3);  iota_default = None
        fmod_scalar: "i64[56]" = torch.ops.aten.fmod.Scalar(add_tensor_1, 56);  add_tensor_1 = None
        index_tensor: "f32[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(mul_tensor_6, [None, None, fmod_scalar]);  mul_tensor_6 = None
        index_tensor_1: "f32[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(index_tensor, [None, fmod_scalar]);  index_tensor = fmod_scalar = None
        view_default_2: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.view.default(index_tensor_1, _shape_param_2);  index_tensor_1 = _shape_param_2 = None
        permute_default: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.permute.default(view_default_2, [0, 1, 3, 2, 4, 5]);  view_default_2 = None
        clone_default: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_3: "f32[8192, 7, 7, 128]" = torch.ops.aten.view.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        view_default_4: "f32[8192, 49, 128]" = torch.ops.aten.view.default(view_default_3, _shape_param_4);  view_default_3 = _shape_param_4 = None
        view_default_5: "f32[401408, 128]" = torch.ops.aten.view.default(view_default_4, _shape_param_5);  view_default_4 = _shape_param_5 = None
        permute_default_1: "f32[128, 401408]" = torch.ops.aten.permute.default(view_default_5, [1, 0])
        sum_dim_int_list_4: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_default_5, [0], True);  view_default_5 = None
        view_default_6: "f32[128]" = torch.ops.aten.view.default(sum_dim_int_list_4, _shape_param_6);  sum_dim_int_list_4 = _shape_param_6 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, permute_default_1, view_default_6)



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
