"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train_000
Pattern hash: 389766e91319
Shape hash: db145a13
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
_shapes_config = "(T([401408, 128], f32), T([56], i64), T([128, 56, 56, 128], f32, stride=(401408, 56, 1, 3136)), T([128], f32), T([128], f32), S([8192, 49, 128]), S([-1, 7, 7, 128]), S([-1, 8, 8, 7, 7, 128]), S([-1, 56, 56, 128]), S([128, 3136, 128]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_5: "f32[401408, 128]", iota: "i64[56]", view_24: "f32[128, 56, 56, 128]", arg28_1: "f32[128]", arg29_1: "f32[128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view_default: "f32[8192, 49, 128]" = torch.ops.aten.view.default(addmm_5, _shape_param_0);  addmm_5 = _shape_param_0 = None
        view_default_1: "f32[8192, 7, 7, 128]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        view_default_2: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.view.default(view_default_1, _shape_param_2);  view_default_1 = _shape_param_2 = None
        permute_default: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.permute.default(view_default_2, [0, 1, 3, 2, 4, 5]);  view_default_2 = None
        clone_default: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_3: "f32[128, 56, 56, 128]" = torch.ops.aten.view.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        add_tensor: "i64[56]" = torch.ops.aten.add.Tensor(iota, 53);  iota = None
        fmod_scalar: "i64[56]" = torch.ops.aten.fmod.Scalar(add_tensor, 56);  add_tensor = None
        index_tensor: "f32[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(view_default_3, [None, fmod_scalar]);  view_default_3 = None
        index_tensor_1: "f32[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(index_tensor, [None, None, fmod_scalar]);  index_tensor = fmod_scalar = None
        inductor_seeds_default: "i64[46]" = torch.ops.prims.inductor_seeds.default(46, device(type='cuda', index=0))
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        lt_scalar: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default, 0.9956521736457944);  inductor_random_default = None
        convert_element_type_default: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_scalar, torch.float32);  lt_scalar = None
        div_tensor: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.9956521736457944);  convert_element_type_default = None
        mul_tensor: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(index_tensor_1, div_tensor);  index_tensor_1 = div_tensor = None
        add_tensor_1: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(view_24, mul_tensor);  view_24 = mul_tensor = None
        view_default_4: "f32[128, 3136, 128]" = torch.ops.aten.view.default(add_tensor_1, _shape_param_4);  add_tensor_1 = _shape_param_4 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(view_default_4, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 3136, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 3136, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_2: "f32[128, 3136, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[128, 3136, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor: "f32[128, 3136, 128]" = torch.ops.aten.sub.Tensor(view_default_4, getitem_1);  view_default_4 = getitem_1 = None
        mul_tensor_1: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_2: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg28_1);  mul_tensor_1 = arg28_1 = None
        add_tensor_3: "f32[128, 3136, 128]" = torch.ops.aten.add.Tensor(mul_tensor_2, arg29_1);  mul_tensor_2 = arg29_1 = None
        clone_default_1: "f32[128, 3136, 128]" = torch.ops.aten.clone.default(add_tensor_3, memory_format = torch.contiguous_format);  add_tensor_3 = None
        _unsafe_view_default: "f32[401408, 128]" = torch.ops.aten._unsafe_view.default(clone_default_1, [401408, 128]);  clone_default_1 = None
        return _unsafe_view_default



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
