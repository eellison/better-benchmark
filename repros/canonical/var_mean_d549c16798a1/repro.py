"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train_000
Pattern hash: d549c16798a1
Shape hash: 49fd03a8
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([25088, 512], f32), T([46], i64), T([128, 14, 14, 512], f32), T([512], f32), T([512], f32), S([512, 49, 512]), S([-1, 7, 7, 512]), S([-1, 2, 2, 7, 7, 512]), S([-1, 14, 14, 512]), S([128, -1, 512]), S([25088, 512]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_81: "f32[25088, 512]", inductor_seeds: "i64[46]", view_545: "f32[128, 14, 14, 512]", arg309_1: "f32[512]", arg310_1: "f32[512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        view_default: "f32[512, 49, 512]" = torch.ops.aten.view.default(addmm_81, _shape_param_0);  addmm_81 = _shape_param_0 = None
        view_default_1: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        view_default_2: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_default_1, _shape_param_2);  view_default_1 = _shape_param_2 = None
        permute_default: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_default_2, [0, 1, 3, 2, 4, 5]);  view_default_2 = None
        clone_default: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_3: "f32[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 38);  inductor_seeds = None
        inductor_random_default: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1, 1], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        lt_scalar: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default, 0.9130434766411781);  inductor_random_default = None
        convert_element_type_default: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_scalar, torch.float32);  lt_scalar = None
        div_tensor: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.9130434766411781);  convert_element_type_default = None
        mul_tensor: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_default_3, div_tensor);  view_default_3 = div_tensor = None
        add_tensor: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_545, mul_tensor);  view_545 = mul_tensor = None
        view_default_4: "f32[128, 196, 512]" = torch.ops.aten.view.default(add_tensor, _shape_param_4);  add_tensor = _shape_param_4 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(view_default_4, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 196, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 196, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(view_default_4, getitem_1);  view_default_4 = getitem_1 = None
        mul_tensor_1: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_2: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg309_1);  mul_tensor_1 = arg309_1 = None
        add_tensor_2: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_tensor_2, arg310_1);  mul_tensor_2 = arg310_1 = None
        view_default_5: "f32[25088, 512]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_5);  add_tensor_2 = _shape_param_5 = None
        div_tensor_1: "f32[128, 196, 1]" = torch.ops.aten.div.Tensor(rsqrt_default, 512);  rsqrt_default = None
        return (view_default_5, div_tensor_1)

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
