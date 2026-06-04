"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train_000
Pattern hash: 40bf68989e6d
Shape hash: 8e6163b6
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([25088, 512], f32), T([46], i64), T([128, 196, 512], f32), T([2048], f32), T([2048], f32), S([128, 196, 512]), S([128, 14, 14, 512]), S([128, 7, 2, 7, 2, 512]), S([128, 7, 7, 2048]), S([6272, 2048]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_87: "f32[25088, 512]", inductor_seeds: "i64[46]", view_594: "f32[128, 196, 512]", arg330_1: "f32[2048]", arg331_1: "f32[2048]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view_default: "f32[128, 196, 512]" = torch.ops.aten.view.default(addmm_87, _shape_param_0);  addmm_87 = _shape_param_0 = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 41);  inductor_seeds = None
        inductor_random_default: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default([128, 1, 1], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        lt_scalar: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default, 0.9086956530809402);  inductor_random_default = None
        convert_element_type_default: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_scalar, torch.float32);  lt_scalar = None
        div_tensor: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.9086956530809402);  convert_element_type_default = None
        mul_tensor: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_default, div_tensor);  view_default = div_tensor = None
        add_tensor: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_594, mul_tensor);  view_594 = mul_tensor = None
        view_default_1: "f32[128, 14, 14, 512]" = torch.ops.aten.view.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None
        view_default_2: "f32[128, 7, 2, 7, 2, 512]" = torch.ops.aten.view.default(view_default_1, _shape_param_2);  view_default_1 = _shape_param_2 = None
        permute_default: "f32[128, 7, 7, 2, 2, 512]" = torch.ops.aten.permute.default(view_default_2, [0, 1, 3, 4, 2, 5]);  view_default_2 = None
        clone_default: "f32[128, 7, 7, 2, 2, 512]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_3: "f32[128, 7, 7, 2048]" = torch.ops.aten.view.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(view_default_3, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 7, 7, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 7, 7, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[128, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[128, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[128, 7, 7, 2048]" = torch.ops.aten.sub.Tensor(view_default_3, getitem_1);  view_default_3 = getitem_1 = None
        mul_tensor_1: "f32[128, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_2: "f32[128, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg330_1);  mul_tensor_1 = arg330_1 = None
        add_tensor_2: "f32[128, 7, 7, 2048]" = torch.ops.aten.add.Tensor(mul_tensor_2, arg331_1);  mul_tensor_2 = arg331_1 = None
        view_default_4: "f32[6272, 2048]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_4);  add_tensor_2 = _shape_param_4 = None
        div_tensor_1: "f32[128, 7, 7, 1]" = torch.ops.aten.div.Tensor(rsqrt_default, 2048);  rsqrt_default = None
        return (view_default_4, div_tensor_1)

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
