"""
Standalone repro captured via capture_hook.
Label: hf_M2M100ForConditionalGeneration_infer_000
Pattern hash: e1bb5773a9cf
Shape hash: 32dff7c6
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
_shapes_config = "(T([128112, 1024], f32), T([64, 128], i64, gen=Index(128112)), T([64, 128], i64), T([64, 128], i32), T([1026, 1024], f32), T([1024], f32), T([1024], f32), S([64, 128, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, arg2_1: "f32[128112, 1024]", arg1_1: "i64[64, 128]", cumsum_1: "i64[64, 128]", convert_element_type_3: "i32[64, 128]", arg198_1: "f32[1026, 1024]", arg199_1: "f32[1024]", arg200_1: "f32[1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        embedding_default: "f32[64, 128, 1024]" = torch.ops.aten.embedding.default(arg2_1, arg1_1, 1);  arg2_1 = arg1_1 = None
        mul_tensor: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(embedding_default, 32.0);  embedding_default = None
        convert_element_type_default: "i32[64, 128]" = torch.ops.prims.convert_element_type.default(cumsum_1, torch.int32);  cumsum_1 = None
        add_tensor: "i32[64, 128]" = torch.ops.aten.add.Tensor(convert_element_type_default, 0);  convert_element_type_default = None
        mul_tensor_1: "i32[64, 128]" = torch.ops.aten.mul.Tensor(add_tensor, convert_element_type_3);  add_tensor = convert_element_type_3 = None
        convert_element_type_default_1: "i64[64, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.int64);  mul_tensor_1 = None
        add_tensor_1: "i64[64, 128]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1);  convert_element_type_default_1 = None
        view_default: "i64[8192]" = torch.ops.aten.view.default(add_tensor_1, [-1]);  add_tensor_1 = None
        index_tensor: "f32[8192, 1024]" = torch.ops.aten.index.Tensor(arg198_1, [view_default]);  arg198_1 = view_default = None
        view_default_1: "f32[64, 128, 1024]" = torch.ops.aten.view.default(index_tensor, _shape_param_0);  index_tensor = _shape_param_0 = None
        add_tensor_2: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(mul_tensor, view_default_1);  mul_tensor = view_default_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_2, [2], correction = 0, keepdim = True)
        getitem: "f32[64, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[64, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(add_tensor_2, getitem_1);  add_tensor_2 = getitem_1 = None
        add_tensor_3: "f32[64, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[64, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_3);  add_tensor_3 = None
        mul_tensor_2: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_3: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_2, arg199_1);  mul_tensor_2 = arg199_1 = None
        add_tensor_4: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_3, arg200_1);  mul_tensor_3 = arg200_1 = None
        view_default_2: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_4, _shape_param_1);  _shape_param_1 = None
        view_default_3: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_4, _shape_param_2);  _shape_param_2 = None
        view_default_4: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_4, _shape_param_3);  add_tensor_4 = _shape_param_3 = None
        return (view_default_3, view_default_4, view_default_2)



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
