"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Whisper_infer_000
Pattern hash: 1f499ddad801
Shape hash: 4244b4a9
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
_shapes_config = "(T([8, 384, 1500], f16), T([1500, 384], f16), T([384], f16), T([384], f16), S([12000, 384]), S([12000, 384]), S([12000, 384]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_1: "f16[8, 384, 1500]", arg5_1: "f16[1500, 384]", arg6_1: "f16[384]", arg7_1: "f16[384]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[8, 384, 1500]" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32);  convolution_1 = None
        mul_tensor: "f32[8, 384, 1500]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.5)
        mul_tensor_1: "f32[8, 384, 1500]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.7071067811865476);  convert_element_type_default = None
        erf_default: "f32[8, 384, 1500]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[8, 384, 1500]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[8, 384, 1500]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None
        convert_element_type_default_1: "f16[8, 384, 1500]" = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.float16);  mul_tensor_2 = None
        permute_default: "f16[8, 1500, 384]" = torch.ops.aten.permute.default(convert_element_type_default_1, [0, 2, 1]);  convert_element_type_default_1 = None
        iota_default: "i64[1500]" = torch.ops.prims.iota.default(1500, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        embedding_default: "f16[1500, 384]" = torch.ops.aten.embedding.default(arg5_1, iota_default);  arg5_1 = iota_default = None
        add_tensor_1: "f16[8, 1500, 384]" = torch.ops.aten.add.Tensor(permute_default, embedding_default);  permute_default = embedding_default = None
        clone_default: "f16[8, 1500, 384]" = torch.ops.aten.clone.default(add_tensor_1, memory_format = torch.contiguous_format);  add_tensor_1 = None
        convert_element_type_default_2: "f32[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(clone_default, torch.float32);  clone_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default_2, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1500, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 1500, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_default_2, getitem_1);  convert_element_type_default_2 = getitem_1 = None
        add_tensor_2: "f32[8, 1500, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[8, 1500, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor_3: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_4: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_tensor_3, arg6_1);  mul_tensor_3 = arg6_1 = None
        add_tensor_3: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(mul_tensor_4, arg7_1);  mul_tensor_4 = arg7_1 = None
        convert_element_type_default_3: "f16[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float16);  add_tensor_3 = None
        view_default: "f16[12000, 384]" = torch.ops.aten.view.default(convert_element_type_default_3, _shape_param_0);  _shape_param_0 = None
        view_default_1: "f16[12000, 384]" = torch.ops.aten.view.default(convert_element_type_default_3, _shape_param_1);  _shape_param_1 = None
        view_default_2: "f16[12000, 384]" = torch.ops.aten.view.default(convert_element_type_default_3, _shape_param_2);  convert_element_type_default_3 = _shape_param_2 = None
        return (view_default, view_default_1, view_default_2)



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
