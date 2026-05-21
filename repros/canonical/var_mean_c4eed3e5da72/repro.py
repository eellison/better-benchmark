"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_DistilBert_infer_000
Pattern hash: c4eed3e5da72
Shape hash: fb69fe11
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
_shapes_config = "(T([30522, 768], f16), T([1, 512], i64, gen=Index(30522)), T([512, 768], f16), T([1, 512], i64, gen=Index(512)), T([768], f16), T([768], f16), S([512, 768]), S([512, 768]), S([512, 768]))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f16[30522, 768]", arg0_1: "i64[1, 512]", arg3_1: "f16[512, 768]", arg2_1: "i64[1, 512]", arg4_1: "f16[768]", arg5_1: "f16[768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        embedding_default: "f16[1, 512, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 0);  arg1_1 = arg0_1 = None
        embedding_default_1: "f16[1, 512, 768]" = torch.ops.aten.embedding.default(arg3_1, arg2_1);  arg3_1 = arg2_1 = None
        add_tensor: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None
        convert_element_type_default: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1);  convert_element_type_default = getitem_1 = None
        add_tensor_1: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg4_1);  mul_tensor = arg4_1 = None
        add_tensor_2: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg5_1);  mul_tensor_1 = arg5_1 = None
        convert_element_type_default_1: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float16);  add_tensor_2 = None
        view_default: "f16[512, 768]" = torch.ops.aten.view.default(convert_element_type_default_1, _shape_param_0);  _shape_param_0 = None
        view_default_1: "f16[512, 768]" = torch.ops.aten.view.default(convert_element_type_default_1, _shape_param_1);  _shape_param_1 = None
        view_default_2: "f16[512, 768]" = torch.ops.aten.view.default(convert_element_type_default_1, _shape_param_2);  convert_element_type_default_1 = _shape_param_2 = None
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
