"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Longformer_infer_000
Pattern hash: 24db25e30504
Shape hash: 43e317b2
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
_shapes_config = "(T([50265, 768], f16), T([1, 4096], i64, gen=Index(50265)), T([1, 4096], i64, gen=Index(4097)), T([1, 4096], i32, gen=Index(2)), T([4098, 768], f16), T([1, 768], f16), T([768], f16), T([768], f16))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f16[50265, 768]", arg0_1: "i64[1, 4096]", cumsum: "i64[1, 4096]", convert_element_type_1: "i32[1, 4096]", arg2_1: "f16[4098, 768]", arg3_1: "f16[1, 768]", arg4_1: "f16[768]", arg5_1: "f16[768]"):
        # No stacktrace found for following nodes
        embedding_default: "f16[1, 4096, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 1);  arg1_1 = arg0_1 = None
        convert_element_type_default: "i32[1, 4096]" = torch.ops.prims.convert_element_type.default(cumsum, torch.int32);  cumsum = None
        mul_tensor: "i32[1, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_default, convert_element_type_1);  convert_element_type_default = convert_element_type_1 = None
        convert_element_type_default_1: "i64[1, 4096]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.int64);  mul_tensor = None
        add_tensor: "i64[1, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1);  convert_element_type_default_1 = None
        embedding_default_1: "f16[1, 4096, 768]" = torch.ops.aten.embedding.default(arg2_1, add_tensor, 1);  arg2_1 = add_tensor = None
        add_tensor_1: "f16[1, 4096, 768]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None
        full_default: "i64[1, 4096]" = torch.ops.aten.full.default([1, 4096], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        embedding_default_2: "f16[1, 4096, 768]" = torch.ops.aten.embedding.default(arg3_1, full_default);  arg3_1 = full_default = None
        add_tensor_2: "f16[1, 4096, 768]" = torch.ops.aten.add.Tensor(add_tensor_1, embedding_default_2);  add_tensor_1 = embedding_default_2 = None
        convert_element_type_default_2: "f32[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float32);  add_tensor_2 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default_2, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 4096, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 4096, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[1, 4096, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_default_2, getitem_1);  convert_element_type_default_2 = getitem_1 = None
        add_tensor_3: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_tensor_3);  add_tensor_3 = None
        mul_tensor_1: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_2: "f32[1, 4096, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg4_1);  mul_tensor_1 = arg4_1 = None
        add_tensor_4: "f32[1, 4096, 768]" = torch.ops.aten.add.Tensor(mul_tensor_2, arg5_1);  mul_tensor_2 = arg5_1 = None
        convert_element_type_default_3: "f16[1, 4096, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.float16);  add_tensor_4 = None
        return convert_element_type_default_3



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
