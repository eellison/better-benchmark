"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf_cuda_h100-1-5-linux.aws.h100_graph33
Pattern hash: dcf1d294b09f
Shape hash: bc2d61fe
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([1, 512], i64, max=2), T([1, 512], i64, max=512), T([30522, 768], f32), T([16, 512], i64, max=30522), T([2, 768], f32), T([512, 768], f32), T([768], f32), T([768], f32), T([768], f32), T([768, 768], f32), T([768], f32), T([768, 768], f32), S([16, 512]), S([8192, 768]))"

class Repro(torch.nn.Module):
    def forward(self, arg2_1: "i64[1, 512]", arg1_1: "i64[1, 512]", arg3_1: "f32[30522, 768]", arg0_1: "i64[16, 512]", arg4_1: "f32[2, 768]", arg5_1: "f32[512, 768]", arg6_1: "f32[768]", arg7_1: "f32[768]", arg9_1: "f32[768]", arg8_1: "f32[768, 768]", arg11_1: "f32[768]", arg10_1: "f32[768, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        expand_default: "i64[1, 512]" = torch.ops.aten.expand.default(arg2_1, [1, -1]);  arg2_1 = None
        gather_default: "i64[1, 512]" = torch.ops.aten.gather.default(expand_default, 1, arg1_1);  expand_default = None
        expand_default_1: "i64[16, 512]" = torch.ops.aten.expand.default(gather_default, _shape_param_0);  gather_default = _shape_param_0 = None
        embedding_default: "f32[16, 512, 768]" = torch.ops.aten.embedding.default(arg3_1, arg0_1, 0);  arg3_1 = arg0_1 = None
        embedding_default_1: "f32[16, 512, 768]" = torch.ops.aten.embedding.default(arg4_1, expand_default_1);  arg4_1 = expand_default_1 = None
        add_tensor: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None
        embedding_default_2: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg5_1, arg1_1);  arg5_1 = arg1_1 = None
        add_tensor_1: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor, embedding_default_2);  add_tensor = embedding_default_2 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[16, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_2: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_tensor_1, getitem_1);  add_tensor_1 = getitem_1 = None
        mul_tensor: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg6_1);  mul_tensor = arg6_1 = None
        add_tensor_3: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg7_1);  mul_tensor_1 = arg7_1 = None
        inductor_seeds_default: "i64[37]" = torch.ops.prims.inductor_seeds.default(37, device(type='cuda', index=0))
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[16, 512, 768]" = torch.ops.prims.inductor_random.default([16, 512, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[16, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor_2: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(gt_scalar, add_tensor_3);  gt_scalar = add_tensor_3 = None
        mul_tensor_3: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.1111111111111112);  mul_tensor_2 = None
        convert_element_type_default: "f16[768]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.float16);  arg9_1 = None
        convert_element_type_default_1: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.float16);  arg8_1 = None
        convert_element_type_default_2: "f16[16, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.float16);  mul_tensor_3 = None
        view_default: "f16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_default_2, _shape_param_1);  convert_element_type_default_2 = _shape_param_1 = None
        permute_default: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        convert_element_type_default_3: "f16[768]" = torch.ops.prims.convert_element_type.default(arg11_1, torch.float16);  arg11_1 = None
        convert_element_type_default_4: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(arg10_1, torch.float16);  arg10_1 = None
        permute_default_1: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0]);  convert_element_type_default_4 = None
        return (convert_element_type_default, view_default, permute_default, convert_element_type_default_3, permute_default_1)


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
