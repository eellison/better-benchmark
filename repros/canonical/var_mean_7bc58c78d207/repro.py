"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf_cuda_h100-1-5-linux.aws.h100_graph33
Pattern hash: 7bc58c78d207
Shape hash: 404a9680
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_65: "f16[8192, 768]", inductor_seeds: "i64[37]", add_88: "f32[16, 512, 768]", arg182_1: "f32[768]", arg183_1: "f32[768]", arg185_1: "f32[768]", arg184_1: "f32[768, 768]", arg187_1: "f32[768]", arg186_1: "f32[768, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f16[16, 512, 768]" = torch.ops.aten.view.default(addmm_65, _shape_param_0);  addmm_65 = _shape_param_0 = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 33);  inductor_seeds = None
        inductor_random_default: "f32[16, 512, 768]" = torch.ops.prims.inductor_random.default([16, 512, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        convert_element_type_default: "f16[16, 512, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.float16);  inductor_random_default = None
        gt_scalar: "b8[16, 512, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_default, 0.1);  convert_element_type_default = None
        mul_tensor: "f16[16, 512, 768]" = torch.ops.aten.mul.Tensor(gt_scalar, view_default);  gt_scalar = view_default = None
        mul_tensor_1: "f16[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        add_tensor: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, add_88);  mul_tensor_1 = add_88 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[16, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor_2: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_3: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, arg182_1);  mul_tensor_2 = arg182_1 = None
        add_tensor_2: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_3, arg183_1);  mul_tensor_3 = arg183_1 = None
        convert_element_type_default_1: "f16[768]" = torch.ops.prims.convert_element_type.default(arg185_1, torch.float16);  arg185_1 = None
        convert_element_type_default_2: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(arg184_1, torch.float16);  arg184_1 = None
        convert_element_type_default_3: "f16[16, 512, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float16);  add_tensor_2 = None
        view_default_1: "f16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_default_3, _shape_param_1);  convert_element_type_default_3 = _shape_param_1 = None
        permute_default: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_default_2, [1, 0]);  convert_element_type_default_2 = None
        convert_element_type_default_4: "f16[768]" = torch.ops.prims.convert_element_type.default(arg187_1, torch.float16);  arg187_1 = None
        convert_element_type_default_5: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(arg186_1, torch.float16);  arg186_1 = None
        permute_default_1: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_default_5, [1, 0]);  convert_element_type_default_5 = None
        return (convert_element_type_default_1, view_default_1, permute_default, convert_element_type_default_4, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([8192, 768], dtype=torch.float16, device='cuda'),
    torch.randint(0, 37, [37], dtype=torch.int64, device='cuda'),
    torch.randn([16, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [16, 512, 768],  # _shape_param_0
    [8192, 768],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
