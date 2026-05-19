"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf-1-6-linux.aws.a100_graph22
Pattern hash: 74b9c51e3fc4
Shape hash: 01f0243a
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
    def forward(self, arg1_1: "bf16[20005, 768]", arg0_1: "i64[4, 128]", arg2_1: "bf16[1, 512, 768]", arg3_1: "bf16[3, 768]", arg4_1: "i64[4, 128]", arg5_1: "bf16[768]", arg6_1: "bf16[768]", arg7_1: "bf16[768, 768]", arg9_1: "bf16[768, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        embedding_default: "bf16[4, 128, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 0);  arg1_1 = arg0_1 = None
        slice_tensor: "bf16[1, 128, 768]" = torch.ops.aten.slice.Tensor(arg2_1, 1, 0, 128);  arg2_1 = None
        add_tensor: "bf16[4, 128, 768]" = torch.ops.aten.add.Tensor(embedding_default, slice_tensor);  embedding_default = slice_tensor = None
        embedding_default_1: "bf16[4, 128, 768]" = torch.ops.aten.embedding.default(arg3_1, arg4_1, 0);  arg3_1 = arg4_1 = None
        add_tensor_1: "bf16[4, 128, 768]" = torch.ops.aten.add.Tensor(add_tensor, embedding_default_1);  add_tensor = embedding_default_1 = None
        mean_dim: "bf16[4, 128, 1]" = torch.ops.aten.mean.dim(add_tensor_1, [-1], True)
        convert_element_type_default: "f32[4, 128, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32)
        var_correction: "f32[4, 128, 1]" = torch.ops.aten.var.correction(convert_element_type_default, [-1], correction = 1.0, keepdim = True);  convert_element_type_default = None
        sqrt_default: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_correction);  var_correction = None
        convert_element_type_default_1: "bf16[4, 128, 1]" = torch.ops.prims.convert_element_type.default(sqrt_default, torch.bfloat16);  sqrt_default = None
        sub_tensor: "bf16[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_tensor_1, mean_dim);  add_tensor_1 = mean_dim = None
        mul_tensor: "bf16[4, 128, 768]" = torch.ops.aten.mul.Tensor(arg5_1, sub_tensor);  arg5_1 = sub_tensor = None
        add_tensor_2: "bf16[4, 128, 1]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1e-06);  convert_element_type_default_1 = None
        div_tensor: "bf16[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_tensor, add_tensor_2);  mul_tensor = add_tensor_2 = None
        add_tensor_3: "bf16[4, 128, 768]" = torch.ops.aten.add.Tensor(div_tensor, arg6_1);  div_tensor = arg6_1 = None
        view_default: "bf16[512, 768]" = torch.ops.aten.view.default(add_tensor_3, _shape_param_0);  _shape_param_0 = None
        permute_default: "bf16[768, 768]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        view_default_1: "bf16[512, 768]" = torch.ops.aten.view.default(add_tensor_3, _shape_param_1);  add_tensor_3 = _shape_param_1 = None
        permute_default_1: "bf16[768, 768]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        return (view_default, permute_default, view_default_1, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([20005, 768], dtype=torch.bfloat16, device='cuda'),
    torch.randint(0, 20005, [4, 128], dtype=torch.int64, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.bfloat16, device='cuda'),
    torch.randn([3, 768], dtype=torch.bfloat16, device='cuda'),
    torch.randint(0, 3, [4, 128], dtype=torch.int64, device='cuda'),
    torch.randn([768], dtype=torch.bfloat16, device='cuda'),
    torch.randn([768], dtype=torch.bfloat16, device='cuda'),
    torch.randn([768, 768], dtype=torch.bfloat16, device='cuda'),
    torch.randn([768, 768], dtype=torch.bfloat16, device='cuda'),
    [512, 768],  # _shape_param_0
    [512, 768],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
