"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-1-5-linux.aws.a100_graph0
Pattern hash: 21ea62bab778
Shape hash: abd438fa
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
    def forward(self, cumsum: "i64[1, 1024]", convert_element_type_1: "i32[1, 1024]", arg1_1: "bf16[50265, 768]", arg0_1: "i64[1, 1024]", arg2_1: "bf16[4098, 768]", arg3_1: "bf16[1, 768]", arg4_1: "bf16[768]", arg5_1: "bf16[768]"):
        # No stacktrace found for following nodes
        full_default: "f32[1, 1024]" = torch.ops.aten.full.default([1, 1024], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "i64[1, 1024]" = torch.ops.aten.full.default([1, 1024], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        unsqueeze_default: "f32[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(full_default, 1);  full_default = None
        unsqueeze_default_1: "f32[1, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        convert_element_type_default: "bf16[1, 1, 1, 1024]" = torch.ops.prims.convert_element_type.default(unsqueeze_default_1, torch.bfloat16);  unsqueeze_default_1 = None
        sub_tensor: "bf16[1, 1, 1, 1024]" = torch.ops.aten.sub.Tensor(1.0, convert_element_type_default);  convert_element_type_default = None
        full_default_2: "bf16[1, 1, 1, 1024]" = torch.ops.aten.full.default([1, 1, 1, 1024], -0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_int: "bf16[1, 1, 1024]" = torch.ops.aten.select.int(full_default_2, 1, 0);  full_default_2 = None
        select_int_1: "bf16[1, 1024]" = torch.ops.aten.select.int(select_int, 1, 0);  select_int = None
        convert_element_type_default_1: "i32[1, 1024]" = torch.ops.prims.convert_element_type.default(cumsum, torch.int32);  cumsum = None
        mul_tensor: "i32[1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, convert_element_type_1);  convert_element_type_default_1 = convert_element_type_1 = None
        convert_element_type_default_2: "i64[1, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.int64);  mul_tensor = None
        add_tensor: "i64[1, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_default_2, 1);  convert_element_type_default_2 = None
        embedding_default: "bf16[1, 1024, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 1);  arg1_1 = arg0_1 = None
        embedding_default_1: "bf16[1, 1024, 768]" = torch.ops.aten.embedding.default(arg2_1, add_tensor, 1);  arg2_1 = add_tensor = None
        embedding_default_2: "bf16[1, 1024, 768]" = torch.ops.aten.embedding.default(arg3_1, full_default_1);  arg3_1 = full_default_1 = None
        add_tensor_1: "bf16[1, 1024, 768]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None
        add_tensor_2: "bf16[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor_1, embedding_default_2);  add_tensor_1 = embedding_default_2 = None
        convert_element_type_default_3: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float32);  add_tensor_2 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default_3, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 1024, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 1024, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_3: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor_3);  add_tensor_3 = None
        sub_tensor_1: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_default_3, getitem_1);  convert_element_type_default_3 = getitem_1 = None
        mul_tensor_1: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default);  sub_tensor_1 = rsqrt_default = None
        mul_tensor_2: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg4_1);  mul_tensor_1 = arg4_1 = None
        add_tensor_4: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_tensor_2, arg5_1);  mul_tensor_2 = arg5_1 = None
        convert_element_type_default_4: "bf16[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.bfloat16);  add_tensor_4 = None
        return (sub_tensor, select_int_1, convert_element_type_default_4)


def _default_make_inputs():
    return [
    torch.randint(0, 1024, [1, 1024], dtype=torch.int64, device='cuda'),
    torch.randint(0, 1024, [1, 1024], dtype=torch.int32, device='cuda'),
    torch.randn([50265, 768], dtype=torch.bfloat16, device='cuda'),
    torch.randint(0, 50265, [1, 1024], dtype=torch.int64, device='cuda'),
    torch.randn([4098, 768], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 768], dtype=torch.bfloat16, device='cuda'),
    torch.randn([768], dtype=torch.bfloat16, device='cuda'),
    torch.randn([768], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
