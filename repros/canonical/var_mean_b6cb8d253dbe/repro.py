"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-1-5-linux.aws.a100_graph2
Pattern hash: b6cb8d253dbe
Shape hash: 716a4d50
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
    def forward(self, addmm_71: "bf16[1024, 768]", convert_element_type_151: "bf16[1, 1024, 768]", arg193_1: "bf16[768]", arg194_1: "bf16[768]", _shape_param_0):
        # No stacktrace found for following nodes
        full_default: "bf16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "bf16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "bf16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "bf16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4: "bf16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_5: "bf16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6: "bf16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_7: "bf16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_8: "bf16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "bf16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_10: "bf16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_11: "bf16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_12: "bf16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_13: "bf16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_14: "bf16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_15: "bf16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_16: "bf16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_17: "bf16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_18: "bf16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_19: "bf16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_20: "bf16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_21: "bf16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_22: "bf16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_23: "bf16[1, 1, 256, 257]" = torch.ops.aten.full.default([1, 1, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_default: "bf16[1, 1024, 768]" = torch.ops.aten.view.default(addmm_71, _shape_param_0);  addmm_71 = _shape_param_0 = None
        add_tensor: "bf16[1, 1024, 768]" = torch.ops.aten.add.Tensor(view_default, convert_element_type_151);  view_default = convert_element_type_151 = None
        convert_element_type_default: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 1024, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 1024, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1);  convert_element_type_default = getitem_1 = None
        mul_tensor: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg193_1);  mul_tensor = arg193_1 = None
        add_tensor_2: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg194_1);  mul_tensor_1 = arg194_1 = None
        convert_element_type_default_1: "bf16[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        return (full_default, full_default_1, full_default_2, full_default_3, full_default_4, full_default_5, full_default_6, full_default_7, full_default_8, full_default_9, full_default_10, full_default_11, full_default_12, full_default_13, full_default_14, full_default_15, full_default_16, full_default_17, full_default_18, full_default_19, full_default_20, full_default_21, full_default_22, full_default_23, convert_element_type_default_1)


def _default_make_inputs():
    return [
    torch.randn([1024, 768], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 1024, 768], dtype=torch.bfloat16, device='cuda'),
    torch.randn([768], dtype=torch.bfloat16, device='cuda'),
    torch.randn([768], dtype=torch.bfloat16, device='cuda'),
    [1, 1024, 768],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
