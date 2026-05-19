"""
Standalone repro captured via capture_hook.
Label: genai_layernorm_bwd_32768x256
Pattern hash: 5bff1ad7f52a
Shape hash: 044f6574
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
    def forward(self, tangents_1: "bf16[]", primals_2: "f32[256]", primals_1: "bf16[32768, 256]", getitem_1: "f32[32768, 1]", rsqrt: "f32[32768, 1]", _shape_param_0):
        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:361 in layernorm_bwd, code: return out.sum()
        expand_default: "bf16[32768, 256]" = torch.ops.aten.expand.default(tangents_1, _shape_param_0);  tangents_1 = _shape_param_0 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:360 in layernorm_bwd, code: out = F.layer_norm(x_f32, w.shape, w, None, 1e-6).to(x.dtype)
        convert_element_type_default: "f32[32768, 256]" = torch.ops.prims.convert_element_type.default(expand_default, torch.float32);  expand_default = None
        mul_tensor: "f32[32768, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_default, primals_2);  primals_2 = None
        mul_tensor_1: "f32[32768, 256]" = torch.ops.aten.mul.Tensor(mul_tensor, 256)
        sum_dim_int_list: "f32[32768, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True)

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:359 in layernorm_bwd, code: x_f32 = x.float()
        convert_element_type_default_1: "f32[32768, 256]" = torch.ops.prims.convert_element_type.default(primals_1, torch.float32);  primals_1 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:360 in layernorm_bwd, code: out = F.layer_norm(x_f32, w.shape, w, None, 1e-6).to(x.dtype)
        sub_tensor: "f32[32768, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_default_1, getitem_1);  convert_element_type_default_1 = getitem_1 = None
        mul_tensor_2: "f32[32768, 256]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        mul_tensor_3: "f32[32768, 256]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1: "f32[32768, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [1], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[32768, 256]" = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1: "f32[32768, 256]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[32768, 256]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor: "f32[32768, 1]" = torch.ops.aten.div.Tensor(rsqrt, 256);  rsqrt = None
        mul_tensor_5: "f32[32768, 256]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_6: "f32[32768, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor_2);  convert_element_type_default = mul_tensor_2 = None
        sum_dim_int_list_2: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0]);  mul_tensor_6 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:359 in layernorm_bwd, code: x_f32 = x.float()
        convert_element_type_default_2: "bf16[32768, 256]" = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.bfloat16);  mul_tensor_5 = None
        return (sum_dim_int_list_2, convert_element_type_default_2)


def _default_make_inputs():
    return [
    torch.randn([], dtype=torch.bfloat16, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([32768, 256], dtype=torch.bfloat16, device='cuda'),
    torch.randn([32768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32768, 1], dtype=torch.float32, device='cuda'),
    [32768, 256],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
