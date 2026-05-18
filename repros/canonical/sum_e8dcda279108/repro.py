"""
Standalone repro captured via capture_hook.
Label: genai_softmax_bwd_32768x256
Pattern hash: e8dcda279108
Shape hash: 6c05d37f
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "bf16[]", _shape_param_0, primals_1: "bf16[32768, 256]", amax: "f32[32768, 1]", sum_1: "f32[32768, 1]"):
        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:243 in sm_bwd, code: return y.sum()
        expand_default: "bf16[32768, 256]" = torch.ops.aten.expand.default(tangents_1, _shape_param_0);  tangents_1 = _shape_param_0 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:242 in sm_bwd, code: y = F.softmax(x, dim=-1)
        convert_element_type_default: "f32[32768, 256]" = torch.ops.prims.convert_element_type.default(expand_default, torch.float32);  expand_default = None
        convert_element_type_default_1: "f32[32768, 256]" = torch.ops.prims.convert_element_type.default(primals_1, torch.float32);  primals_1 = None
        sub_tensor: "f32[32768, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_default_1, amax);  convert_element_type_default_1 = amax = None
        exp_default: "f32[32768, 256]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        div_tensor: "f32[32768, 256]" = torch.ops.aten.div.Tensor(exp_default, sum_1);  exp_default = sum_1 = None
        convert_element_type_default_2: "bf16[32768, 256]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        convert_element_type_default_3: "f32[32768, 256]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_2, torch.float32);  convert_element_type_default_2 = None
        mul_tensor: "f32[32768, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_default, convert_element_type_default_3);  convert_element_type_default = None
        sum_dim_int_list: "f32[32768, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True)
        neg_default: "f32[32768, 256]" = torch.ops.aten.neg.default(convert_element_type_default_3);  convert_element_type_default_3 = None
        fma_default: "f32[32768, 256]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None
        convert_element_type_default_4: "bf16[32768, 256]" = torch.ops.prims.convert_element_type.default(fma_default, torch.bfloat16);  fma_default = None
        return convert_element_type_default_4


def _default_make_inputs():
    return [
    torch.randn([], dtype=torch.bfloat16, device='cuda'),
    [32768, 256],  # _shape_param_0
    torch.randn([32768, 256], dtype=torch.bfloat16, device='cuda'),
    torch.randn([32768, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32768, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
