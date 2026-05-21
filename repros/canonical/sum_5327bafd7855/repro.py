"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Bart_train
Pattern hash: 5327bafd7855
Shape hash: fa913322
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
_shapes_config = "(T([48, 512, 512], f32), T([4, 12, 512, 512], b8), T([4, 12, 512, 512], f32), S([4, 12, 512, 512]), S([48, 512, 512]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_3: "f32[48, 512, 512]", gt_1: "b8[4, 12, 512, 512]", where_2: "f32[4, 12, 512, 512]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_3, _shape_param_0);  bmm_3 = _shape_param_0 = None
        convert_element_type_default: "f32[4, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_tensor: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  reshape_default = mul_tensor = None
        mul_tensor_2: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, where_2);  mul_tensor_1 = None
        sum_dim_int_list: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
        neg_default: "f32[4, 12, 512, 512]" = torch.ops.aten.neg.default(where_2);  where_2 = None
        fma_default: "f32[4, 12, 512, 512]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_2);  neg_default = sum_dim_int_list = mul_tensor_2 = None
        reshape_default_1: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(fma_default, _shape_param_1);  fma_default = _shape_param_1 = None
        return reshape_default_1



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
