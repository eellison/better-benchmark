"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_GPT2_train
Pattern hash: f0af8f55001e
Shape hash: 862b5290
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
_shapes_config = "(T([2048, 768], f32), T([768], f32), T([4, 512, 768], f32), T([4, 512, 1], f32), T([4, 512, 768], f32), T([4, 512, 768], b8), S([4, 512, 768]), S([2048, 768]))"

class Repro(torch.nn.Module):
    def forward(self, mm_93: "f32[2048, 768]", primals_10: "f32[768]", mul_6: "f32[4, 512, 768]", div_23: "f32[4, 512, 1]", add_146: "f32[4, 512, 768]", gt_1: "b8[4, 512, 768]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        reshape_default: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_93, _shape_param_0);  mm_93 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(reshape_default, primals_10);  reshape_default = primals_10 = None
        mul_tensor_1: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_6);  mul_tensor = None
        sum_dim_int_list_1: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_6, sum_dim_int_list_1);  mul_6 = sum_dim_int_list_1 = None
        sub_tensor: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_23, sub_tensor_1);  div_23 = sub_tensor_1 = None
        add_tensor: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_146, mul_tensor_4);  add_146 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_default: "f32[4, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_tensor_5: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_6: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor, mul_tensor_5);  add_tensor = mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        reshape_default_1: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_tensor_6, _shape_param_1);  mul_tensor_6 = _shape_param_1 = None
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
