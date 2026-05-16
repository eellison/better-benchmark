"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['1', '8192'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_351: "f32[4096, 8192]", addmm_1: "f32[4096, 8192]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:421 in forward, code: hidden_states = self.c_proj(hidden_states)
        reshape_default: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(mm_351, [32, 128, 8192]);  mm_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:419 in forward, code: hidden_states = self.c_fc(hidden_states)
        reshape_default_1: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_1, [32, 128, 8192]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:62 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_tensor: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(reshape_default_1, 0.5)
        mul_tensor_1: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  mul_tensor = None
        pow_tensor_scalar: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(reshape_default_1, 3.0)
        mul_tensor_2: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(reshape_default_1, mul_tensor_2);  mul_tensor_2 = None
        mul_tensor_3: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7978845608028654);  add_tensor = None
        tanh_default: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_tensor_3);  mul_tensor_3 = None
        add_tensor_1: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_default, 1.0)
        mul_tensor_4: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(reshape_default, add_tensor_1);  reshape_default = add_tensor_1 = None
        mul_tensor_5: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_default, tanh_default);  tanh_default = None
        sub_tensor: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_tensor_5);  mul_tensor_5 = None
        mul_tensor_6: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_tensor_1, sub_tensor);  mul_tensor_1 = sub_tensor = None
        mul_tensor_7: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_tensor_6, 0.7978845608028654);  mul_tensor_6 = None
        mul_tensor_8: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.044715)
        pow_tensor_scalar_1: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(reshape_default_1, 2.0);  reshape_default_1 = None
        mul_scalar: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 3.0);  pow_tensor_scalar_1 = None
        mul_tensor_9: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_tensor_8, mul_scalar);  mul_tensor_8 = mul_scalar = None
        add_tensor_2: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(mul_tensor_7, mul_tensor_9);  mul_tensor_7 = mul_tensor_9 = None
        mul_tensor_10: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_tensor_4, 0.5);  mul_tensor_4 = None
        add_tensor_3: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_10);  add_tensor_2 = mul_tensor_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:419 in forward, code: hidden_states = self.c_fc(hidden_states)
        reshape_default_2: "f32[4096, 8192]" = torch.ops.aten.reshape.default(add_tensor_3, [4096, 8192]);  add_tensor_3 = None
        permute_default: "f32[8192, 4096]" = torch.ops.aten.permute.default(reshape_default_2, [1, 0])
        sum_dim_int_list: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(reshape_default_2, [0], True);  reshape_default_2 = None
        reshape_default_3: "f32[8192]" = torch.ops.aten.reshape.default(sum_dim_int_list, [8192]);  sum_dim_int_list = None
        return (permute_default, reshape_default_3)


def _default_make_inputs():
    return [
    torch.randn([4096, 8192], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 8192], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
