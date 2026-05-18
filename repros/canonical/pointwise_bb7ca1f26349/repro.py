"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_422: "f32[4096, 1024]", gt_4: "b8[32, 128, 1024]", mm_4: "f32[4096, 1024]", mm_5: "f32[4096, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:199 in forward, code: hidden_states = self.wo(hidden_states)
        reshape_default: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_422, [32, 128, 1024]);  mm_422 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:187 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_default: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(gt_4, torch.float32);  gt_4 = None
        mul_tensor: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  reshape_default = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:184 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        reshape_default_1: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_4, [32, 128, 1024]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:62 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_tensor_2: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(reshape_default_1, 0.5)
        pow_tensor_scalar: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(reshape_default_1, 3.0)
        mul_tensor_3: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(reshape_default_1, mul_tensor_3);  mul_tensor_3 = None
        mul_tensor_4: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7978845608028654);  add_tensor = None
        tanh_default: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_tensor_4);  mul_tensor_4 = None
        add_tensor_1: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_default, 1.0)
        mul_tensor_5: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_2, add_tensor_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:186 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_tensor_6: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_1, mul_tensor_5);  mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:185 in forward, code: hidden_linear = self.wi_1(hidden_states)
        reshape_default_2: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_5, [32, 128, 1024]);  mm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:186 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_tensor_7: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_1, reshape_default_2);  mul_tensor_1 = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:185 in forward, code: hidden_linear = self.wi_1(hidden_states)
        reshape_default_3: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_tensor_6, [4096, 1024]);  mul_tensor_6 = None
        permute_default: "f32[1024, 4096]" = torch.ops.aten.permute.default(reshape_default_3, [1, 0]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:62 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_tensor_8: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_7, mul_tensor_2);  mul_tensor_2 = None
        mul_tensor_9: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_7, add_tensor_1);  mul_tensor_7 = add_tensor_1 = None
        mul_tensor_10: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_default, tanh_default);  tanh_default = None
        sub_tensor: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_tensor_10);  mul_tensor_10 = None
        mul_tensor_11: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_8, sub_tensor);  mul_tensor_8 = sub_tensor = None
        mul_tensor_12: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_11, 0.7978845608028654);  mul_tensor_11 = None
        mul_tensor_13: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_12, 0.044715)
        pow_tensor_scalar_1: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(reshape_default_1, 2.0);  reshape_default_1 = None
        mul_scalar: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 3.0);  pow_tensor_scalar_1 = None
        mul_tensor_14: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_13, mul_scalar);  mul_tensor_13 = mul_scalar = None
        add_tensor_2: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_12, mul_tensor_14);  mul_tensor_12 = mul_tensor_14 = None
        mul_tensor_15: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_9, 0.5);  mul_tensor_9 = None
        add_tensor_3: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_15);  add_tensor_2 = mul_tensor_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:184 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        reshape_default_4: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_tensor_3, [4096, 1024]);  add_tensor_3 = None
        permute_default_1: "f32[1024, 4096]" = torch.ops.aten.permute.default(reshape_default_4, [1, 0]);  reshape_default_4 = None
        return (permute_default, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([4096, 1024], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [32, 128, 1024], dtype=torch.bool, device='cuda'),
    torch.randn([4096, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
