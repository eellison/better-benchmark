"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_inference
Pattern hash: d9382b416011
Shape hash: 436c061a
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_17: "f32[48, 128, 64]", _shape_param_0, _shape_param_1, _shape_param_2, arg81_1: "f32[512, 384]", mm_53: "f32[1024, 1024]", _shape_param_3, mm_54: "f32[1024, 1024]", _shape_param_4, _shape_param_5, arg74_1: "f32[512, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default: "f32[8, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_17, _shape_param_0);  bmm_17 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default: "f32[8, 128, 6, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[8, 128, 6, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        reshape_default_1: "f32[8, 128, 384]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        reshape_default_2: "f32[1024, 384]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_1: "f32[384, 512]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        reshape_default_3: "f32[8, 128, 1024]" = torch.ops.aten.reshape.default(mm_53, _shape_param_3);  mm_53 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_tensor: "f32[8, 128, 1024]" = torch.ops.aten.mul.Tensor(reshape_default_3, 0.5)
        pow_tensor_scalar: "f32[8, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(reshape_default_3, 3.0)
        mul_tensor_1: "f32[8, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor: "f32[8, 128, 1024]" = torch.ops.aten.add.Tensor(reshape_default_3, mul_tensor_1);  reshape_default_3 = mul_tensor_1 = None
        mul_tensor_2: "f32[8, 128, 1024]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7978845608028654);  add_tensor = None
        tanh_default: "f32[8, 128, 1024]" = torch.ops.aten.tanh.default(mul_tensor_2);  mul_tensor_2 = None
        add_tensor_1: "f32[8, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_default, 1.0);  tanh_default = None
        mul_tensor_3: "f32[8, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor_1);  mul_tensor = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        reshape_default_4: "f32[8, 128, 1024]" = torch.ops.aten.reshape.default(mm_54, _shape_param_4);  mm_54 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_tensor_4: "f32[8, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_3, reshape_default_4);  mul_tensor_3 = reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        reshape_default_5: "f32[1024, 1024]" = torch.ops.aten.reshape.default(mul_tensor_4, _shape_param_5);  mul_tensor_4 = _shape_param_5 = None
        permute_default_2: "f32[1024, 512]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        return (reshape_default_2, permute_default_1, reshape_default_5, permute_default_2)


def _default_make_inputs():
    return [
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    [8, 6, 128, 64],  # _shape_param_0
    [8, 128, -1],  # _shape_param_1
    [1024, 384],  # _shape_param_2
    torch.randn([512, 384], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    [8, 128, 1024],  # _shape_param_3
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    [8, 128, 1024],  # _shape_param_4
    [1024, 1024],  # _shape_param_5
    torch.randn([512, 1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
