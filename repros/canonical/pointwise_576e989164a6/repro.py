"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForCausalLM_inference
Pattern hash: 576e989164a6
Shape hash: 715deb0b
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_55: "f32[16, 128, 256]", _shape_param_0, _shape_param_1, _shape_param_2, arg305_1: "f32[4096, 4096]", addmm_54: "f32[128, 16384]", _shape_param_3, _shape_param_4, arg308_1: "f32[4096, 16384]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        reshape_default: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_55, _shape_param_0);  bmm_55 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_default: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        reshape_default_1: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default_2: "f32[128, 4096]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_1: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        reshape_default_3: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_54, _shape_param_3);  addmm_54 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_tensor: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(reshape_default_3, 0.5)
        pow_tensor_scalar: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(reshape_default_3, 3.0)
        mul_tensor_1: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(reshape_default_3, mul_tensor_1);  reshape_default_3 = mul_tensor_1 = None
        mul_tensor_2: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7978845608028654);  add_tensor = None
        tanh_default: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_tensor_2);  mul_tensor_2 = None
        add_tensor_1: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_default, 1.0);  tanh_default = None
        mul_tensor_3: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor_1);  mul_tensor = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        reshape_default_4: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_tensor_3, _shape_param_4);  mul_tensor_3 = _shape_param_4 = None
        permute_default_2: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg308_1, [1, 0]);  arg308_1 = None
        return (reshape_default_2, permute_default_1, reshape_default_4, permute_default_2)


def _default_make_inputs():
    return [
    torch.randn([16, 128, 256], dtype=torch.float32, device='cuda'),
    [1, 16, 128, 256],  # _shape_param_0
    [1, 128, 4096],  # _shape_param_1
    [128, 4096],  # _shape_param_2
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 16384], dtype=torch.float32, device='cuda'),
    [1, 128, 16384],  # _shape_param_3
    [128, 16384],  # _shape_param_4
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
