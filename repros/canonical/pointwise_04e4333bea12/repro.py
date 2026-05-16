"""
Standalone repro captured via capture_hook.
Label: phi_2
Pattern hash: 04e4333bea12
Shape hash: 199a8655
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_343: "f16[4, 32, 512, 80]", _shape_param_0, _shape_param_1, arg446_1: "f16[2560, 2560]", addmm_190: "f16[2048, 10240]", _shape_param_2, _shape_param_3, arg450_1: "f16[2560, 10240]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default: "f16[4, 512, 32, 80]" = torch.ops.aten.permute.default(getitem_343, [0, 2, 1, 3]);  getitem_343 = None
        clone_default: "f16[4, 512, 32, 80]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:251 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default: "f16[4, 512, 2560]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:252 in forward, code: attn_output = self.dense(attn_output)
        reshape_default_1: "f16[2048, 2560]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default_1: "f16[2560, 2560]" = torch.ops.aten.permute.default(arg446_1, [1, 0]);  arg446_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:265 in forward, code: hidden_states = self.fc1(hidden_states)
        reshape_default_2: "f16[4, 512, 10240]" = torch.ops.aten.reshape.default(addmm_190, _shape_param_2);  addmm_190 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_tensor: "f16[4, 512, 10240]" = torch.ops.aten.mul.Tensor(reshape_default_2, 0.5)
        pow_tensor_scalar: "f16[4, 512, 10240]" = torch.ops.aten.pow.Tensor_Scalar(reshape_default_2, 3.0)
        mul_tensor_1: "f16[4, 512, 10240]" = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor: "f16[4, 512, 10240]" = torch.ops.aten.add.Tensor(reshape_default_2, mul_tensor_1);  reshape_default_2 = mul_tensor_1 = None
        mul_tensor_2: "f16[4, 512, 10240]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7978845608028654);  add_tensor = None
        tanh_default: "f16[4, 512, 10240]" = torch.ops.aten.tanh.default(mul_tensor_2);  mul_tensor_2 = None
        add_tensor_1: "f16[4, 512, 10240]" = torch.ops.aten.add.Tensor(tanh_default, 1.0);  tanh_default = None
        mul_tensor_3: "f16[4, 512, 10240]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor_1);  mul_tensor = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:267 in forward, code: hidden_states = self.fc2(hidden_states)
        reshape_default_3: "f16[2048, 10240]" = torch.ops.aten.reshape.default(mul_tensor_3, _shape_param_3);  mul_tensor_3 = _shape_param_3 = None
        permute_default_2: "f16[10240, 2560]" = torch.ops.aten.permute.default(arg450_1, [1, 0]);  arg450_1 = None
        return (reshape_default_1, permute_default_1, reshape_default_3, permute_default_2)


def _default_make_inputs():
    return [
    torch.randn([4, 32, 512, 80], dtype=torch.float16, device='cuda'),
    [4, 512, -1],  # _shape_param_0
    [2048, 2560],  # _shape_param_1
    torch.randn([2560, 2560], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 10240], dtype=torch.float16, device='cuda'),
    [4, 512, 10240],  # _shape_param_2
    [2048, 10240],  # _shape_param_3
    torch.randn([2560, 10240], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
