"""
Standalone repro captured via capture_hook.
Label: torchbench_moondream_infer
Pattern hash: dea063f9ad2f
Shape hash: c5d66f2c
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
_shapes_config = "(T([1, 32, 512, 64], f16), T([2048, 2048], f16), T([512, 8192], f16), T([2048, 8192], f16), S([1, 512, -1]), S([512, 2048]), S([1, 512, 8192]), S([512, 8192]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_255: "f16[1, 32, 512, 64]", arg333_1: "f16[2048, 2048]", addmm_142: "f16[512, 8192]", arg337_1: "f16[2048, 8192]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default: "f16[1, 512, 32, 64]" = torch.ops.aten.permute.default(getitem_255, [0, 2, 1, 3]);  getitem_255 = None
        clone_default: "f16[1, 512, 32, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:251 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:252 in forward, code: attn_output = self.dense(attn_output)
        reshape_default_1: "f16[512, 2048]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default_1: "f16[2048, 2048]" = torch.ops.aten.permute.default(arg333_1, [1, 0]);  arg333_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:265 in forward, code: hidden_states = self.fc1(hidden_states)
        reshape_default_2: "f16[1, 512, 8192]" = torch.ops.aten.reshape.default(addmm_142, _shape_param_2);  addmm_142 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_tensor: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(reshape_default_2, 0.5)
        pow_tensor_scalar: "f16[1, 512, 8192]" = torch.ops.aten.pow.Tensor_Scalar(reshape_default_2, 3.0)
        mul_tensor_1: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(reshape_default_2, mul_tensor_1);  reshape_default_2 = mul_tensor_1 = None
        mul_tensor_2: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7978845608028654);  add_tensor = None
        tanh_default: "f16[1, 512, 8192]" = torch.ops.aten.tanh.default(mul_tensor_2);  mul_tensor_2 = None
        add_tensor_1: "f16[1, 512, 8192]" = torch.ops.aten.add.Tensor(tanh_default, 1.0);  tanh_default = None
        mul_tensor_3: "f16[1, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor_1);  mul_tensor = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:267 in forward, code: hidden_states = self.fc2(hidden_states)
        reshape_default_3: "f16[512, 8192]" = torch.ops.aten.reshape.default(mul_tensor_3, _shape_param_3);  mul_tensor_3 = _shape_param_3 = None
        permute_default_2: "f16[8192, 2048]" = torch.ops.aten.permute.default(arg337_1, [1, 0]);  arg337_1 = None
        return (reshape_default_1, permute_default_1, reshape_default_3, permute_default_2)



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
