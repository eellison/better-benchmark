"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForCausalLM_train
Pattern hash: 3eb3759859a5
Shape hash: beb809fa
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([1, 16, 128, 256], f32, stride=(524288, 256, 4096, 1)), T([4096, 4096], f32), T([128, 16384], f32), T([4096, 16384], f32), S([1, 128, 4096]), S([128, 4096]), S([1, 128, 16384]), S([128, 16384]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_114: "f32[1, 16, 128, 256]", primals_306: "f32[4096, 4096]", addmm_54: "f32[128, 16384]", primals_309: "f32[4096, 16384]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_default: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_114, [0, 2, 1, 3]);  getitem_114 = None
        clone_default: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        reshape_default: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        permute_default_1: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_306, [1, 0]);  primals_306 = None
        reshape_default_1: "f32[128, 4096]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        reshape_default_2: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_54, _shape_param_2);  addmm_54 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_tensor: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(reshape_default_2, 0.5)
        pow_tensor_scalar: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(reshape_default_2, 3.0)
        mul_tensor_1: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(reshape_default_2, mul_tensor_1);  reshape_default_2 = mul_tensor_1 = None
        mul_tensor_2: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7978845608028654);  add_tensor = None
        tanh_default: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_tensor_2);  mul_tensor_2 = None
        add_tensor_1: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_default, 1.0);  tanh_default = None
        mul_tensor_3: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor_1);  mul_tensor = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        reshape_default_3: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_tensor_3, _shape_param_3);  mul_tensor_3 = _shape_param_3 = None
        permute_default_2: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_309, [1, 0]);  primals_309 = None
        return (permute_default_1, reshape_default_1, reshape_default_3, permute_default_2)


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
