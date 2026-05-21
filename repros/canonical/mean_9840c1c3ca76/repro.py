"""
Standalone repro captured via capture_hook.
Label: torchbench_llava_infer
Pattern hash: 9840c1c3ca76
Shape hash: af568cdb
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
_shapes_config = "(T([512, 4096], f16), T([1, 512, 4096], f16), T([4096], f16), S([1, 512, 4096]), S([512, 4096]), S([512, 4096]))"

class Repro(torch.nn.Module):
    def forward(self, mm_220: "f16[512, 4096]", add_219: "f16[1, 512, 4096]", arg287_1: "f16[4096]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:288 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_220, _shape_param_0);  mm_220 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:325 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "f16[1, 512, 4096]" = torch.ops.aten.add.Tensor(add_219, reshape_default);  add_219 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:64 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default: "f32[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:65 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar: "f32[1, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2)
        mean_dim: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:66 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor_1: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_default, rsqrt_default);  convert_element_type_default = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:67 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_1: "f16[1, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float16);  mul_tensor = None
        mul_tensor_1: "f16[1, 512, 4096]" = torch.ops.aten.mul.Tensor(arg287_1, convert_element_type_default_1);  arg287_1 = convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:183 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        reshape_default_1: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_1);  _shape_param_1 = None
        reshape_default_2: "f16[512, 4096]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_2);  mul_tensor_1 = _shape_param_2 = None
        return (reshape_default_1, reshape_default_2)



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
