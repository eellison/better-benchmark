"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_T5_infer
Pattern hash: 61cb3a0a580b
Shape hash: 6f24d739
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
_shapes_config = "(T([32128, 512], f16), T([1, 2048], i64, gen=Index(32128)), T([512], f16), T([512, 512], f16), T([512, 512], f16), S([2048, 512]), S([2048, 512]))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f16[32128, 512]", arg0_1: "i64[1, 2048]", arg2_1: "f16[512]", arg3_1: "f16[512, 512]", arg4_1: "f16[512, 512]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:669 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding_default: "f16[1, 2048, 512]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_default: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(embedding_default, torch.float32)
        pow_tensor_scalar: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2);  convert_element_type_default = None
        mean_dim: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(embedding_default, rsqrt_default);  embedding_default = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_default_1: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float16);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_1: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg2_1, convert_element_type_default_1);  arg2_1 = convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_0);  _shape_param_0 = None
        permute_default: "f16[512, 512]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_1: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_1);  mul_tensor_1 = _shape_param_1 = None
        permute_default_1: "f16[512, 512]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        return (reshape_default, permute_default, reshape_default_1, permute_default_1)



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
