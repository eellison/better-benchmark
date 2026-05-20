"""
Standalone repro captured via capture_hook.
Label: hf_GPTNeoForCausalLM_train
Pattern hash: b194dff598e9
Shape hash: 3092e0fe
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([4096, 2048], f32), T([32, 128, 2048], f32), T([2048], f32), T([2048], f32), T([2048, 2048], f32), T([2048, 2048], f32), S([32, 128, 2048]), S([4096, 2048]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_68: "f32[4096, 2048]", add_205: "f32[32, 128, 2048]", primals_326: "f32[2048]", primals_327: "f32[2048]", primals_328: "f32[2048, 2048]", primals_329: "f32[2048, 2048]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        reshape_default: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_68, _shape_param_0);  addmm_68 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_tensor: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_205, reshape_default);  add_205 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_326);  mul_tensor = primals_326 = None
        add_tensor_2: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_327);  mul_tensor_1 = primals_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        permute_default: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_328, [1, 0]);  primals_328 = None
        reshape_default_1: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        permute_default_1: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_329, [1, 0]);  primals_329 = None
        return (permute_default, reshape_default_1, permute_default_1)


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
