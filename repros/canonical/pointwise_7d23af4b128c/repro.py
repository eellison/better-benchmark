"""
Standalone repro captured via capture_hook.
Label: hf_XGLMForCausalLM_train
Pattern hash: 7d23af4b128c
Shape hash: 680fcfbf
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([4096, 1024], f32), T([3], i64, gen=Index(3)), T([32, 128, 1024], f32), T([512, 128, 128], f32), T([512, 128, 64], f32), T([512, 128, 64], f32), T([512, 64, 128], f32, stride=(8192, 1, 64)), S([32, 128, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_5: "f32[4096, 1024]", inductor_seeds_default: "i64[3]", add_3: "f32[32, 128, 1024]", mul_4: "f32[512, 128, 128]", view_11: "f32[512, 128, 64]", view_9: "f32[512, 128, 64]", permute_6: "f32[512, 64, 128]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:334 in forward, code: hidden_states = self.fc2(hidden_states)
        reshape_default: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(addmm_5, _shape_param_0);  addmm_5 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:335 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2);  inductor_seeds_default = None
        inductor_random_default: "f32[32, 128, 1024]" = torch.ops.prims.inductor_random.default([32, 128, 1024], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[32, 128, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(gt_scalar, reshape_default);  gt_scalar = reshape_default = None
        mul_tensor_1: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:336 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_3, mul_tensor_1);  add_3 = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:228 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        permute_default: "f32[512, 128, 128]" = torch.ops.aten.permute.default(mul_4, [0, 2, 1]);  mul_4 = None
        permute_default_1: "f32[512, 64, 128]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:194 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_default_2: "f32[512, 64, 128]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_default_3: "f32[512, 128, 64]" = torch.ops.aten.permute.default(permute_6, [0, 2, 1]);  permute_6 = None
        return (add_tensor, permute_default, permute_default_1, permute_default_2, permute_default_3)


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
