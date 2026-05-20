"""
Standalone repro captured via capture_hook.
Label: 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_train
Pattern hash: 2de488797120
Shape hash: 2f37475b
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([2048, 2560], f32), T([2], i64, max=2), T([16, 128, 2560], f32), T([512, 128, 80], f32), T([512, 128, 80], f32), T([512, 80, 128], f32), S([16, 128, 2560]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_5: "f32[2048, 2560]", inductor_seeds_default: "i64[2]", add_3: "f32[16, 128, 2560]", view_13: "f32[512, 128, 80]", view_9: "f32[512, 128, 80]", view_10: "f32[512, 80, 128]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:292 in forward, code: hidden_states = self.fc2(hidden_states)
        reshape_default: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_5, _shape_param_0);  addmm_5 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:293 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1);  inductor_seeds_default = None
        inductor_random_default: "f32[16, 128, 2560]" = torch.ops.prims.inductor_random.default([16, 128, 2560], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[16, 128, 2560]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(gt_scalar, reshape_default);  gt_scalar = reshape_default = None
        mul_tensor_1: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:294 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_3, mul_tensor_1);  add_3 = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default: "f32[512, 80, 128]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
        permute_default_1: "f32[512, 80, 128]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_default_2: "f32[512, 128, 80]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
        return (add_tensor, permute_default, permute_default_1, permute_default_2)


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
