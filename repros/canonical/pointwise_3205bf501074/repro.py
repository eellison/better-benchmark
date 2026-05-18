"""
Standalone repro captured via capture_hook.
Label: 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_training
Pattern hash: 3205bf501074
Shape hash: b0063428
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_9: "f32[1024, 2560]", _shape_param_0, inductor_seeds_default: "i64[3]", add_6: "f32[8, 128, 2560]", rsqrt_2: "f32[8, 128, 1]", view_25: "f32[256, 128, 80]", view_21: "f32[256, 128, 80]", view_22: "f32[256, 80, 128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        reshape_default: "f32[8, 128, 2560]" = torch.ops.aten.reshape.default(addmm_9, _shape_param_0);  addmm_9 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:391 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 128, 2560]" = torch.ops.prims.inductor_random.default([8, 128, 2560], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[8, 128, 2560]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[8, 128, 2560]" = torch.ops.aten.mul.Tensor(gt_scalar, reshape_default);  gt_scalar = reshape_default = None
        mul_tensor_1: "f32[8, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "f32[8, 128, 2560]" = torch.ops.aten.add.Tensor(add_6, mul_tensor_1);  add_6 = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        div_tensor: "f32[8, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 2560);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default: "f32[256, 80, 128]" = torch.ops.aten.permute.default(view_25, [0, 2, 1]);  view_25 = None
        permute_default_1: "f32[256, 80, 128]" = torch.ops.aten.permute.default(view_21, [0, 2, 1]);  view_21 = None
        permute_default_2: "f32[256, 128, 80]" = torch.ops.aten.permute.default(view_22, [0, 2, 1]);  view_22 = None
        return (add_tensor, div_tensor, permute_default, permute_default_1, permute_default_2)


def _default_make_inputs():
    return [
    torch.randn([1024, 2560], dtype=torch.float32, device='cuda'),
    [8, 128, 2560],  # _shape_param_0
    torch.randint(0, 2, [3], dtype=torch.int64, device='cuda'),
    torch.randn([8, 128, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256, 128, 80], dtype=torch.float32, device='cuda'),
    torch.randn([256, 128, 80], dtype=torch.float32, device='cuda'),
    torch.randn([256, 80, 128], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
