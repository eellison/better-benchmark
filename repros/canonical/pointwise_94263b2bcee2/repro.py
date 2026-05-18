"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[2048, 256]", arg2_1: "bf16[256]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:146 in forward, code: scores = router_logits.sigmoid()
        sigmoid_default: "f32[2048, 256]" = torch.ops.aten.sigmoid.default(mm);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:125 in get_topk_indices, code: scores_for_choice = scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias.unsqueeze(0)
        unsqueeze_default: "bf16[1, 256]" = torch.ops.aten.unsqueeze.default(arg2_1, 0);  arg2_1 = None
        add_tensor: "f32[2048, 256]" = torch.ops.aten.add.Tensor(sigmoid_default, unsqueeze_default);  sigmoid_default = unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:127 in get_topk_indices, code: scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
        reshape_default: "f32[2048, 8, 32]" = torch.ops.aten.reshape.default(add_tensor, [-1, 8, 32]);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:128 in get_topk_indices, code: .topk(2, dim=-1)[0]
        topk_default = torch.ops.aten.topk.default(reshape_default, 2);  reshape_default = None
        getitem: "f32[2048, 8, 2]" = topk_default[0];  topk_default = None
        _output_to_half_0: "bf16[2048, 8, 2]" = torch.ops.prims.convert_element_type.default(getitem, torch.bfloat16);  getitem = None
        return _output_to_half_0


def _default_make_inputs():
    return [
    torch.randn([2048, 256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
