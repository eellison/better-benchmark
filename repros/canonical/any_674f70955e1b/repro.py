"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_infer
Pattern hash: 674f70955e1b
Shape hash: 0a3b54e6
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 1024]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1191 in forward, code: is_index_global_attn = attention_mask > 0
        gt_scalar: "b8[8, 1024]" = torch.ops.aten.gt.Scalar(arg0_1, 0)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1194 in forward, code: is_global_attn = is_index_global_attn.flatten().any().item()
        reshape_default: "b8[8192]" = torch.ops.aten.reshape.default(gt_scalar, _shape_param_0);  gt_scalar = _shape_param_0 = None
        any_default: "b8[]" = torch.ops.aten.any.default(reshape_default);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1190 in forward, code: is_index_masked = attention_mask < 0
        lt_scalar: "b8[8, 1024]" = torch.ops.aten.lt.Scalar(arg0_1, 0);  arg0_1 = None
        return (any_default, lt_scalar)


def _default_make_inputs():
    return [
    torch.randn([8, 1024], dtype=torch.float32, device='cuda'),
    [8192],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
