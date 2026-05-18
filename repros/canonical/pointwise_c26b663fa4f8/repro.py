"""
Standalone repro captured via capture_hook.
Label: hf_LayoutLMForMaskedLM_infer
Pattern hash: c26b663fa4f8
Shape hash: 531dbf99
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
    def forward(self, addmm_default: "f32[16384, 30524]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:496 in forward, code: attention_mask = torch.ones(input_shape, device=device)
        full_default: "f32[32, 512]" = torch.ops.aten.full.default([32, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:503 in forward, code: extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        unsqueeze_default: "f32[32, 1, 512]" = torch.ops.aten.unsqueeze.default(full_default, 1);  full_default = None
        unsqueeze_default_1: "f32[32, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:506 in forward, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min
        sub_tensor: "f32[32, 1, 1, 512]" = torch.ops.aten.sub.Tensor(1.0, unsqueeze_default_1);  unsqueeze_default_1 = None
        full_default_1: "f32[32, 1, 1, 512]" = torch.ops.aten.full.default([32, 1, 1, 512], -0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:378 in forward, code: hidden_states = self.decoder(hidden_states)
        slice_tensor: "f32[16384, 30522]" = torch.ops.aten.slice.Tensor(addmm_default, 1, 0, -2);  addmm_default = None
        reshape_default: "f32[32, 512, 30522]" = torch.ops.aten.reshape.default(slice_tensor, _shape_param_0);  slice_tensor = _shape_param_0 = None
        return (sub_tensor, full_default_1, reshape_default)


def _default_make_inputs():
    return [
    torch.randn([16384, 30524], dtype=torch.float32, device='cuda'),
    [32, 512, 30522],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
