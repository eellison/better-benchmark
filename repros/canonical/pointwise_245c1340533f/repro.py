"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_infer
Pattern hash: 245c1340533f
Shape hash: e5f7fe67
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
    def forward(self, mm_default: "f32[32768, 30524]", _shape_param_0, arg1118_1: "f32[30522]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:507 in forward, code: hidden_states = hidden_states.matmul(torch.cat([self.decoder.weight.t(), self.dense.weight], dim=0))
        slice_tensor: "f32[32768, 30522]" = torch.ops.aten.slice.Tensor(mm_default, 1, 0, -2);  mm_default = None
        reshape_default: "f32[256, 128, 30522]" = torch.ops.aten.reshape.default(slice_tensor, _shape_param_0);  slice_tensor = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:508 in forward, code: hidden_states += self.decoder.bias
        add_tensor: "f32[256, 128, 30522]" = torch.ops.aten.add.Tensor(reshape_default, arg1118_1);  reshape_default = arg1118_1 = None
        return add_tensor


def _default_make_inputs():
    return [
    torch.randn([32768, 30524], dtype=torch.float32, device='cuda'),
    [256, 128, 30522],  # _shape_param_0
    torch.randn([30522], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
