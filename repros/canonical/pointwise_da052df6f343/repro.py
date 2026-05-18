"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_training
Pattern hash: da052df6f343
Shape hash: 3ec39694
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_695: "f32[1024, 128]", _shape_param_0, primals_50: "f32[128]", _shape_param_1, primals_48: "f32[128, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        reshape_default: "f32[8, 128, 128]" = torch.ops.aten.reshape.default(mm_695, _shape_param_0);  mm_695 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_tensor: "f32[8, 128, 128]" = torch.ops.aten.mul.Tensor(reshape_default, primals_50);  reshape_default = primals_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        reshape_default_1: "f32[1024, 128]" = torch.ops.aten.reshape.default(mul_tensor, _shape_param_1);  mul_tensor = _shape_param_1 = None
        permute_default: "f32[512, 128]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
        permute_default_1: "f32[128, 512]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_1, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([1024, 128], dtype=torch.float32, device='cuda'),
    [8, 128, 128],  # _shape_param_0
    torch.randn([128], dtype=torch.float32, device='cuda'),
    [1024, 128],  # _shape_param_1
    torch.randn([128, 512], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
