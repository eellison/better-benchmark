"""
Standalone repro captured via capture_hook.
Label: hf_t5_base_train
Pattern hash: a45454119949
Shape hash: 5e595c5e
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, primals_1: "f32[4, 128, 768]", primals_2: "f32[32128, 768]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1095 in torch_dynamo_resume_in_forward_at_1079, code: sequence_output = sequence_output * (self.model_dim**-0.5)
        mul_tensor: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(primals_1, 0.03608439182435161);  primals_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1097 in torch_dynamo_resume_in_forward_at_1079, code: lm_logits = self.lm_head(sequence_output)
        permute_default: "f32[768, 32128]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        reshape_default: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_tensor, _shape_param_0);  mul_tensor = _shape_param_0 = None
        return (permute_default, reshape_default)


def _default_make_inputs():
    return [
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32128, 768], dtype=torch.float32, device='cuda'),
    [512, 768],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
