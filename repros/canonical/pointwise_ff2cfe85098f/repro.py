"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_training
Pattern hash: ff2cfe85098f
Shape hash: 6b04a0af
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
    def forward(self, mm_715: "f32[1024, 512]", _shape_param_0, mul_784: "f32[8, 128, 512]", mm_721: "f32[1024, 512]", _shape_param_1, mm_723: "f32[1024, 512]", _shape_param_2, primals_8: "f32[512]", _shape_param_3, primals_4: "f32[512, 384]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        reshape_default: "f32[8, 128, 512]" = torch.ops.aten.reshape.default(mm_715, _shape_param_0);  mm_715 = _shape_param_0 = None
        add_tensor: "f32[8, 128, 512]" = torch.ops.aten.add.Tensor(mul_784, reshape_default);  mul_784 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        reshape_default_1: "f32[8, 128, 512]" = torch.ops.aten.reshape.default(mm_721, _shape_param_1);  mm_721 = _shape_param_1 = None
        add_tensor_1: "f32[8, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_1);  add_tensor = reshape_default_1 = None
        reshape_default_2: "f32[8, 128, 512]" = torch.ops.aten.reshape.default(mm_723, _shape_param_2);  mm_723 = _shape_param_2 = None
        add_tensor_2: "f32[8, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_2);  add_tensor_1 = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_tensor: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor_2, primals_8);  add_tensor_2 = primals_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:132 in forward, code: inputs_embeds = self.embedding_transformation(inputs_embeds)
        reshape_default_3: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_tensor, _shape_param_3);  mul_tensor = _shape_param_3 = None
        permute_default: "f32[384, 512]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_default_1: "f32[512, 384]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_3, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    [8, 128, 512],  # _shape_param_0
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    [8, 128, 512],  # _shape_param_1
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    [8, 128, 512],  # _shape_param_2
    torch.randn([512], dtype=torch.float32, device='cuda'),
    [1024, 512],  # _shape_param_3
    torch.randn([512, 384], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
