"""
Standalone repro captured via capture_hook.
Label: hf_BartForCausalLM_training
Pattern hash: 442665c4fed5
Shape hash: 798c8fb5
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
    def forward(self, getitem_6: "f32[8, 1024, 1]", add_4: "f32[8, 1024, 1024]", getitem_7: "f32[8, 1024, 1]", primals_17: "f32[1024]", primals_18: "f32[1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:388 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        add_tensor: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_default: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_4, getitem_7);  add_4 = getitem_7 = None
        mul_tensor: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_1: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_17);  mul_tensor = primals_17 = None
        add_tensor_1: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_18);  mul_tensor_1 = primals_18 = None
        div_tensor: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_default, 1024);  rsqrt_default = None
        return (add_tensor_1, div_tensor)


def _default_make_inputs():
    return [
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
