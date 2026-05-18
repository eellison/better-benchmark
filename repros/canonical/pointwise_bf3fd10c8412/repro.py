"""
Standalone repro captured via capture_hook.
Label: hf_TrOCRForCausalLM_inference
Pattern hash: bf3fd10c8412
Shape hash: 849f3e93
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
    def forward(self, add_99: "f32[8, 256, 1024]", getitem_49: "f32[8, 256, 1]", getitem_48: "f32[8, 256, 1]", arg195_1: "f32[1024]", arg196_1: "f32[1024]", _shape_param_0, arg1_1: "f32[50265, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:380 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        sub_tensor: "f32[8, 256, 1024]" = torch.ops.aten.sub.Tensor(add_99, getitem_49);  add_99 = getitem_49 = None
        add_tensor: "f32[8, 256, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_default: "f32[8, 256, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[8, 256, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 256, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, arg195_1);  mul_tensor = arg195_1 = None
        add_tensor_1: "f32[8, 256, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg196_1);  mul_tensor_1 = arg196_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:756 in forward, code: logits = self.output_projection(outputs[0])
        reshape_default: "f32[2048, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_0);  add_tensor_1 = _shape_param_0 = None
        permute_default: "f32[1024, 50265]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        constant_pad_nd_default: "f32[1024, 50268]" = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 3, 0, 0]);  permute_default = None
        return (reshape_default, constant_pad_nd_default)


def _default_make_inputs():
    return [
    torch.randn([8, 256, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    [2048, 1024],  # _shape_param_0
    torch.randn([50265, 1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
