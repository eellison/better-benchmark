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
    def forward(self, convert_element_type_265: "f32[4, 512, 768]", getitem_157: "f32[4, 512, 1]", getitem_156: "f32[4, 512, 1]", arg195_1: "f16[768]", arg196_1: "f16[768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:666 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        sub_tensor: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_265, getitem_157);  convert_element_type_265 = getitem_157 = None
        add_tensor: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-05);  getitem_156 = None
        rsqrt_default: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg195_1);  mul_tensor = arg195_1 = None
        add_tensor_1: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg196_1);  mul_tensor_1 = arg196_1 = None
        convert_element_type_default: "f16[4, 512, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:833 in forward, code: logits = self.lm_head(outputs[0]).contiguous()
        reshape_default: "f16[2048, 768]" = torch.ops.aten.reshape.default(convert_element_type_default, [2048, 768]);  convert_element_type_default = None
        return reshape_default


def _default_make_inputs():
    return [
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float16, device='cuda'),
    torch.randn([768], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
