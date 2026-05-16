"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, add_133: "f32[32, 512, 768]", getitem_45: "f32[32, 512, 1]", getitem_44: "f32[32, 512, 1]", arg259_1: "f32[768]", arg260_1: "f32[768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:532 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_tensor: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_133, getitem_45);  add_133 = getitem_45 = None
        add_tensor: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_default: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg259_1);  mul_tensor = arg259_1 = None
        add_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg260_1);  mul_tensor_1 = arg260_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: mixed_query_layer = self.query(hidden_states)
        reshape_default: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_tensor_1, [16384, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:343 in forward, code: mixed_key_layer = self.key(hidden_states)
        reshape_default_1: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_tensor_1, [16384, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:344 in forward, code: mixed_value_layer = self.value(hidden_states)
        reshape_default_2: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_tensor_1, [16384, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:365 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        reshape_default_3: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_tensor_1, [16384, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:346 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_default: "f32[32, 768, 512]" = torch.ops.aten.permute.default(add_tensor_1, [0, 2, 1]);  add_tensor_1 = None
        return (reshape_default, reshape_default_1, reshape_default_2, reshape_default_3, permute_default)


def _default_make_inputs():
    return [
    torch.randn([32, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
