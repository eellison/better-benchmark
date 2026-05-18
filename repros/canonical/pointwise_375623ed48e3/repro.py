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
    def forward(self, addmm_79: "f32[16384, 384]", convolution_23: "f32[32, 384, 512]", arg267_1: "f32[384, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: mixed_query_layer = self.query(hidden_states)
        reshape_default: "f32[32, 512, 384]" = torch.ops.aten.reshape.default(addmm_79, [32, 512, 384]);  addmm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:350 in forward, code: query_layer = mixed_query_layer.view(
        reshape_default_1: "f32[32, 512, 6, 64]" = torch.ops.aten.reshape.default(reshape_default, [32, -1, 6, 64])

        # No stacktrace found for following nodes
        permute_default: "f32[32, 6, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:283 in forward, code: x += self.bias
        add_tensor: "f32[32, 384, 512]" = torch.ops.aten.add.Tensor(convolution_23, arg267_1);  convolution_23 = arg267_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:347 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_default_1: "f32[32, 512, 384]" = torch.ops.aten.permute.default(add_tensor, [0, 2, 1]);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:359 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_tensor: "f32[32, 512, 384]" = torch.ops.aten.mul.Tensor(permute_default_1, reshape_default);  permute_default_1 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:361 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        clone_default: "f32[32, 512, 384]" = torch.ops.aten.clone.default(mul_tensor, memory_format = torch.contiguous_format);  mul_tensor = None
        reshape_default_2: "f32[16384, 384]" = torch.ops.aten.reshape.default(clone_default, [16384, 384]);  clone_default = None
        return (permute_default, reshape_default_2)


def _default_make_inputs():
    return [
    torch.randn([16384, 384], dtype=torch.float32, device='cuda'),
    torch.randn([32, 384, 512], dtype=torch.float32, device='cuda'),
    torch.randn([384, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
