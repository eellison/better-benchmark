"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_80: "f32[16384, 384]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:365 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        reshape_default: "f32[32, 512, 384]" = torch.ops.aten.reshape.default(addmm_80, [32, 512, 384]);  addmm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:367 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        permute_default: "f32[32, 384, 512]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1]);  reshape_default = None
        clone_default: "f32[32, 384, 512]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        unsqueeze_default: "f32[32, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_default, -1);  clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:368 in forward, code: conv_out_layer = nn.functional.unfold(
        constant_pad_nd_default: "f32[32, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_default, [0, 0, 4, 4], 0.0);  unsqueeze_default = None
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_1: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_default, 0);  iota_default = None
        iota_default_1: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_2: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_default_1, -1);  iota_default_1 = None
        add_tensor: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_default_1, unsqueeze_default_2);  unsqueeze_default_1 = unsqueeze_default_2 = None
        unsqueeze_default_3: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_tensor, -1);  add_tensor = None
        unsqueeze_default_4: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, -1);  unsqueeze_default_3 = None
        iota_default_2: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_5: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_default_2, 0);  iota_default_2 = None
        iota_default_3: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_6: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_default_3, -1);  iota_default_3 = None
        add_tensor_1: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_default_5, unsqueeze_default_6);  unsqueeze_default_5 = unsqueeze_default_6 = None
        index_tensor: "f32[32, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(constant_pad_nd_default, [None, None, unsqueeze_default_4, add_tensor_1]);  constant_pad_nd_default = unsqueeze_default_4 = add_tensor_1 = None
        permute_default_1: "f32[32, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_tensor, [0, 1, 2, 4, 3, 5]);  index_tensor = None
        reshape_default_1: "f32[32, 3456, 512]" = torch.ops.aten.reshape.default(permute_default_1, [32, 3456, 512]);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:375 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        permute_default_2: "f32[32, 512, 3456]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1]);  reshape_default_1 = None
        reshape_default_2: "f32[32, 512, 384, 9]" = torch.ops.aten.reshape.default(permute_default_2, [32, 512, 384, 9]);  permute_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:378 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        clone_default_1: "f32[32, 512, 384, 9]" = torch.ops.aten.clone.default(reshape_default_2, memory_format = torch.contiguous_format);  reshape_default_2 = None
        reshape_default_3: "f32[98304, 64, 9]" = torch.ops.aten.reshape.default(clone_default_1, [98304, 64, 9]);  clone_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:379 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_default: "f32[98304, 64, 9]" = torch.ops.aten.expand.default(reshape_default_3, [98304, 64, 9]);  reshape_default_3 = None
        return expand_default


def _default_make_inputs():
    return [
    torch.randn([16384, 384], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
