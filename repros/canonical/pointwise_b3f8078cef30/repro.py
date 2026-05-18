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
    def forward(self, mm_default_1: "f32[512, 30524]", mm_725: "f32[32768, 384]", primals_1: "i64[256, 128]", full_default_3: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:632 in forward, code: hidden_states = hidden_states.matmul(torch.cat([self.decoder.weight.t(), self.dense.weight], dim=0))
        slice_tensor: "f32[512, 30522]" = torch.ops.aten.slice.Tensor(mm_default_1, 1, 0, -2);  mm_default_1 = None
        slice_tensor_1: "f32[128, 30522]" = torch.ops.aten.slice.Tensor(slice_tensor, 0, 0, 128)
        slice_tensor_2: "f32[384, 30522]" = torch.ops.aten.slice.Tensor(slice_tensor, 0, 128, 512);  slice_tensor = None
        permute_default: "f32[30522, 128]" = torch.ops.aten.permute.default(slice_tensor_1, [1, 0]);  slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:208 in forward, code: inputs_embeds = self.embedding_transformation(inputs_embeds)
        reshape_default: "f32[256, 128, 384]" = torch.ops.aten.reshape.default(mm_725, [256, 128, 384]);  mm_725 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:199 in forward, code: inputs_embeds = torch.cat(
        slice_tensor_3: "f32[256, 128, 128]" = torch.ops.aten.slice.Tensor(reshape_default, 2, 0, 128)
        slice_tensor_4: "f32[256, 128, 128]" = torch.ops.aten.slice.Tensor(reshape_default, 2, 128, 256)
        slice_tensor_5: "f32[256, 128, 128]" = torch.ops.aten.slice.Tensor(reshape_default, 2, 256, 384);  reshape_default = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default: "f32[256, 127, 128]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_5, [0, 0, -1, 0, 0, 0]);  slice_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: nn.functional.pad(inputs_embeds[:, :-1], [0, 0, 1, 0, 0, 0], value=0.0),
        full_default: "f32[256, 128, 128]" = torch.ops.aten.full.default([256, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default: "f32[256, 128, 128]" = torch.ops.aten.slice_scatter.default(full_default, constant_pad_nd_default, 1, 0, -1);  constant_pad_nd_default = None
        add_tensor: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(slice_tensor_4, slice_scatter_default);  slice_tensor_4 = slice_scatter_default = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default_1: "f32[256, 127, 128]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_3, [0, 0, 0, -1, 0, 0]);  slice_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:201 in forward, code: nn.functional.pad(inputs_embeds[:, 1:], [0, 0, 0, 1, 0, 0], value=0.0),
        slice_scatter_default_1: "f32[256, 128, 128]" = torch.ops.aten.slice_scatter.default(full_default, constant_pad_nd_default_1, 1, 1, 9223372036854775807);  full_default = constant_pad_nd_default_1 = None
        add_tensor_1: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(add_tensor, slice_scatter_default_1);  add_tensor = slice_scatter_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:189 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        eq_scalar: "b8[256, 128]" = torch.ops.aten.eq.Scalar(primals_1, 0)
        unsqueeze_default: "b8[256, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[256, 128, 128]" = torch.ops.aten.where.self(unsqueeze_default, full_default_3, add_tensor_1);  unsqueeze_default = full_default_3 = add_tensor_1 = None
        full_default_4: "f32[30522, 128]" = torch.ops.aten.full.default([30522, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[30522, 128]" = torch.ops.aten.index_put.default(full_default_4, [primals_1], where_self, True);  full_default_4 = primals_1 = where_self = None
        add_tensor_2: "f32[30522, 128]" = torch.ops.aten.add.Tensor(permute_default, index_put_default);  permute_default = index_put_default = None
        return (slice_tensor_2, add_tensor_2)


def _default_make_inputs():
    return [
    torch.randn([512, 30524], dtype=torch.float32, device='cuda'),
    torch.randn([32768, 384], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [256, 128], dtype=torch.int64, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
