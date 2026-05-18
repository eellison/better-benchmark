"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_inference
Pattern hash: 92cc85092323
Shape hash: c1236973
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
    def forward(self, arg2_1: "f32[30522, 128]", arg0_1: "i64[8, 128]", _shape_param_0, arg3_1: "f32[512, 384]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:113 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding_default: "f32[8, 128, 128]" = torch.ops.aten.embedding.default(arg2_1, arg0_1, 0);  arg2_1 = arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:125 in forward, code: nn.functional.pad(inputs_embeds[:, 1:], [0, 0, 0, 1, 0, 0], value=0.0),
        slice_tensor: "f32[8, 127, 128]" = torch.ops.aten.slice.Tensor(embedding_default, 1, 1, 9223372036854775807)

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default: "f32[8, 128, 128]" = torch.ops.aten.constant_pad_nd.default(slice_tensor, [0, 0, 0, 1, 0, 0], 0.0);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:127 in forward, code: nn.functional.pad(inputs_embeds[:, :-1], [0, 0, 1, 0, 0, 0], value=0.0),
        slice_tensor_1: "f32[8, 127, 128]" = torch.ops.aten.slice.Tensor(embedding_default, 1, 0, -1)

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default_1: "f32[8, 128, 128]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_1, [0, 0, 1, 0, 0, 0], 0.0);  slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:123 in forward, code: inputs_embeds = torch.cat(
        cat_default: "f32[8, 128, 384]" = torch.ops.aten.cat.default([constant_pad_nd_default, embedding_default, constant_pad_nd_default_1], 2);  constant_pad_nd_default = embedding_default = constant_pad_nd_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:132 in forward, code: inputs_embeds = self.embedding_transformation(inputs_embeds)
        reshape_default: "f32[1024, 384]" = torch.ops.aten.reshape.default(cat_default, _shape_param_0);  cat_default = _shape_param_0 = None
        permute_default: "f32[384, 512]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        return (reshape_default, permute_default)


def _default_make_inputs():
    return [
    torch.randn([30522, 128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 128], dtype=torch.int64, device='cuda'),
    [1024, 384],  # _shape_param_0
    torch.randn([512, 384], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
