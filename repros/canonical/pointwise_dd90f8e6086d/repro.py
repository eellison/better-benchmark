"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Longformer_infer
Pattern hash: dd90f8e6086d
Shape hash: 4fa461c7
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([4096, 768], f16), S([4096, 1, 768]), S([4096, 1, 12, 64]), S([12, 4096, 64]), S([192, 768, 64]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_68: "f16[4096, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        reshape_default: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_68, _shape_param_0);  addmm_68 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        reshape_default_1: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(reshape_default_1, [1, 0, 2, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_default_1: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        reshape_default_2: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_default_1, _shape_param_2);  permute_default_1 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default: "f16[12, 4608, 64]" = torch.ops.aten.constant_pad_nd.default(reshape_default_2, [0, 0, 256, 256], -1.0);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_default: "f16[12, 16, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_default, [12, 16, 768, 64], [294912, 16384, 64, 1]);  constant_pad_nd_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_default: "f16[12, 16, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_default, 4);  as_strided_default = None
        clone_default: "f16[12, 16, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_default, memory_format = torch.contiguous_format);  unsqueeze_default = None
        reshape_default_3: "f16[192, 768, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        return reshape_default_3



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
