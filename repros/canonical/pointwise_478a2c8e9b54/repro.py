"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_infer
Pattern hash: 478a2c8e9b54
Shape hash: e4b27ea8
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
_shapes_config = "(T([4096, 768], f16), T([1, 12, 4096], i64, gen=Index(4096)), S([1, 4096, 768]), S([1, 4096, 12, 64]), S([-1, -1, -1, 64]), S([1, 12, -1, 64, 64]), S([1, 12, 64, 128, 64]), S([768, 128, 64]))"

class Repro(torch.nn.Module):
    def forward(self, mm_19: "f16[4096, 768]", remainder_5: "i64[1, 12, 4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:512 in forward, code: value_vectors = self.value(hidden_states)
        reshape_default: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(mm_19, _shape_param_0);  mm_19 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:380 in _split_hidden_size_dim, code: x = x.view(*new_x_shape)
        reshape_default_1: "f16[1, 4096, 12, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:381 in _split_hidden_size_dim, code: return x.transpose(2, 1)
        permute_default: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1063 in _gather_by_expansion, code: vectors = vectors.repeat(1, 1, num_hashes, 1)
        repeat_default: "f16[1, 12, 4096, 64]" = torch.ops.aten.repeat.default(permute_default, [1, 1, 1, 1]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1062 in _gather_by_expansion, code: expanded_idxs = idxs.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
        unsqueeze_default: "i64[1, 12, 4096, 1]" = torch.ops.aten.unsqueeze.default(remainder_5, -1);  remainder_5 = None
        expand_default: "i64[1, 12, 4096, 64]" = torch.ops.aten.expand.default(unsqueeze_default, _shape_param_2);  unsqueeze_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1064 in _gather_by_expansion, code: return torch.gather(vectors, 2, expanded_idxs)
        gather_default: "f16[1, 12, 4096, 64]" = torch.ops.aten.gather.default(repeat_default, 2, expand_default);  repeat_default = expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        reshape_default_2: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(gather_default, _shape_param_3);  gather_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:372 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_tensor: "f16[1, 12, 1, 64, 64]" = torch.ops.aten.slice.Tensor(reshape_default_2, 2, -1, 9223372036854775807)
        slice_tensor_1: "f16[1, 12, 63, 64, 64]" = torch.ops.aten.slice.Tensor(reshape_default_2, 2, 0, -1)
        cat_default: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.cat.default([slice_tensor, slice_tensor_1], 2);  slice_tensor = slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:373 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_default_1: "f16[1, 12, 64, 128, 64]" = torch.ops.aten.cat.default([cat_default, reshape_default_2], 3);  cat_default = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:896 in _attend, code: out_vectors = torch.matmul(attention_probs, value_vectors)
        expand_default_1: "f16[1, 12, 64, 128, 64]" = torch.ops.aten.expand.default(cat_default_1, _shape_param_4);  cat_default_1 = _shape_param_4 = None
        reshape_default_3: "f16[768, 128, 64]" = torch.ops.aten.reshape.default(expand_default_1, _shape_param_5);  expand_default_1 = _shape_param_5 = None
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
