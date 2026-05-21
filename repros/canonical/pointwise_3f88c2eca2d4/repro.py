"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_infer
Pattern hash: 3f88c2eca2d4
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
_shapes_config = "(T([4096, 768], f16), S([1, 4096, 768]), S([1, 4096, 12, 64]), S([1, 12, 64, 64, 64]), S([1, 12, 64, 64, 128]), S([768, 64, 128]))"

class Repro(torch.nn.Module):
    def forward(self, mm_15: "f16[4096, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1163 in forward, code: key_vectors = self.key(hidden_states)
        reshape_default: "f16[1, 4096, 768]" = torch.ops.aten.reshape.default(mm_15, _shape_param_0);  mm_15 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:380 in _split_hidden_size_dim, code: x = x.view(*new_x_shape)
        reshape_default_1: "f16[1, 4096, 12, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:381 in _split_hidden_size_dim, code: return x.transpose(2, 1)
        permute_default: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1188 in forward, code: key_vectors = key_vectors / np.sqrt(self.attention_head_size)
        full_default: "f64[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_tensor: "f16[1, 12, 4096, 64]" = torch.ops.aten.div.Tensor(permute_default, full_default);  permute_default = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        reshape_default_2: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(div_tensor, _shape_param_2);  div_tensor = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:372 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_tensor: "f16[1, 12, 1, 64, 64]" = torch.ops.aten.slice.Tensor(reshape_default_2, 2, -1, 9223372036854775807)
        slice_tensor_1: "f16[1, 12, 63, 64, 64]" = torch.ops.aten.slice.Tensor(reshape_default_2, 2, 0, -1)
        cat_default: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.cat.default([slice_tensor, slice_tensor_1], 2);  slice_tensor = slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:373 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_default_1: "f16[1, 12, 64, 128, 64]" = torch.ops.aten.cat.default([cat_default, reshape_default_2], 3);  cat_default = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1236 in forward, code: query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        permute_default_1: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.permute.default(cat_default_1, [0, 1, 2, 4, 3]);  cat_default_1 = None
        expand_default: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.expand.default(permute_default_1, _shape_param_3);  permute_default_1 = _shape_param_3 = None
        reshape_default_3: "f16[768, 64, 128]" = torch.ops.aten.reshape.default(expand_default, _shape_param_4);  expand_default = _shape_param_4 = None
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
