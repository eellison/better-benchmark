"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Longformer_train
Pattern hash: 84e8764f321e
Shape hash: 7412e5df
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
_shapes_config = "(T([72, 512, 64], f32), T([1572864], f32), T([2359296], i64, gen=Index(1572864)), S([24, 3, 512, 64, 1]), S([24, 1024, 64]), S([2, 12, 1024, 64]), S([1024, 2, 768]), S([2048, 768]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_71: "f32[72, 512, 64]", full_default_121: "f32[1572864]", view_1433: "i64[2359296]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        reshape_default: "f32[24, 3, 512, 64, 1]" = torch.ops.aten.reshape.default(bmm_71, _shape_param_0);  bmm_71 = _shape_param_0 = None
        squeeze_dim: "f32[24, 3, 512, 64]" = torch.ops.aten.squeeze.dim(reshape_default, 4);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        reshape_default_1: "f32[2359296]" = torch.ops.aten.reshape.default(squeeze_dim, [-1]);  squeeze_dim = None
        index_put_default: "f32[1572864]" = torch.ops.aten.index_put.default(full_default_121, [view_1433], reshape_default_1, True);  full_default_121 = view_1433 = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:515 in forward, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_default: "f32[24, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_default, [24, 2, 512, 64], [64, 786432, 1536, 1], 0);  index_put_default = None
        reshape_default_2: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(as_strided_default, _shape_param_1);  as_strided_default = _shape_param_1 = None
        reshape_default_3: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_2);  reshape_default_2 = _shape_param_2 = None
        permute_default: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None
        permute_default_1: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_default, [1, 0, 2, 3]);  permute_default = None
        reshape_default_4: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(permute_default_1, _shape_param_3);  permute_default_1 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_tensor: "f32[1024, 2, 768]" = torch.ops.aten.div.Tensor(reshape_default_4, 8.0);  reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        reshape_default_5: "f32[2048, 768]" = torch.ops.aten.reshape.default(div_tensor, _shape_param_4);  div_tensor = _shape_param_4 = None
        return reshape_default_5



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
