"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Longformer_infer
Pattern hash: a1cfcccec3c9
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
_shapes_config = "(T([4096, 768], f16), S([4096, 1, 768]), S([4096, 1, 12, 64]), S([12, 4096, 64]), S([12, 8, 512, 64]), S([180, 512, 64]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_66: "f16[4096, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        reshape_default: "f16[4096, 1, 768]" = torch.ops.aten.reshape.default(addmm_66, _shape_param_0);  addmm_66 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_tensor: "f16[4096, 1, 768]" = torch.ops.aten.div.Tensor(reshape_default, 8.0);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        reshape_default_1: "f16[4096, 1, 12, 64]" = torch.ops.aten.reshape.default(div_tensor, _shape_param_1);  div_tensor = _shape_param_1 = None
        permute_default: "f16[1, 4096, 12, 64]" = torch.ops.aten.permute.default(reshape_default_1, [1, 0, 2, 3]);  reshape_default_1 = None
        permute_default_1: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        reshape_default_2: "f16[12, 4096, 64]" = torch.ops.aten.reshape.default(permute_default_1, _shape_param_2);  permute_default_1 = _shape_param_2 = None
        reshape_default_3: "f16[12, 8, 512, 64]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        as_strided_default: "f16[12, 15, 512, 64]" = torch.ops.aten.as_strided.default(reshape_default_3, [12, 15, 512, 64], [64, 196608, 768, 1]);  reshape_default_3 = None
        unsqueeze_default: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_default, 4);  as_strided_default = None
        clone_default: "f16[12, 15, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_default, memory_format = torch.contiguous_format);  unsqueeze_default = None
        reshape_default_4: "f16[180, 512, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None
        return reshape_default_4



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
