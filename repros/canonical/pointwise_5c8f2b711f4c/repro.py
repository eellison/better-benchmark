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
    def forward(self, arg0_1: "i64[16, 512]", arg1_1: "f32[32000, 1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1238 in forward, code: input_ids = input_ids.transpose(0, 1).contiguous()
        permute_default: "i64[512, 16]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        clone_default: "i64[512, 16]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1307 in forward, code: word_emb_k = self.word_embedding(input_ids)
        embedding_default: "f32[512, 16, 1024]" = torch.ops.aten.embedding.default(arg1_1, clone_default);  arg1_1 = clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:416 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default: "f32[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(embedding_default, 3)
        unsqueeze_default_1: "f32[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 4);  unsqueeze_default = None
        reshape_default: "f32[1, 8192, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_1, _shape_param_0);  unsqueeze_default_1 = _shape_param_0 = None
        squeeze_dim: "f32[8192, 1024]" = torch.ops.aten.squeeze.dim(reshape_default, 0);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:417 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_default_2: "f32[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(embedding_default, 3)
        unsqueeze_default_3: "f32[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 4);  unsqueeze_default_2 = None
        reshape_default_1: "f32[1, 8192, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_3, _shape_param_1);  unsqueeze_default_3 = _shape_param_1 = None
        squeeze_dim_1: "f32[8192, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_1, 0);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:418 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        unsqueeze_default_4: "f32[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(embedding_default, 3)
        unsqueeze_default_5: "f32[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 4);  unsqueeze_default_4 = None
        reshape_default_2: "f32[1, 8192, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_5, _shape_param_2);  unsqueeze_default_5 = _shape_param_2 = None
        squeeze_dim_2: "f32[8192, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_2, 0);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1106 in cache_mem, code: new_mem = curr_out[cutoff:]
        slice_tensor: "f32[512, 16, 1024]" = torch.ops.aten.slice.Tensor(embedding_default, 0, -512, 9223372036854775807);  embedding_default = None
        return (squeeze_dim, squeeze_dim_1, squeeze_dim_2, slice_tensor)


def _default_make_inputs():
    return [
    torch.randint(0, 32000, [16, 512], dtype=torch.int64, device='cuda'),
    torch.randn([32000, 1024], dtype=torch.float32, device='cuda'),
    [1, 8192, 1024],  # _shape_param_0
    [1, 8192, 1024],  # _shape_param_1
    [1, 8192, 1024],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
