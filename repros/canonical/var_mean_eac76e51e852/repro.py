"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=var_mean, ranges=[], reduction_ranges=[]
#   origins: ['aten.var_mean.correction']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg3_1: "f32[30522, 128]", arg1_1: "i64[64, 512]", arg2_1: "i64[1, 512]", arg5_1: "f32[2, 128]", arg6_1: "f32[512, 128]", arg4_1: "i64[1, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding_default: "f32[64, 512, 128]" = torch.ops.aten.embedding.default(arg3_1, arg1_1, 0);  arg3_1 = arg1_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:758 in forward, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_default: "i64[64, 512]" = torch.ops.aten.expand.default(arg2_1, [64, 512]);  arg2_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_default_1: "f32[64, 512, 128]" = torch.ops.aten.embedding.default(arg5_1, expand_default);  arg5_1 = expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:189 in forward, code: embeddings = inputs_embeds + token_type_embeddings
        add_tensor: "f32[64, 512, 128]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:191 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_default_2: "f32[1, 512, 128]" = torch.ops.aten.embedding.default(arg6_1, arg4_1);  arg6_1 = arg4_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:192 in forward, code: embeddings += position_embeddings
        add_tensor_1: "f32[64, 512, 128]" = torch.ops.aten.add.Tensor(add_tensor, embedding_default_2);  add_tensor = embedding_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:193 in forward, code: embeddings = self.LayerNorm(embeddings)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True);  add_tensor_1 = None
        getitem: "f32[64, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[64, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([30522, 128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [64, 512], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [1, 512], dtype=torch.int64, device='cuda'),
    torch.randn([2, 128], dtype=torch.float32, device='cuda'),
    torch.randn([512, 128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [1, 512], dtype=torch.int64, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
