"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_infer
Pattern hash: 0ed414912696
Shape hash: 7df6f7e9
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg3_1: "f32[30000, 128]", arg0_1: "i64[8, 512]", arg2_1: "i64[1, 512]", arg1_1: "i64[1, 512]", arg4_1: "f32[2, 128]", arg5_1: "f32[512, 128]", arg6_1: "f32[128]", arg7_1: "f32[128]", arg8_1: "f32[4096, 128]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:101 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding_default: "f32[8, 512, 128]" = torch.ops.aten.embedding.default(arg3_1, arg0_1, 0);  arg3_1 = arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:94 in forward, code: buffered_token_type_ids = self.token_type_ids.expand(position_ids.shape[0], -1)
        expand_default: "i64[1, 512]" = torch.ops.aten.expand.default(arg2_1, [1, -1]);  arg2_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:95 in forward, code: buffered_token_type_ids = torch.gather(buffered_token_type_ids, dim=1, index=position_ids)
        gather_default: "i64[1, 512]" = torch.ops.aten.gather.default(expand_default, 1, arg1_1);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:96 in forward, code: token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_default_1: "i64[8, 512]" = torch.ops.aten.expand.default(gather_default, _shape_param_0);  gather_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:102 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_default_1: "f32[8, 512, 128]" = torch.ops.aten.embedding.default(arg4_1, expand_default_1);  arg4_1 = expand_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:103 in forward, code: embeddings = inputs_embeds + token_type_embeddings
        add_tensor: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:105 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_default_2: "f32[1, 512, 128]" = torch.ops.aten.embedding.default(arg5_1, arg1_1);  arg5_1 = arg1_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:106 in forward, code: embeddings = embeddings + position_embeddings
        add_tensor_1: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(add_tensor, embedding_default_2);  add_tensor = embedding_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:108 in forward, code: embeddings = self.LayerNorm(embeddings)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(add_tensor_1, getitem_1);  add_tensor_1 = getitem_1 = None
        add_tensor_2: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, arg6_1);  mul_tensor = arg6_1 = None
        add_tensor_3: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg7_1);  mul_tensor_1 = arg7_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:276 in forward, code: hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        reshape_default: "f32[4096, 128]" = torch.ops.aten.reshape.default(add_tensor_3, _shape_param_1);  add_tensor_3 = _shape_param_1 = None
        permute_default: "f32[128, 4096]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        return (reshape_default, permute_default)


def _default_make_inputs():
    return [
    torch.randn([30000, 128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 30000, [8, 512], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [1, 512], dtype=torch.int64, device='cuda'),
    torch.randint(0, 512, [1, 512], dtype=torch.int64, device='cuda'),
    torch.randn([2, 128], dtype=torch.float32, device='cuda'),
    torch.randn([512, 128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 128], dtype=torch.float32, device='cuda'),
    [8, 512],  # _shape_param_0
    [4096, 128],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
