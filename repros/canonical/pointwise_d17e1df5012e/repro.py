"""
Standalone repro captured via capture_hook.
Label: hf_XLNetLMHeadModel_inference
Pattern hash: d17e1df5012e
Shape hash: 99874ed5
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[8, 512]", arg1_1: "f32[32000, 1024]", _shape_param_0, arg2_1: "f32[1024, 16, 64]", _shape_param_1, _shape_param_2, arg3_1: "f32[1024, 16, 64]", _shape_param_3, _shape_param_4, _shape_param_5, arg5_1: "f32[1024, 16, 64]", _shape_param_6):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1046 in forward, code: input_ids = input_ids.transpose(0, 1).contiguous()
        permute_default: "i64[512, 8]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        clone_default: "i64[512, 8]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1115 in forward, code: word_emb_k = self.word_embedding(input_ids)
        embedding_default: "f32[512, 8, 1024]" = torch.ops.aten.embedding.default(arg1_1, clone_default);  arg1_1 = clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default: "f32[512, 8, 1024, 1]" = torch.ops.aten.unsqueeze.default(embedding_default, 3)
        unsqueeze_default_1: "f32[512, 8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 4);  unsqueeze_default = None
        reshape_default: "f32[1, 4096, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_1, _shape_param_0);  unsqueeze_default_1 = _shape_param_0 = None
        squeeze_dim: "f32[4096, 1024]" = torch.ops.aten.squeeze.dim(reshape_default, 0);  reshape_default = None
        unsqueeze_default_2: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, 3);  arg2_1 = None
        unsqueeze_default_3: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 4);  unsqueeze_default_2 = None
        reshape_default_1: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_3, _shape_param_1);  unsqueeze_default_3 = _shape_param_1 = None
        squeeze_dim_1: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_1, 0);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_default_4: "f32[512, 8, 1024, 1]" = torch.ops.aten.unsqueeze.default(embedding_default, 3);  embedding_default = None
        unsqueeze_default_5: "f32[512, 8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 4);  unsqueeze_default_4 = None
        reshape_default_2: "f32[1, 4096, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_5, _shape_param_2);  unsqueeze_default_5 = _shape_param_2 = None
        squeeze_dim_2: "f32[4096, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_2, 0);  reshape_default_2 = None
        unsqueeze_default_6: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, 3);  arg3_1 = None
        unsqueeze_default_7: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 4);  unsqueeze_default_6 = None
        reshape_default_3: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_7, _shape_param_3);  unsqueeze_default_7 = _shape_param_3 = None
        squeeze_dim_3: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_3, 0);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:971 in relative_positional_encoding, code: fwd_pos_seq = torch.arange(beg, end, -1.0, dtype=torch.int64, device=device).float()
        iota_default: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 512, step = -1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_default: "f32[1024]" = torch.ops.prims.convert_element_type.default(iota_default, torch.float32);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:931 in positional_embedding, code: sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq)
        unsqueeze_default_8: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default, 1);  convert_element_type_default = None
        permute_default_1: "f32[1024, 1]" = torch.ops.aten.permute.default(unsqueeze_default_8, [0, 1]);  unsqueeze_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:942 in relative_positional_encoding, code: freq_seq = torch.arange(0, self.d_model, 2.0, dtype=torch.int64, device=device).float()
        iota_default_1: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_default_1: "f32[512]" = torch.ops.prims.convert_element_type.default(iota_default_1, torch.float32);  iota_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:943 in relative_positional_encoding, code: inv_freq = 1 / torch.pow(10000, (freq_seq / self.d_model))
        div_tensor: "f32[512]" = torch.ops.aten.div.Tensor(convert_element_type_default_1, 1024);  convert_element_type_default_1 = None
        pow_scalar: "f32[512]" = torch.ops.aten.pow.Scalar(10000, div_tensor);  div_tensor = None
        reciprocal_default: "f32[512]" = torch.ops.aten.reciprocal.default(pow_scalar);  pow_scalar = None
        mul_tensor: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:931 in positional_embedding, code: sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq)
        unsqueeze_default_9: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, 1);  mul_tensor = None
        permute_default_2: "f32[1, 512]" = torch.ops.aten.permute.default(unsqueeze_default_9, [1, 0]);  unsqueeze_default_9 = None
        mul_tensor_1: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(permute_default_1, permute_default_2);  permute_default_1 = permute_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:932 in positional_embedding, code: pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        sin_default: "f32[1024, 512]" = torch.ops.aten.sin.default(mul_tensor_1)
        cos_default: "f32[1024, 512]" = torch.ops.aten.cos.default(mul_tensor_1);  mul_tensor_1 = None
        cat_default: "f32[1024, 1024]" = torch.ops.aten.cat.default([sin_default, cos_default], -1);  sin_default = cos_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:933 in positional_embedding, code: pos_emb = pos_emb[:, None, :]
        unsqueeze_default_10: "f32[1024, 1, 1024]" = torch.ops.aten.unsqueeze.default(cat_default, 1);  cat_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:936 in positional_embedding, code: pos_emb = pos_emb.expand(-1, bsz, -1)
        expand_default: "f32[1024, 8, 1024]" = torch.ops.aten.expand.default(unsqueeze_default_10, _shape_param_4);  unsqueeze_default_10 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1143 in forward, code: pos_emb = self.dropout(pos_emb)
        clone_default_1: "f32[1024, 8, 1024]" = torch.ops.aten.clone.default(expand_default);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        unsqueeze_default_11: "f32[1024, 8, 1024, 1]" = torch.ops.aten.unsqueeze.default(clone_default_1, 3);  clone_default_1 = None
        unsqueeze_default_12: "f32[1024, 8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 4);  unsqueeze_default_11 = None
        reshape_default_4: "f32[1, 8192, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_12, _shape_param_5);  unsqueeze_default_12 = _shape_param_5 = None
        squeeze_dim_4: "f32[8192, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_4, 0);  reshape_default_4 = None
        unsqueeze_default_13: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, 3);  arg5_1 = None
        unsqueeze_default_14: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 4);  unsqueeze_default_13 = None
        reshape_default_5: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_14, _shape_param_6);  unsqueeze_default_14 = _shape_param_6 = None
        squeeze_dim_5: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_5, 0);  reshape_default_5 = None
        return (squeeze_dim, squeeze_dim_1, squeeze_dim_2, squeeze_dim_3, squeeze_dim_4, squeeze_dim_5)


def _default_make_inputs():
    return [
    torch.randint(0, 2, [8, 512], dtype=torch.int64, device='cuda'),
    torch.randn([32000, 1024], dtype=torch.float32, device='cuda'),
    [1, 4096, 1024],  # _shape_param_0
    torch.randn([1024, 16, 64], dtype=torch.float32, device='cuda'),
    [1, 1024, 1024],  # _shape_param_1
    [1, 4096, 1024],  # _shape_param_2
    torch.randn([1024, 16, 64], dtype=torch.float32, device='cuda'),
    [1, 1024, 1024],  # _shape_param_3
    [-1, 8, -1],  # _shape_param_4
    [1, 8192, 1024],  # _shape_param_5
    torch.randn([1024, 16, 64], dtype=torch.float32, device='cuda'),
    [1, 1024, 1024],  # _shape_param_6
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
