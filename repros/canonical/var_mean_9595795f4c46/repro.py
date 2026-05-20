"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train
Pattern hash: 9595795f4c46
Shape hash: a92bfc00
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([8192, 768], f32), T([36], i64, max=36), T([8, 1024, 768], f32), T([768], f32), T([768], f32), T([8, 1024, 1], f32), T([384, 256, 768], f32, stride=(197120, 769, 1)), T([384, 768, 64], f32), T([288, 512, 64], f32), T([288, 64, 512], f32), T([8, 1024, 1], f32), T([8, 1024, 1], f32), T([384, 256, 768], f32, stride=(197120, 769, 1)), T([384, 768, 64], f32), T([288, 512, 64], f32), T([288, 64, 512], f32), T([8, 1024, 1], f32), T([8, 1024, 1], f32), T([384, 256, 768], f32, stride=(197120, 769, 1)), T([384, 768, 64], f32), T([288, 512, 64], f32), T([288, 64, 512], f32), T([8, 1024, 1], f32), T([8, 1024, 1], f32), T([384, 256, 768], f32, stride=(197120, 769, 1)), T([384, 768, 64], f32), T([288, 512, 64], f32), T([288, 64, 512], f32), T([8, 1024, 1], f32), T([8, 1024, 1], f32), T([384, 256, 768], f32, stride=(197120, 769, 1)), T([384, 768, 64], f32), T([288, 512, 64], f32), T([288, 64, 512], f32), T([8, 1024, 1], f32), T([8, 1024, 1], f32), T([384, 256, 768], f32, stride=(197120, 769, 1)), T([384, 768, 64], f32), T([288, 512, 64], f32), T([288, 64, 512], f32), T([8, 1024, 1], f32), T([8, 1024, 1], f32), T([384, 256, 768], f32, stride=(197120, 769, 1)), T([384, 768, 64], f32), T([288, 512, 64], f32), T([288, 64, 512], f32), T([8, 1024, 1], f32), T([8, 1024, 1], f32), T([384, 256, 768], f32, stride=(197120, 769, 1)), T([384, 768, 64], f32), T([288, 512, 64], f32), T([288, 64, 512], f32), T([8, 1024, 1], f32), T([8, 1024, 1], f32), T([384, 256, 768], f32, stride=(197120, 769, 1)), T([384, 768, 64], f32), T([288, 512, 64], f32), T([288, 64, 512], f32), T([8, 1024, 1], f32), T([8, 1024, 1], f32), T([384, 256, 768], f32, stride=(197120, 769, 1)), T([384, 768, 64], f32), T([288, 512, 64], f32), T([288, 64, 512], f32), T([8, 1024, 1], f32), T([8, 1024, 1], f32), T([384, 256, 768], f32, stride=(197120, 769, 1)), T([384, 768, 64], f32), T([288, 512, 64], f32), T([288, 64, 512], f32), T([8, 1024, 1], f32), T([8, 1024, 1], f32), T([384, 256, 768], f32, stride=(197120, 769, 1)), T([384, 768, 64], f32), T([288, 512, 64], f32), T([288, 64, 512], f32), S([8, 1024, 768]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_23: "f32[8192, 768]", inductor_seeds_default: "i64[36]", add_175: "f32[8, 1024, 768]", primals_194: "f32[768]", primals_195: "f32[768]", rsqrt_22: "f32[8, 1024, 1]", view_1368: "f32[384, 256, 768]", view_1369: "f32[384, 768, 64]", view_1297: "f32[288, 512, 64]", view_1298: "f32[288, 64, 512]", rsqrt_21: "f32[8, 1024, 1]", rsqrt_20: "f32[8, 1024, 1]", view_1253: "f32[384, 256, 768]", view_1254: "f32[384, 768, 64]", view_1182: "f32[288, 512, 64]", view_1183: "f32[288, 64, 512]", rsqrt_19: "f32[8, 1024, 1]", rsqrt_18: "f32[8, 1024, 1]", view_1138: "f32[384, 256, 768]", view_1139: "f32[384, 768, 64]", view_1067: "f32[288, 512, 64]", view_1068: "f32[288, 64, 512]", rsqrt_17: "f32[8, 1024, 1]", rsqrt_16: "f32[8, 1024, 1]", view_1023: "f32[384, 256, 768]", view_1024: "f32[384, 768, 64]", view_952: "f32[288, 512, 64]", view_953: "f32[288, 64, 512]", rsqrt_15: "f32[8, 1024, 1]", rsqrt_14: "f32[8, 1024, 1]", view_908: "f32[384, 256, 768]", view_909: "f32[384, 768, 64]", view_837: "f32[288, 512, 64]", view_838: "f32[288, 64, 512]", rsqrt_13: "f32[8, 1024, 1]", rsqrt_12: "f32[8, 1024, 1]", view_793: "f32[384, 256, 768]", view_794: "f32[384, 768, 64]", view_722: "f32[288, 512, 64]", view_723: "f32[288, 64, 512]", rsqrt_11: "f32[8, 1024, 1]", rsqrt_10: "f32[8, 1024, 1]", view_678: "f32[384, 256, 768]", view_679: "f32[384, 768, 64]", view_607: "f32[288, 512, 64]", view_608: "f32[288, 64, 512]", rsqrt_9: "f32[8, 1024, 1]", rsqrt_8: "f32[8, 1024, 1]", view_563: "f32[384, 256, 768]", view_564: "f32[384, 768, 64]", view_492: "f32[288, 512, 64]", view_493: "f32[288, 64, 512]", rsqrt_7: "f32[8, 1024, 1]", rsqrt_6: "f32[8, 1024, 1]", view_448: "f32[384, 256, 768]", view_449: "f32[384, 768, 64]", view_377: "f32[288, 512, 64]", view_378: "f32[288, 64, 512]", rsqrt_5: "f32[8, 1024, 1]", rsqrt_4: "f32[8, 1024, 1]", view_333: "f32[384, 256, 768]", view_334: "f32[384, 768, 64]", view_262: "f32[288, 512, 64]", view_263: "f32[288, 64, 512]", rsqrt_3: "f32[8, 1024, 1]", rsqrt_2: "f32[8, 1024, 1]", view_218: "f32[384, 256, 768]", view_219: "f32[384, 768, 64]", view_147: "f32[288, 512, 64]", view_148: "f32[288, 64, 512]", rsqrt_1: "f32[8, 1024, 1]", rsqrt: "f32[8, 1024, 1]", view_103: "f32[384, 256, 768]", view_104: "f32[384, 768, 64]", view_32: "f32[288, 512, 64]", view_33: "f32[288, 64, 512]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(addmm_23, _shape_param_0);  addmm_23 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 35);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 1e-30);  inductor_random_default = None
        mul_tensor: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_scalar, reshape_default);  gt_scalar = reshape_default = None
        mul_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.0);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_tensor: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, add_175);  mul_tensor_1 = add_175 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1024, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 1024, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, primals_194);  mul_tensor_2 = primals_194 = None
        add_tensor_2: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_tensor_3, primals_195);  mul_tensor_3 = primals_195 = None
        div_tensor: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_default, 768);  rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_1: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        permute_default: "f32[384, 768, 256]" = torch.ops.aten.permute.default(view_1368, [0, 2, 1]);  view_1368 = None
        permute_default_1: "f32[384, 64, 768]" = torch.ops.aten.permute.default(view_1369, [0, 2, 1]);  view_1369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        permute_default_2: "f32[288, 64, 512]" = torch.ops.aten.permute.default(view_1297, [0, 2, 1]);  view_1297 = None
        permute_default_3: "f32[288, 512, 64]" = torch.ops.aten.permute.default(view_1298, [0, 2, 1]);  view_1298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_2: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_3: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        permute_default_4: "f32[384, 768, 256]" = torch.ops.aten.permute.default(view_1253, [0, 2, 1]);  view_1253 = None
        permute_default_5: "f32[384, 64, 768]" = torch.ops.aten.permute.default(view_1254, [0, 2, 1]);  view_1254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        permute_default_6: "f32[288, 64, 512]" = torch.ops.aten.permute.default(view_1182, [0, 2, 1]);  view_1182 = None
        permute_default_7: "f32[288, 512, 64]" = torch.ops.aten.permute.default(view_1183, [0, 2, 1]);  view_1183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_4: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_5: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        permute_default_8: "f32[384, 768, 256]" = torch.ops.aten.permute.default(view_1138, [0, 2, 1]);  view_1138 = None
        permute_default_9: "f32[384, 64, 768]" = torch.ops.aten.permute.default(view_1139, [0, 2, 1]);  view_1139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        permute_default_10: "f32[288, 64, 512]" = torch.ops.aten.permute.default(view_1067, [0, 2, 1]);  view_1067 = None
        permute_default_11: "f32[288, 512, 64]" = torch.ops.aten.permute.default(view_1068, [0, 2, 1]);  view_1068 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_6: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_7: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        permute_default_12: "f32[384, 768, 256]" = torch.ops.aten.permute.default(view_1023, [0, 2, 1]);  view_1023 = None
        permute_default_13: "f32[384, 64, 768]" = torch.ops.aten.permute.default(view_1024, [0, 2, 1]);  view_1024 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        permute_default_14: "f32[288, 64, 512]" = torch.ops.aten.permute.default(view_952, [0, 2, 1]);  view_952 = None
        permute_default_15: "f32[288, 512, 64]" = torch.ops.aten.permute.default(view_953, [0, 2, 1]);  view_953 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_8: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_9: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        permute_default_16: "f32[384, 768, 256]" = torch.ops.aten.permute.default(view_908, [0, 2, 1]);  view_908 = None
        permute_default_17: "f32[384, 64, 768]" = torch.ops.aten.permute.default(view_909, [0, 2, 1]);  view_909 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        permute_default_18: "f32[288, 64, 512]" = torch.ops.aten.permute.default(view_837, [0, 2, 1]);  view_837 = None
        permute_default_19: "f32[288, 512, 64]" = torch.ops.aten.permute.default(view_838, [0, 2, 1]);  view_838 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_10: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_11: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        permute_default_20: "f32[384, 768, 256]" = torch.ops.aten.permute.default(view_793, [0, 2, 1]);  view_793 = None
        permute_default_21: "f32[384, 64, 768]" = torch.ops.aten.permute.default(view_794, [0, 2, 1]);  view_794 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        permute_default_22: "f32[288, 64, 512]" = torch.ops.aten.permute.default(view_722, [0, 2, 1]);  view_722 = None
        permute_default_23: "f32[288, 512, 64]" = torch.ops.aten.permute.default(view_723, [0, 2, 1]);  view_723 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_12: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_13: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        permute_default_24: "f32[384, 768, 256]" = torch.ops.aten.permute.default(view_678, [0, 2, 1]);  view_678 = None
        permute_default_25: "f32[384, 64, 768]" = torch.ops.aten.permute.default(view_679, [0, 2, 1]);  view_679 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        permute_default_26: "f32[288, 64, 512]" = torch.ops.aten.permute.default(view_607, [0, 2, 1]);  view_607 = None
        permute_default_27: "f32[288, 512, 64]" = torch.ops.aten.permute.default(view_608, [0, 2, 1]);  view_608 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_14: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_15: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        permute_default_28: "f32[384, 768, 256]" = torch.ops.aten.permute.default(view_563, [0, 2, 1]);  view_563 = None
        permute_default_29: "f32[384, 64, 768]" = torch.ops.aten.permute.default(view_564, [0, 2, 1]);  view_564 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        permute_default_30: "f32[288, 64, 512]" = torch.ops.aten.permute.default(view_492, [0, 2, 1]);  view_492 = None
        permute_default_31: "f32[288, 512, 64]" = torch.ops.aten.permute.default(view_493, [0, 2, 1]);  view_493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_16: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_17: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        permute_default_32: "f32[384, 768, 256]" = torch.ops.aten.permute.default(view_448, [0, 2, 1]);  view_448 = None
        permute_default_33: "f32[384, 64, 768]" = torch.ops.aten.permute.default(view_449, [0, 2, 1]);  view_449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        permute_default_34: "f32[288, 64, 512]" = torch.ops.aten.permute.default(view_377, [0, 2, 1]);  view_377 = None
        permute_default_35: "f32[288, 512, 64]" = torch.ops.aten.permute.default(view_378, [0, 2, 1]);  view_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_18: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_19: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        permute_default_36: "f32[384, 768, 256]" = torch.ops.aten.permute.default(view_333, [0, 2, 1]);  view_333 = None
        permute_default_37: "f32[384, 64, 768]" = torch.ops.aten.permute.default(view_334, [0, 2, 1]);  view_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        permute_default_38: "f32[288, 64, 512]" = torch.ops.aten.permute.default(view_262, [0, 2, 1]);  view_262 = None
        permute_default_39: "f32[288, 512, 64]" = torch.ops.aten.permute.default(view_263, [0, 2, 1]);  view_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_20: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_21: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        permute_default_40: "f32[384, 768, 256]" = torch.ops.aten.permute.default(view_218, [0, 2, 1]);  view_218 = None
        permute_default_41: "f32[384, 64, 768]" = torch.ops.aten.permute.default(view_219, [0, 2, 1]);  view_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        permute_default_42: "f32[288, 64, 512]" = torch.ops.aten.permute.default(view_147, [0, 2, 1]);  view_147 = None
        permute_default_43: "f32[288, 512, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1]);  view_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_22: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_23: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        permute_default_44: "f32[384, 768, 256]" = torch.ops.aten.permute.default(view_103, [0, 2, 1]);  view_103 = None
        permute_default_45: "f32[384, 64, 768]" = torch.ops.aten.permute.default(view_104, [0, 2, 1]);  view_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        permute_default_46: "f32[288, 64, 512]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None
        permute_default_47: "f32[288, 512, 64]" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
        return (add_tensor_2, div_tensor, div_tensor_1, permute_default, permute_default_1, permute_default_2, permute_default_3, div_tensor_2, div_tensor_3, permute_default_4, permute_default_5, permute_default_6, permute_default_7, div_tensor_4, div_tensor_5, permute_default_8, permute_default_9, permute_default_10, permute_default_11, div_tensor_6, div_tensor_7, permute_default_12, permute_default_13, permute_default_14, permute_default_15, div_tensor_8, div_tensor_9, permute_default_16, permute_default_17, permute_default_18, permute_default_19, div_tensor_10, div_tensor_11, permute_default_20, permute_default_21, permute_default_22, permute_default_23, div_tensor_12, div_tensor_13, permute_default_24, permute_default_25, permute_default_26, permute_default_27, div_tensor_14, div_tensor_15, permute_default_28, permute_default_29, permute_default_30, permute_default_31, div_tensor_16, div_tensor_17, permute_default_32, permute_default_33, permute_default_34, permute_default_35, div_tensor_18, div_tensor_19, permute_default_36, permute_default_37, permute_default_38, permute_default_39, div_tensor_20, div_tensor_21, permute_default_40, permute_default_41, permute_default_42, permute_default_43, div_tensor_22, div_tensor_23, permute_default_44, permute_default_45, permute_default_46, permute_default_47)


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
