"""
Standalone repro captured via capture_hook.
Label: hf_t5_base_train
Pattern hash: 8b424d3b29c5
Shape hash: f1adc8c8
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_71: "f32[2048, 768]", _shape_param_0, inductor_seeds_default: "i64[50]", add_64: "f32[4, 512, 768]", primals_100: "f32[768]", relu_11: "f32[4, 512, 3072]", view_291: "f32[48, 512, 512]", view_292: "f32[48, 512, 64]", view_285: "f32[48, 512, 64]", view_286: "f32[48, 64, 512]", relu_10: "f32[4, 512, 3072]", view_266: "f32[48, 512, 512]", view_267: "f32[48, 512, 64]", view_260: "f32[48, 512, 64]", view_261: "f32[48, 64, 512]", relu_9: "f32[4, 512, 3072]", view_241: "f32[48, 512, 512]", view_242: "f32[48, 512, 64]", view_235: "f32[48, 512, 64]", view_236: "f32[48, 64, 512]", relu_8: "f32[4, 512, 3072]", view_216: "f32[48, 512, 512]", view_217: "f32[48, 512, 64]", view_210: "f32[48, 512, 64]", view_211: "f32[48, 64, 512]", relu_7: "f32[4, 512, 3072]", view_191: "f32[48, 512, 512]", view_192: "f32[48, 512, 64]", view_185: "f32[48, 512, 64]", view_186: "f32[48, 64, 512]", relu_6: "f32[4, 512, 3072]", view_166: "f32[48, 512, 512]", view_167: "f32[48, 512, 64]", view_160: "f32[48, 512, 64]", view_161: "f32[48, 64, 512]", relu_5: "f32[4, 512, 3072]", view_141: "f32[48, 512, 512]", view_142: "f32[48, 512, 64]", view_135: "f32[48, 512, 64]", view_136: "f32[48, 64, 512]", relu_4: "f32[4, 512, 3072]", view_116: "f32[48, 512, 512]", view_117: "f32[48, 512, 64]", view_110: "f32[48, 512, 64]", view_111: "f32[48, 64, 512]", relu_3: "f32[4, 512, 3072]", view_91: "f32[48, 512, 512]", view_92: "f32[48, 512, 64]", view_85: "f32[48, 512, 64]", view_86: "f32[48, 64, 512]", relu_2: "f32[4, 512, 3072]", view_66: "f32[48, 512, 512]", view_67: "f32[48, 512, 64]", view_60: "f32[48, 512, 64]", view_61: "f32[48, 64, 512]", relu_1: "f32[4, 512, 3072]", view_41: "f32[48, 512, 512]", view_42: "f32[48, 512, 64]", view_35: "f32[48, 512, 64]", view_36: "f32[48, 64, 512]", relu: "f32[4, 512, 3072]", view_16: "f32[48, 512, 512]", view_17: "f32[48, 512, 64]", view_10: "f32[48, 512, 64]", view_11: "f32[48, 64, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        reshape_default: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_71, _shape_param_0);  mm_71 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 48)
        inductor_random_default: "f32[4, 512, 768]" = torch.ops.prims.inductor_random.default([4, 512, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[4, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(gt_scalar, reshape_default);  gt_scalar = reshape_default = None
        mul_tensor_1: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        add_tensor: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_64, mul_tensor_1);  add_64 = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar: "f32[4, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor, 2)
        mean_dim: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor_1: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_2: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor, rsqrt_default);  add_tensor = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_3: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(primals_100, mul_tensor_2);  primals_100 = mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:755 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 49);  inductor_seeds_default = None
        inductor_random_default_1: "f32[4, 512, 768]" = torch.ops.prims.inductor_random.default([4, 512, 768], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_scalar_1: "b8[4, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.1);  inductor_random_default_1 = None
        mul_tensor_4: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(gt_scalar_1, mul_tensor_3);  gt_scalar_1 = mul_tensor_3 = None
        mul_tensor_5: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_4, 1.1111111111111112);  mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar: "b8[4, 512, 3072]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_291, [0, 2, 1]);  view_291 = None
        permute_default_1: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_292, [0, 2, 1]);  view_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_2: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_285, [0, 2, 1]);  view_285 = None
        permute_default_3: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_286, [0, 2, 1]);  view_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_1: "b8[4, 512, 3072]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_4: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_266, [0, 2, 1]);  view_266 = None
        permute_default_5: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_267, [0, 2, 1]);  view_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_6: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_260, [0, 2, 1]);  view_260 = None
        permute_default_7: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_261, [0, 2, 1]);  view_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_2: "b8[4, 512, 3072]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_8: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_241, [0, 2, 1]);  view_241 = None
        permute_default_9: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_242, [0, 2, 1]);  view_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_10: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_235, [0, 2, 1]);  view_235 = None
        permute_default_11: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_236, [0, 2, 1]);  view_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_3: "b8[4, 512, 3072]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_12: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_216, [0, 2, 1]);  view_216 = None
        permute_default_13: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_217, [0, 2, 1]);  view_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_14: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
        permute_default_15: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_4: "b8[4, 512, 3072]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_16: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_191, [0, 2, 1]);  view_191 = None
        permute_default_17: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_192, [0, 2, 1]);  view_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_18: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_185, [0, 2, 1]);  view_185 = None
        permute_default_19: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1]);  view_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_5: "b8[4, 512, 3072]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_20: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
        permute_default_21: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_22: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_160, [0, 2, 1]);  view_160 = None
        permute_default_23: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_161, [0, 2, 1]);  view_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_6: "b8[4, 512, 3072]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_24: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
        permute_default_25: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_142, [0, 2, 1]);  view_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_26: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_135, [0, 2, 1]);  view_135 = None
        permute_default_27: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1]);  view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_7: "b8[4, 512, 3072]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_28: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_116, [0, 2, 1]);  view_116 = None
        permute_default_29: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_117, [0, 2, 1]);  view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_30: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_110, [0, 2, 1]);  view_110 = None
        permute_default_31: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_111, [0, 2, 1]);  view_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_8: "b8[4, 512, 3072]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_32: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_91, [0, 2, 1]);  view_91 = None
        permute_default_33: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_92, [0, 2, 1]);  view_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_34: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_85, [0, 2, 1]);  view_85 = None
        permute_default_35: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_86, [0, 2, 1]);  view_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_9: "b8[4, 512, 3072]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_36: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_66, [0, 2, 1]);  view_66 = None
        permute_default_37: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_67, [0, 2, 1]);  view_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_38: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_60, [0, 2, 1]);  view_60 = None
        permute_default_39: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_61, [0, 2, 1]);  view_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_10: "b8[4, 512, 3072]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_40: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_41, [0, 2, 1]);  view_41 = None
        permute_default_41: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_42, [0, 2, 1]);  view_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_42: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None
        permute_default_43: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_36, [0, 2, 1]);  view_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_11: "b8[4, 512, 3072]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_44: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_16, [0, 2, 1]);  view_16 = None
        permute_default_45: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_17, [0, 2, 1]);  view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_46: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
        permute_default_47: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
        return (mul_tensor_5, le_scalar, permute_default, permute_default_1, permute_default_2, permute_default_3, le_scalar_1, permute_default_4, permute_default_5, permute_default_6, permute_default_7, le_scalar_2, permute_default_8, permute_default_9, permute_default_10, permute_default_11, le_scalar_3, permute_default_12, permute_default_13, permute_default_14, permute_default_15, le_scalar_4, permute_default_16, permute_default_17, permute_default_18, permute_default_19, le_scalar_5, permute_default_20, permute_default_21, permute_default_22, permute_default_23, le_scalar_6, permute_default_24, permute_default_25, permute_default_26, permute_default_27, le_scalar_7, permute_default_28, permute_default_29, permute_default_30, permute_default_31, le_scalar_8, permute_default_32, permute_default_33, permute_default_34, permute_default_35, le_scalar_9, permute_default_36, permute_default_37, permute_default_38, permute_default_39, le_scalar_10, permute_default_40, permute_default_41, permute_default_42, permute_default_43, le_scalar_11, permute_default_44, permute_default_45, permute_default_46, permute_default_47)


def _default_make_inputs():
    return [
    torch.randn([2048, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_0
    torch.randint(0, 2, [50], dtype=torch.int64, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
