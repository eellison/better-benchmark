"""
Standalone repro captured via capture_hook.
Label: hf_t5_base_train
Pattern hash: 89793f1853c4
Shape hash: 8ec4462e
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_119: "f32[512, 768]", _shape_param_0, inductor_seeds_default: "i64[74]", add_102: "f32[4, 128, 768]", primals_160: "f32[768]", relu_11: "f32[4, 128, 3072]", view_542: "f32[48, 128, 512]", view_543: "f32[48, 512, 64]", view_536: "f32[48, 128, 64]", view_537: "f32[48, 64, 512]", view_521: "f32[48, 128, 128]", view_522: "f32[48, 128, 64]", view_515: "f32[48, 128, 64]", view_516: "f32[48, 64, 128]", relu_10: "f32[4, 128, 3072]", view_496: "f32[48, 128, 512]", view_497: "f32[48, 512, 64]", view_490: "f32[48, 128, 64]", view_491: "f32[48, 64, 512]", view_475: "f32[48, 128, 128]", view_476: "f32[48, 128, 64]", view_469: "f32[48, 128, 64]", view_470: "f32[48, 64, 128]", relu_9: "f32[4, 128, 3072]", view_450: "f32[48, 128, 512]", view_451: "f32[48, 512, 64]", view_444: "f32[48, 128, 64]", view_445: "f32[48, 64, 512]", view_429: "f32[48, 128, 128]", view_430: "f32[48, 128, 64]", view_423: "f32[48, 128, 64]", view_424: "f32[48, 64, 128]", relu_8: "f32[4, 128, 3072]", view_404: "f32[48, 128, 512]", view_405: "f32[48, 512, 64]", view_398: "f32[48, 128, 64]", view_399: "f32[48, 64, 512]", view_383: "f32[48, 128, 128]", view_384: "f32[48, 128, 64]", view_377: "f32[48, 128, 64]", view_378: "f32[48, 64, 128]", relu_7: "f32[4, 128, 3072]", view_358: "f32[48, 128, 512]", view_359: "f32[48, 512, 64]", view_352: "f32[48, 128, 64]", view_353: "f32[48, 64, 512]", view_337: "f32[48, 128, 128]", view_338: "f32[48, 128, 64]", view_331: "f32[48, 128, 64]", view_332: "f32[48, 64, 128]", relu_6: "f32[4, 128, 3072]", view_312: "f32[48, 128, 512]", view_313: "f32[48, 512, 64]", view_306: "f32[48, 128, 64]", view_307: "f32[48, 64, 512]", view_291: "f32[48, 128, 128]", view_292: "f32[48, 128, 64]", view_285: "f32[48, 128, 64]", view_286: "f32[48, 64, 128]", relu_5: "f32[4, 128, 3072]", view_266: "f32[48, 128, 512]", view_267: "f32[48, 512, 64]", view_260: "f32[48, 128, 64]", view_261: "f32[48, 64, 512]", view_245: "f32[48, 128, 128]", view_246: "f32[48, 128, 64]", view_239: "f32[48, 128, 64]", view_240: "f32[48, 64, 128]", relu_4: "f32[4, 128, 3072]", view_220: "f32[48, 128, 512]", view_221: "f32[48, 512, 64]", view_214: "f32[48, 128, 64]", view_215: "f32[48, 64, 512]", view_199: "f32[48, 128, 128]", view_200: "f32[48, 128, 64]", view_193: "f32[48, 128, 64]", view_194: "f32[48, 64, 128]", relu_3: "f32[4, 128, 3072]", view_174: "f32[48, 128, 512]", view_175: "f32[48, 512, 64]", view_168: "f32[48, 128, 64]", view_169: "f32[48, 64, 512]", view_153: "f32[48, 128, 128]", view_154: "f32[48, 128, 64]", view_147: "f32[48, 128, 64]", view_148: "f32[48, 64, 128]", relu_2: "f32[4, 128, 3072]", view_128: "f32[48, 128, 512]", view_129: "f32[48, 512, 64]", view_122: "f32[48, 128, 64]", view_123: "f32[48, 64, 512]", view_107: "f32[48, 128, 128]", view_108: "f32[48, 128, 64]", view_101: "f32[48, 128, 64]", view_102: "f32[48, 64, 128]", relu_1: "f32[4, 128, 3072]", view_82: "f32[48, 128, 512]", view_83: "f32[48, 512, 64]", view_76: "f32[48, 128, 64]", view_77: "f32[48, 64, 512]", view_61: "f32[48, 128, 128]", view_62: "f32[48, 128, 64]", view_55: "f32[48, 128, 64]", view_56: "f32[48, 64, 128]", relu: "f32[4, 128, 3072]", view_36: "f32[48, 128, 512]", view_37: "f32[48, 512, 64]", view_30: "f32[48, 128, 64]", view_31: "f32[48, 64, 512]", view_15: "f32[48, 128, 128]", view_16: "f32[48, 128, 64]", view_9: "f32[48, 128, 64]", view_10: "f32[48, 64, 128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        reshape_default: "f32[4, 128, 768]" = torch.ops.aten.reshape.default(mm_119, _shape_param_0);  mm_119 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 72)
        inductor_random_default: "f32[4, 128, 768]" = torch.ops.prims.inductor_random.default([4, 128, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[4, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(gt_scalar, reshape_default);  gt_scalar = reshape_default = None
        mul_tensor_1: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        add_tensor: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_102, mul_tensor_1);  add_102 = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar: "f32[4, 128, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor, 2)
        mean_dim: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor_1: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[4, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_2: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_tensor, rsqrt_default);  add_tensor = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_3: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(primals_160, mul_tensor_2);  primals_160 = mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:755 in torch_dynamo_resume_in_forward_at_681, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 73);  inductor_seeds_default = None
        inductor_random_default_1: "f32[4, 128, 768]" = torch.ops.prims.inductor_random.default([4, 128, 768], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_scalar_1: "b8[4, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.1);  inductor_random_default_1 = None
        mul_tensor_4: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(gt_scalar_1, mul_tensor_3);  gt_scalar_1 = mul_tensor_3 = None
        mul_tensor_5: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_4, 1.1111111111111112);  mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar: "b8[4, 128, 3072]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default: "f32[48, 512, 128]" = torch.ops.aten.permute.default(view_542, [0, 2, 1]);  view_542 = None
        permute_default_1: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_543, [0, 2, 1]);  view_543 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_2: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_536, [0, 2, 1]);  view_536 = None
        permute_default_3: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_537, [0, 2, 1]);  view_537 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_4: "f32[48, 128, 128]" = torch.ops.aten.permute.default(view_521, [0, 2, 1]);  view_521 = None
        permute_default_5: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_522, [0, 2, 1]);  view_522 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_6: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_515, [0, 2, 1]);  view_515 = None
        permute_default_7: "f32[48, 128, 64]" = torch.ops.aten.permute.default(view_516, [0, 2, 1]);  view_516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_1: "b8[4, 128, 3072]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_8: "f32[48, 512, 128]" = torch.ops.aten.permute.default(view_496, [0, 2, 1]);  view_496 = None
        permute_default_9: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_497, [0, 2, 1]);  view_497 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_10: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_490, [0, 2, 1]);  view_490 = None
        permute_default_11: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_491, [0, 2, 1]);  view_491 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_12: "f32[48, 128, 128]" = torch.ops.aten.permute.default(view_475, [0, 2, 1]);  view_475 = None
        permute_default_13: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_476, [0, 2, 1]);  view_476 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_14: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_469, [0, 2, 1]);  view_469 = None
        permute_default_15: "f32[48, 128, 64]" = torch.ops.aten.permute.default(view_470, [0, 2, 1]);  view_470 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_2: "b8[4, 128, 3072]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_16: "f32[48, 512, 128]" = torch.ops.aten.permute.default(view_450, [0, 2, 1]);  view_450 = None
        permute_default_17: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_451, [0, 2, 1]);  view_451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_18: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_444, [0, 2, 1]);  view_444 = None
        permute_default_19: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_445, [0, 2, 1]);  view_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_20: "f32[48, 128, 128]" = torch.ops.aten.permute.default(view_429, [0, 2, 1]);  view_429 = None
        permute_default_21: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_430, [0, 2, 1]);  view_430 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_22: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_423, [0, 2, 1]);  view_423 = None
        permute_default_23: "f32[48, 128, 64]" = torch.ops.aten.permute.default(view_424, [0, 2, 1]);  view_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_3: "b8[4, 128, 3072]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_24: "f32[48, 512, 128]" = torch.ops.aten.permute.default(view_404, [0, 2, 1]);  view_404 = None
        permute_default_25: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_405, [0, 2, 1]);  view_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_26: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_398, [0, 2, 1]);  view_398 = None
        permute_default_27: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_399, [0, 2, 1]);  view_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_28: "f32[48, 128, 128]" = torch.ops.aten.permute.default(view_383, [0, 2, 1]);  view_383 = None
        permute_default_29: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_384, [0, 2, 1]);  view_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_30: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_377, [0, 2, 1]);  view_377 = None
        permute_default_31: "f32[48, 128, 64]" = torch.ops.aten.permute.default(view_378, [0, 2, 1]);  view_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_4: "b8[4, 128, 3072]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_32: "f32[48, 512, 128]" = torch.ops.aten.permute.default(view_358, [0, 2, 1]);  view_358 = None
        permute_default_33: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_359, [0, 2, 1]);  view_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_34: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_352, [0, 2, 1]);  view_352 = None
        permute_default_35: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_353, [0, 2, 1]);  view_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_36: "f32[48, 128, 128]" = torch.ops.aten.permute.default(view_337, [0, 2, 1]);  view_337 = None
        permute_default_37: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_338, [0, 2, 1]);  view_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_38: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_331, [0, 2, 1]);  view_331 = None
        permute_default_39: "f32[48, 128, 64]" = torch.ops.aten.permute.default(view_332, [0, 2, 1]);  view_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_5: "b8[4, 128, 3072]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_40: "f32[48, 512, 128]" = torch.ops.aten.permute.default(view_312, [0, 2, 1]);  view_312 = None
        permute_default_41: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_313, [0, 2, 1]);  view_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_42: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_306, [0, 2, 1]);  view_306 = None
        permute_default_43: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_307, [0, 2, 1]);  view_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_44: "f32[48, 128, 128]" = torch.ops.aten.permute.default(view_291, [0, 2, 1]);  view_291 = None
        permute_default_45: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_292, [0, 2, 1]);  view_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_46: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_285, [0, 2, 1]);  view_285 = None
        permute_default_47: "f32[48, 128, 64]" = torch.ops.aten.permute.default(view_286, [0, 2, 1]);  view_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_6: "b8[4, 128, 3072]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_48: "f32[48, 512, 128]" = torch.ops.aten.permute.default(view_266, [0, 2, 1]);  view_266 = None
        permute_default_49: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_267, [0, 2, 1]);  view_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_50: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_260, [0, 2, 1]);  view_260 = None
        permute_default_51: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_261, [0, 2, 1]);  view_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_52: "f32[48, 128, 128]" = torch.ops.aten.permute.default(view_245, [0, 2, 1]);  view_245 = None
        permute_default_53: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_246, [0, 2, 1]);  view_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_54: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_239, [0, 2, 1]);  view_239 = None
        permute_default_55: "f32[48, 128, 64]" = torch.ops.aten.permute.default(view_240, [0, 2, 1]);  view_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_7: "b8[4, 128, 3072]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_56: "f32[48, 512, 128]" = torch.ops.aten.permute.default(view_220, [0, 2, 1]);  view_220 = None
        permute_default_57: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_221, [0, 2, 1]);  view_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_58: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_214, [0, 2, 1]);  view_214 = None
        permute_default_59: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_215, [0, 2, 1]);  view_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_60: "f32[48, 128, 128]" = torch.ops.aten.permute.default(view_199, [0, 2, 1]);  view_199 = None
        permute_default_61: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_200, [0, 2, 1]);  view_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_62: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_193, [0, 2, 1]);  view_193 = None
        permute_default_63: "f32[48, 128, 64]" = torch.ops.aten.permute.default(view_194, [0, 2, 1]);  view_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_8: "b8[4, 128, 3072]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_64: "f32[48, 512, 128]" = torch.ops.aten.permute.default(view_174, [0, 2, 1]);  view_174 = None
        permute_default_65: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_175, [0, 2, 1]);  view_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_66: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
        permute_default_67: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_169, [0, 2, 1]);  view_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_68: "f32[48, 128, 128]" = torch.ops.aten.permute.default(view_153, [0, 2, 1]);  view_153 = None
        permute_default_69: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_154, [0, 2, 1]);  view_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_70: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_147, [0, 2, 1]);  view_147 = None
        permute_default_71: "f32[48, 128, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1]);  view_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_9: "b8[4, 128, 3072]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_72: "f32[48, 512, 128]" = torch.ops.aten.permute.default(view_128, [0, 2, 1]);  view_128 = None
        permute_default_73: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_129, [0, 2, 1]);  view_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_74: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
        permute_default_75: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_76: "f32[48, 128, 128]" = torch.ops.aten.permute.default(view_107, [0, 2, 1]);  view_107 = None
        permute_default_77: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_108, [0, 2, 1]);  view_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_78: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None
        permute_default_79: "f32[48, 128, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1]);  view_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_10: "b8[4, 128, 3072]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_80: "f32[48, 512, 128]" = torch.ops.aten.permute.default(view_82, [0, 2, 1]);  view_82 = None
        permute_default_81: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_83, [0, 2, 1]);  view_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_82: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_76, [0, 2, 1]);  view_76 = None
        permute_default_83: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_77, [0, 2, 1]);  view_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_84: "f32[48, 128, 128]" = torch.ops.aten.permute.default(view_61, [0, 2, 1]);  view_61 = None
        permute_default_85: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_62, [0, 2, 1]);  view_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_86: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None
        permute_default_87: "f32[48, 128, 64]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_11: "b8[4, 128, 3072]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_88: "f32[48, 512, 128]" = torch.ops.aten.permute.default(view_36, [0, 2, 1]);  view_36 = None
        permute_default_89: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_37, [0, 2, 1]);  view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_90: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_30, [0, 2, 1]);  view_30 = None
        permute_default_91: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_92: "f32[48, 128, 128]" = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
        permute_default_93: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_16, [0, 2, 1]);  view_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_94: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_default_95: "f32[48, 128, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
        return (mul_tensor_5, le_scalar, permute_default, permute_default_1, permute_default_2, permute_default_3, permute_default_4, permute_default_5, permute_default_6, permute_default_7, le_scalar_1, permute_default_8, permute_default_9, permute_default_10, permute_default_11, permute_default_12, permute_default_13, permute_default_14, permute_default_15, le_scalar_2, permute_default_16, permute_default_17, permute_default_18, permute_default_19, permute_default_20, permute_default_21, permute_default_22, permute_default_23, le_scalar_3, permute_default_24, permute_default_25, permute_default_26, permute_default_27, permute_default_28, permute_default_29, permute_default_30, permute_default_31, le_scalar_4, permute_default_32, permute_default_33, permute_default_34, permute_default_35, permute_default_36, permute_default_37, permute_default_38, permute_default_39, le_scalar_5, permute_default_40, permute_default_41, permute_default_42, permute_default_43, permute_default_44, permute_default_45, permute_default_46, permute_default_47, le_scalar_6, permute_default_48, permute_default_49, permute_default_50, permute_default_51, permute_default_52, permute_default_53, permute_default_54, permute_default_55, le_scalar_7, permute_default_56, permute_default_57, permute_default_58, permute_default_59, permute_default_60, permute_default_61, permute_default_62, permute_default_63, le_scalar_8, permute_default_64, permute_default_65, permute_default_66, permute_default_67, permute_default_68, permute_default_69, permute_default_70, permute_default_71, le_scalar_9, permute_default_72, permute_default_73, permute_default_74, permute_default_75, permute_default_76, permute_default_77, permute_default_78, permute_default_79, le_scalar_10, permute_default_80, permute_default_81, permute_default_82, permute_default_83, permute_default_84, permute_default_85, permute_default_86, permute_default_87, le_scalar_11, permute_default_88, permute_default_89, permute_default_90, permute_default_91, permute_default_92, permute_default_93, permute_default_94, permute_default_95)


def _default_make_inputs():
    return [
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    [4, 128, 768],  # _shape_param_0
    torch.randint(0, 2, [74], dtype=torch.int64, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 128], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
