"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_train
Pattern hash: 8c08b1531d0c
Shape hash: af5ba384
"""
_shapes_config = "(T([4096, 250112], f32), T([32, 128], i64), T([], f32), T([192, 128, 128], f32), T([192, 128, 64], f32), T([192, 128, 64], f32), T([192, 64, 128], f32), T([192, 128, 128], f32), T([192, 128, 64], f32), T([192, 128, 64], f32), T([192, 64, 128], f32), T([192, 128, 128], f32), T([192, 128, 64], f32), T([192, 128, 64], f32), T([192, 64, 128], f32), T([192, 128, 128], f32), T([192, 128, 64], f32), T([192, 128, 64], f32), T([192, 64, 128], f32), T([192, 128, 128], f32), T([192, 128, 64], f32), T([192, 128, 64], f32), T([192, 64, 128], f32), T([192, 128, 128], f32), T([192, 128, 64], f32), T([192, 128, 64], f32), T([192, 64, 128], f32), T([192, 128, 128], f32), T([192, 128, 64], f32), T([192, 128, 64], f32), T([192, 64, 128], f32), T([192, 128, 128], f32), T([192, 128, 64], f32), T([192, 128, 64], f32), T([192, 64, 128], f32), T([192, 128, 128], f32), T([192, 128, 64], f32), T([192, 128, 64], f32), T([192, 64, 128], f32), T([192, 128, 128], f32), T([192, 128, 64], f32), T([192, 128, 64], f32), T([192, 64, 128], f32), T([192, 128, 128], f32), T([192, 128, 64], f32), T([192, 128, 64], f32), T([192, 64, 128], f32), T([192, 128, 128], f32), T([192, 128, 64], f32), T([192, 128, 64], f32), T([192, 64, 128], f32), T([192, 128, 128], f32), T([192, 128, 64], f32), T([192, 128, 64], f32), T([192, 64, 128], f32), T([192, 128, 128], f32), T([192, 128, 64], f32), T([192, 128, 64], f32), T([192, 64, 128], f32), T([192, 128, 128], f32), T([192, 128, 64], f32), T([192, 128, 64], f32), T([192, 64, 128], f32), T([192, 128, 128], f32), T([192, 128, 64], f32), T([192, 128, 64], f32), T([192, 64, 128], f32), T([192, 128, 128], f32), T([192, 128, 64], f32), T([192, 128, 64], f32), T([192, 64, 128], f32), T([192, 128, 128], f32), T([192, 128, 64], f32), T([192, 128, 64], f32), T([192, 64, 128], f32), T([192, 128, 128], f32), T([192, 128, 64], f32), T([192, 128, 64], f32), T([192, 64, 128], f32), T([192, 128, 128], f32), T([192, 128, 64], f32), T([192, 128, 64], f32), T([192, 64, 128], f32), T([192, 128, 128], f32), T([192, 128, 64], f32), T([192, 128, 64], f32), T([192, 64, 128], f32), T([192, 128, 128], f32), T([192, 128, 64], f32), T([192, 128, 64], f32), T([192, 64, 128], f32), T([192, 128, 128], f32), T([192, 128, 64], f32), T([192, 128, 64], f32), T([192, 64, 128], f32), T([192, 128, 128], f32), T([192, 128, 64], f32), T([192, 128, 64], f32), T([192, 64, 128], f32), S([32, 128, 250112]), S([-1, 250112]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_144: "f32[4096, 250112]", primals_77: "i64[32, 128]", full_default: "f32[]", view_590: "f32[192, 128, 128]", view_591: "f32[192, 128, 64]", view_584: "f32[192, 128, 64]", view_585: "f32[192, 64, 128]", view_569: "f32[192, 128, 128]", view_570: "f32[192, 128, 64]", view_563: "f32[192, 128, 64]", view_564: "f32[192, 64, 128]", view_542: "f32[192, 128, 128]", view_543: "f32[192, 128, 64]", view_536: "f32[192, 128, 64]", view_537: "f32[192, 64, 128]", view_521: "f32[192, 128, 128]", view_522: "f32[192, 128, 64]", view_515: "f32[192, 128, 64]", view_516: "f32[192, 64, 128]", view_494: "f32[192, 128, 128]", view_495: "f32[192, 128, 64]", view_488: "f32[192, 128, 64]", view_489: "f32[192, 64, 128]", view_473: "f32[192, 128, 128]", view_474: "f32[192, 128, 64]", view_467: "f32[192, 128, 64]", view_468: "f32[192, 64, 128]", view_446: "f32[192, 128, 128]", view_447: "f32[192, 128, 64]", view_440: "f32[192, 128, 64]", view_441: "f32[192, 64, 128]", view_425: "f32[192, 128, 128]", view_426: "f32[192, 128, 64]", view_419: "f32[192, 128, 64]", view_420: "f32[192, 64, 128]", view_398: "f32[192, 128, 128]", view_399: "f32[192, 128, 64]", view_392: "f32[192, 128, 64]", view_393: "f32[192, 64, 128]", view_377: "f32[192, 128, 128]", view_378: "f32[192, 128, 64]", view_371: "f32[192, 128, 64]", view_372: "f32[192, 64, 128]", view_350: "f32[192, 128, 128]", view_351: "f32[192, 128, 64]", view_344: "f32[192, 128, 64]", view_345: "f32[192, 64, 128]", view_329: "f32[192, 128, 128]", view_330: "f32[192, 128, 64]", view_323: "f32[192, 128, 64]", view_324: "f32[192, 64, 128]", view_302: "f32[192, 128, 128]", view_303: "f32[192, 128, 64]", view_296: "f32[192, 128, 64]", view_297: "f32[192, 64, 128]", view_281: "f32[192, 128, 128]", view_282: "f32[192, 128, 64]", view_275: "f32[192, 128, 64]", view_276: "f32[192, 64, 128]", view_254: "f32[192, 128, 128]", view_255: "f32[192, 128, 64]", view_248: "f32[192, 128, 64]", view_249: "f32[192, 64, 128]", view_233: "f32[192, 128, 128]", view_234: "f32[192, 128, 64]", view_227: "f32[192, 128, 64]", view_228: "f32[192, 64, 128]", view_205: "f32[192, 128, 128]", view_206: "f32[192, 128, 64]", view_199: "f32[192, 128, 64]", view_200: "f32[192, 64, 128]", view_178: "f32[192, 128, 128]", view_179: "f32[192, 128, 64]", view_172: "f32[192, 128, 64]", view_173: "f32[192, 64, 128]", view_151: "f32[192, 128, 128]", view_152: "f32[192, 128, 64]", view_145: "f32[192, 128, 64]", view_146: "f32[192, 64, 128]", view_124: "f32[192, 128, 128]", view_125: "f32[192, 128, 64]", view_118: "f32[192, 128, 64]", view_119: "f32[192, 64, 128]", view_97: "f32[192, 128, 128]", view_98: "f32[192, 128, 64]", view_91: "f32[192, 128, 64]", view_92: "f32[192, 64, 128]", view_70: "f32[192, 128, 128]", view_71: "f32[192, 128, 64]", view_64: "f32[192, 128, 64]", view_65: "f32[192, 64, 128]", view_43: "f32[192, 128, 128]", view_44: "f32[192, 128, 64]", view_37: "f32[192, 128, 64]", view_38: "f32[192, 64, 128]", view_16: "f32[192, 128, 128]", view_17: "f32[192, 128, 64]", view_10: "f32[192, 128, 64]", view_11: "f32[192, 64, 128]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:1143 in forward, code: lm_logits = self.lm_head(sequence_output)
        reshape_default: "f32[32, 128, 250112]" = torch.ops.aten.reshape.default(mm_144, _shape_param_0);  mm_144 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:1150 in forward, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        reshape_default_1: "f32[4096, 250112]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        reshape_default_2: "i64[4096]" = torch.ops.aten.reshape.default(primals_77, [-1]);  primals_77 = None
        amax_default: "f32[4096, 1]" = torch.ops.aten.amax.default(reshape_default_1, [1], True)
        sub_tensor: "f32[4096, 250112]" = torch.ops.aten.sub.Tensor(reshape_default_1, amax_default);  reshape_default_1 = amax_default = None
        exp_default: "f32[4096, 250112]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[4096, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[4096, 250112]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        ne_scalar: "b8[4096]" = torch.ops.aten.ne.Scalar(reshape_default_2, -100)
        full_default_1: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[4096]" = torch.ops.aten.where.self(ne_scalar, reshape_default_2, full_default_1);  reshape_default_2 = full_default_1 = None
        unsqueeze_default: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[4096, 1]" = torch.ops.aten.gather.default(sub_tensor_1, 1, unsqueeze_default);  sub_tensor_1 = unsqueeze_default = None
        squeeze_dim: "f32[4096]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[4096]" = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        where_self_1: "f32[4096]" = torch.ops.aten.where.self(ne_scalar, neg_default, full_default);  neg_default = full_default = None
        sum_default: "i64[]" = torch.ops.aten.sum.default(ne_scalar);  ne_scalar = None
        convert_element_type_default: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default, torch.float32);  sum_default = None
        sum_default_1: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default_1, convert_element_type_default);  sum_default_1 = convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default: "f32[192, 128, 128]" = torch.ops.aten.permute.default(view_590, [0, 2, 1]);  view_590 = None
        permute_default_1: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_591, [0, 2, 1]);  view_591 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_2: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_584, [0, 2, 1]);  view_584 = None
        permute_default_3: "f32[192, 128, 64]" = torch.ops.aten.permute.default(view_585, [0, 2, 1]);  view_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_4: "f32[192, 128, 128]" = torch.ops.aten.permute.default(view_569, [0, 2, 1]);  view_569 = None
        permute_default_5: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_570, [0, 2, 1]);  view_570 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_6: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_563, [0, 2, 1]);  view_563 = None
        permute_default_7: "f32[192, 128, 64]" = torch.ops.aten.permute.default(view_564, [0, 2, 1]);  view_564 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_8: "f32[192, 128, 128]" = torch.ops.aten.permute.default(view_542, [0, 2, 1]);  view_542 = None
        permute_default_9: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_543, [0, 2, 1]);  view_543 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_10: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_536, [0, 2, 1]);  view_536 = None
        permute_default_11: "f32[192, 128, 64]" = torch.ops.aten.permute.default(view_537, [0, 2, 1]);  view_537 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_12: "f32[192, 128, 128]" = torch.ops.aten.permute.default(view_521, [0, 2, 1]);  view_521 = None
        permute_default_13: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_522, [0, 2, 1]);  view_522 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_14: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_515, [0, 2, 1]);  view_515 = None
        permute_default_15: "f32[192, 128, 64]" = torch.ops.aten.permute.default(view_516, [0, 2, 1]);  view_516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_16: "f32[192, 128, 128]" = torch.ops.aten.permute.default(view_494, [0, 2, 1]);  view_494 = None
        permute_default_17: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_495, [0, 2, 1]);  view_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_18: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_488, [0, 2, 1]);  view_488 = None
        permute_default_19: "f32[192, 128, 64]" = torch.ops.aten.permute.default(view_489, [0, 2, 1]);  view_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_20: "f32[192, 128, 128]" = torch.ops.aten.permute.default(view_473, [0, 2, 1]);  view_473 = None
        permute_default_21: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_474, [0, 2, 1]);  view_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_22: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_467, [0, 2, 1]);  view_467 = None
        permute_default_23: "f32[192, 128, 64]" = torch.ops.aten.permute.default(view_468, [0, 2, 1]);  view_468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_24: "f32[192, 128, 128]" = torch.ops.aten.permute.default(view_446, [0, 2, 1]);  view_446 = None
        permute_default_25: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_447, [0, 2, 1]);  view_447 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_26: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_440, [0, 2, 1]);  view_440 = None
        permute_default_27: "f32[192, 128, 64]" = torch.ops.aten.permute.default(view_441, [0, 2, 1]);  view_441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_28: "f32[192, 128, 128]" = torch.ops.aten.permute.default(view_425, [0, 2, 1]);  view_425 = None
        permute_default_29: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_426, [0, 2, 1]);  view_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_30: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_419, [0, 2, 1]);  view_419 = None
        permute_default_31: "f32[192, 128, 64]" = torch.ops.aten.permute.default(view_420, [0, 2, 1]);  view_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_32: "f32[192, 128, 128]" = torch.ops.aten.permute.default(view_398, [0, 2, 1]);  view_398 = None
        permute_default_33: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_399, [0, 2, 1]);  view_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_34: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_392, [0, 2, 1]);  view_392 = None
        permute_default_35: "f32[192, 128, 64]" = torch.ops.aten.permute.default(view_393, [0, 2, 1]);  view_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_36: "f32[192, 128, 128]" = torch.ops.aten.permute.default(view_377, [0, 2, 1]);  view_377 = None
        permute_default_37: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_378, [0, 2, 1]);  view_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_38: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_371, [0, 2, 1]);  view_371 = None
        permute_default_39: "f32[192, 128, 64]" = torch.ops.aten.permute.default(view_372, [0, 2, 1]);  view_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_40: "f32[192, 128, 128]" = torch.ops.aten.permute.default(view_350, [0, 2, 1]);  view_350 = None
        permute_default_41: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_351, [0, 2, 1]);  view_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_42: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_344, [0, 2, 1]);  view_344 = None
        permute_default_43: "f32[192, 128, 64]" = torch.ops.aten.permute.default(view_345, [0, 2, 1]);  view_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_44: "f32[192, 128, 128]" = torch.ops.aten.permute.default(view_329, [0, 2, 1]);  view_329 = None
        permute_default_45: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_330, [0, 2, 1]);  view_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_46: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_323, [0, 2, 1]);  view_323 = None
        permute_default_47: "f32[192, 128, 64]" = torch.ops.aten.permute.default(view_324, [0, 2, 1]);  view_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_48: "f32[192, 128, 128]" = torch.ops.aten.permute.default(view_302, [0, 2, 1]);  view_302 = None
        permute_default_49: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_303, [0, 2, 1]);  view_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_50: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_296, [0, 2, 1]);  view_296 = None
        permute_default_51: "f32[192, 128, 64]" = torch.ops.aten.permute.default(view_297, [0, 2, 1]);  view_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_52: "f32[192, 128, 128]" = torch.ops.aten.permute.default(view_281, [0, 2, 1]);  view_281 = None
        permute_default_53: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_282, [0, 2, 1]);  view_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_54: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_275, [0, 2, 1]);  view_275 = None
        permute_default_55: "f32[192, 128, 64]" = torch.ops.aten.permute.default(view_276, [0, 2, 1]);  view_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_56: "f32[192, 128, 128]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
        permute_default_57: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_58: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_248, [0, 2, 1]);  view_248 = None
        permute_default_59: "f32[192, 128, 64]" = torch.ops.aten.permute.default(view_249, [0, 2, 1]);  view_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_60: "f32[192, 128, 128]" = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None
        permute_default_61: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_234, [0, 2, 1]);  view_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_62: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_227, [0, 2, 1]);  view_227 = None
        permute_default_63: "f32[192, 128, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1]);  view_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_64: "f32[192, 128, 128]" = torch.ops.aten.permute.default(view_205, [0, 2, 1]);  view_205 = None
        permute_default_65: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_206, [0, 2, 1]);  view_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_66: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_199, [0, 2, 1]);  view_199 = None
        permute_default_67: "f32[192, 128, 64]" = torch.ops.aten.permute.default(view_200, [0, 2, 1]);  view_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_68: "f32[192, 128, 128]" = torch.ops.aten.permute.default(view_178, [0, 2, 1]);  view_178 = None
        permute_default_69: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_179, [0, 2, 1]);  view_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_70: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_172, [0, 2, 1]);  view_172 = None
        permute_default_71: "f32[192, 128, 64]" = torch.ops.aten.permute.default(view_173, [0, 2, 1]);  view_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_72: "f32[192, 128, 128]" = torch.ops.aten.permute.default(view_151, [0, 2, 1]);  view_151 = None
        permute_default_73: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_152, [0, 2, 1]);  view_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_74: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None
        permute_default_75: "f32[192, 128, 64]" = torch.ops.aten.permute.default(view_146, [0, 2, 1]);  view_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_76: "f32[192, 128, 128]" = torch.ops.aten.permute.default(view_124, [0, 2, 1]);  view_124 = None
        permute_default_77: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_125, [0, 2, 1]);  view_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_78: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_118, [0, 2, 1]);  view_118 = None
        permute_default_79: "f32[192, 128, 64]" = torch.ops.aten.permute.default(view_119, [0, 2, 1]);  view_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_80: "f32[192, 128, 128]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
        permute_default_81: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_82: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_91, [0, 2, 1]);  view_91 = None
        permute_default_83: "f32[192, 128, 64]" = torch.ops.aten.permute.default(view_92, [0, 2, 1]);  view_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_84: "f32[192, 128, 128]" = torch.ops.aten.permute.default(view_70, [0, 2, 1]);  view_70 = None
        permute_default_85: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_71, [0, 2, 1]);  view_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_86: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_64, [0, 2, 1]);  view_64 = None
        permute_default_87: "f32[192, 128, 64]" = torch.ops.aten.permute.default(view_65, [0, 2, 1]);  view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_88: "f32[192, 128, 128]" = torch.ops.aten.permute.default(view_43, [0, 2, 1]);  view_43 = None
        permute_default_89: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_44, [0, 2, 1]);  view_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_90: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_37, [0, 2, 1]);  view_37 = None
        permute_default_91: "f32[192, 128, 64]" = torch.ops.aten.permute.default(view_38, [0, 2, 1]);  view_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_92: "f32[192, 128, 128]" = torch.ops.aten.permute.default(view_16, [0, 2, 1]);  view_16 = None
        permute_default_93: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_17, [0, 2, 1]);  view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_94: "f32[192, 64, 128]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
        permute_default_95: "f32[192, 128, 64]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
        return (div_tensor, permute_default, permute_default_1, permute_default_2, permute_default_3, permute_default_4, permute_default_5, permute_default_6, permute_default_7, permute_default_8, permute_default_9, permute_default_10, permute_default_11, permute_default_12, permute_default_13, permute_default_14, permute_default_15, permute_default_16, permute_default_17, permute_default_18, permute_default_19, permute_default_20, permute_default_21, permute_default_22, permute_default_23, permute_default_24, permute_default_25, permute_default_26, permute_default_27, permute_default_28, permute_default_29, permute_default_30, permute_default_31, permute_default_32, permute_default_33, permute_default_34, permute_default_35, permute_default_36, permute_default_37, permute_default_38, permute_default_39, permute_default_40, permute_default_41, permute_default_42, permute_default_43, permute_default_44, permute_default_45, permute_default_46, permute_default_47, permute_default_48, permute_default_49, permute_default_50, permute_default_51, permute_default_52, permute_default_53, permute_default_54, permute_default_55, permute_default_56, permute_default_57, permute_default_58, permute_default_59, permute_default_60, permute_default_61, permute_default_62, permute_default_63, permute_default_64, permute_default_65, permute_default_66, permute_default_67, permute_default_68, permute_default_69, permute_default_70, permute_default_71, permute_default_72, permute_default_73, permute_default_74, permute_default_75, permute_default_76, permute_default_77, permute_default_78, permute_default_79, permute_default_80, permute_default_81, permute_default_82, permute_default_83, permute_default_84, permute_default_85, permute_default_86, permute_default_87, permute_default_88, permute_default_89, permute_default_90, permute_default_91, permute_default_92, permute_default_93, permute_default_94, permute_default_95)



def make_inputs():
    return [
    torch.randn([4096, 250112], dtype=torch.float32, device='cuda'),
    torch.randint(0, 128, [32, 128], dtype=torch.int64, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 64, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 64, 128], dtype=torch.float32, device='cuda'),
    [32, 128, 250112],  # _shape_param_0
    [-1, 250112],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
