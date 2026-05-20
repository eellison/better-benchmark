"""
Standalone repro captured via capture_hook.
Label: hf_ElectraForCausalLM_train
Pattern hash: 0295d490aeb6
Shape hash: eb493895
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([32768, 30522], f32), T([64, 512, 128], f32), T([64, 512, 128], f32), T([32768, 128], f32), T([64, 512, 256], f32), T([64, 512, 256], f32), T([32768, 256], f32), T([32768, 1024], f32), T([64, 512, 256], f32), T([64, 512, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([64, 512, 256], f32), T([64, 512, 256], f32), T([32768, 256], f32), T([32768, 1024], f32), T([64, 512, 256], f32), T([64, 512, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([64, 512, 256], f32), T([64, 512, 256], f32), T([32768, 256], f32), T([32768, 1024], f32), T([64, 512, 256], f32), T([64, 512, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([64, 512, 256], f32), T([64, 512, 256], f32), T([32768, 256], f32), T([32768, 1024], f32), T([64, 512, 256], f32), T([64, 512, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([64, 512, 256], f32), T([64, 512, 256], f32), T([32768, 256], f32), T([32768, 1024], f32), T([64, 512, 256], f32), T([64, 512, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([64, 512, 256], f32), T([64, 512, 256], f32), T([32768, 256], f32), T([32768, 1024], f32), T([64, 512, 256], f32), T([64, 512, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([64, 512, 256], f32), T([64, 512, 256], f32), T([32768, 256], f32), T([32768, 1024], f32), T([64, 512, 256], f32), T([64, 512, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([64, 512, 256], f32), T([64, 512, 256], f32), T([32768, 256], f32), T([32768, 1024], f32), T([64, 512, 256], f32), T([64, 512, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([64, 512, 256], f32), T([64, 512, 256], f32), T([32768, 256], f32), T([32768, 1024], f32), T([64, 512, 256], f32), T([64, 512, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([64, 512, 256], f32), T([64, 512, 256], f32), T([32768, 256], f32), T([32768, 1024], f32), T([64, 512, 256], f32), T([64, 512, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([64, 512, 256], f32), T([64, 512, 256], f32), T([32768, 256], f32), T([32768, 1024], f32), T([64, 512, 256], f32), T([64, 512, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([64, 512, 256], f32), T([64, 512, 256], f32), T([32768, 256], f32), T([32768, 1024], f32), T([64, 512, 256], f32), T([64, 512, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([32768, 256], f32), T([32768, 128], f32), T([64, 512, 128], b8), T([128], f32), T([64, 512, 128], f32), T([64, 512, 1], f32), T([1, 512], i64, gen=Index(512)), T([], f32), T([1, 512], i64, gen=Index(2)), T([64, 512], i64, gen=Index(30522)), T([30522, 128], f32), S([30522]), S([128]), S([256]), S([1024]), S([256]), S([256]), S([256]), S([256]), S([256]), S([1024]), S([256]), S([256]), S([256]), S([256]), S([256]), S([1024]), S([256]), S([256]), S([256]), S([256]), S([256]), S([1024]), S([256]), S([256]), S([256]), S([256]), S([256]), S([1024]), S([256]), S([256]), S([256]), S([256]), S([256]), S([1024]), S([256]), S([256]), S([256]), S([256]), S([256]), S([1024]), S([256]), S([256]), S([256]), S([256]), S([256]), S([1024]), S([256]), S([256]), S([256]), S([256]), S([256]), S([1024]), S([256]), S([256]), S([256]), S([256]), S([256]), S([1024]), S([256]), S([256]), S([256]), S([256]), S([256]), S([1024]), S([256]), S([256]), S([256]), S([256]), S([256]), S([1024]), S([256]), S([256]), S([256]), S([256]), S([256]), S([64, 512, 128]), S([64, 512]))"

class Repro(torch.nn.Module):
    def forward(self, view_273: "f32[32768, 30522]", view_275: "f32[64, 512, 128]", mul_187: "f32[64, 512, 128]", view_276: "f32[32768, 128]", view_278: "f32[64, 512, 256]", mul_182: "f32[64, 512, 256]", view_279: "f32[32768, 256]", view_282: "f32[32768, 1024]", add_110: "f32[64, 512, 256]", mul_175: "f32[64, 512, 256]", view_285: "f32[32768, 256]", view_296: "f32[32768, 256]", view_300: "f32[32768, 256]", view_304: "f32[32768, 256]", add_113: "f32[64, 512, 256]", mul_167: "f32[64, 512, 256]", view_307: "f32[32768, 256]", view_310: "f32[32768, 1024]", add_116: "f32[64, 512, 256]", mul_160: "f32[64, 512, 256]", view_313: "f32[32768, 256]", view_324: "f32[32768, 256]", view_328: "f32[32768, 256]", view_332: "f32[32768, 256]", add_119: "f32[64, 512, 256]", mul_152: "f32[64, 512, 256]", view_335: "f32[32768, 256]", view_338: "f32[32768, 1024]", add_122: "f32[64, 512, 256]", mul_145: "f32[64, 512, 256]", view_341: "f32[32768, 256]", view_352: "f32[32768, 256]", view_356: "f32[32768, 256]", view_360: "f32[32768, 256]", add_125: "f32[64, 512, 256]", mul_137: "f32[64, 512, 256]", view_363: "f32[32768, 256]", view_366: "f32[32768, 1024]", add_128: "f32[64, 512, 256]", mul_130: "f32[64, 512, 256]", view_369: "f32[32768, 256]", view_380: "f32[32768, 256]", view_384: "f32[32768, 256]", view_388: "f32[32768, 256]", add_131: "f32[64, 512, 256]", mul_122: "f32[64, 512, 256]", view_391: "f32[32768, 256]", view_394: "f32[32768, 1024]", add_134: "f32[64, 512, 256]", mul_115: "f32[64, 512, 256]", view_397: "f32[32768, 256]", view_408: "f32[32768, 256]", view_412: "f32[32768, 256]", view_416: "f32[32768, 256]", add_137: "f32[64, 512, 256]", mul_107: "f32[64, 512, 256]", view_419: "f32[32768, 256]", view_422: "f32[32768, 1024]", add_140: "f32[64, 512, 256]", mul_100: "f32[64, 512, 256]", view_425: "f32[32768, 256]", view_436: "f32[32768, 256]", view_440: "f32[32768, 256]", view_444: "f32[32768, 256]", add_143: "f32[64, 512, 256]", mul_92: "f32[64, 512, 256]", view_447: "f32[32768, 256]", view_450: "f32[32768, 1024]", add_146: "f32[64, 512, 256]", mul_85: "f32[64, 512, 256]", view_453: "f32[32768, 256]", view_464: "f32[32768, 256]", view_468: "f32[32768, 256]", view_472: "f32[32768, 256]", add_149: "f32[64, 512, 256]", mul_77: "f32[64, 512, 256]", view_475: "f32[32768, 256]", view_478: "f32[32768, 1024]", add_152: "f32[64, 512, 256]", mul_70: "f32[64, 512, 256]", view_481: "f32[32768, 256]", view_492: "f32[32768, 256]", view_496: "f32[32768, 256]", view_500: "f32[32768, 256]", add_155: "f32[64, 512, 256]", mul_62: "f32[64, 512, 256]", view_503: "f32[32768, 256]", view_506: "f32[32768, 1024]", add_158: "f32[64, 512, 256]", mul_55: "f32[64, 512, 256]", view_509: "f32[32768, 256]", view_520: "f32[32768, 256]", view_524: "f32[32768, 256]", view_528: "f32[32768, 256]", add_161: "f32[64, 512, 256]", mul_47: "f32[64, 512, 256]", view_531: "f32[32768, 256]", view_534: "f32[32768, 1024]", add_164: "f32[64, 512, 256]", mul_40: "f32[64, 512, 256]", view_537: "f32[32768, 256]", view_548: "f32[32768, 256]", view_552: "f32[32768, 256]", view_556: "f32[32768, 256]", add_167: "f32[64, 512, 256]", mul_32: "f32[64, 512, 256]", view_559: "f32[32768, 256]", view_562: "f32[32768, 1024]", add_170: "f32[64, 512, 256]", mul_25: "f32[64, 512, 256]", view_565: "f32[32768, 256]", view_576: "f32[32768, 256]", view_580: "f32[32768, 256]", view_584: "f32[32768, 256]", add_173: "f32[64, 512, 256]", mul_17: "f32[64, 512, 256]", view_587: "f32[32768, 256]", view_590: "f32[32768, 1024]", add_176: "f32[64, 512, 256]", mul_10: "f32[64, 512, 256]", view_593: "f32[32768, 256]", view_604: "f32[32768, 256]", view_608: "f32[32768, 256]", view_612: "f32[32768, 256]", view_615: "f32[32768, 256]", mm_148: "f32[32768, 128]", gt: "b8[64, 512, 128]", primals_8: "f32[128]", mul: "f32[64, 512, 128]", div_39: "f32[64, 512, 1]", primals_3: "i64[1, 512]", full_default_1: "f32[]", gather: "i64[1, 512]", primals_2: "i64[64, 512]", mm_1: "f32[30522, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:1351 in forward, code: logits = self.generator_lm_head(self.generator_predictions(hidden_states[:, slice_indices, :]))
        sum_dim_int_list: "f32[1, 30522]" = torch.ops.aten.sum.dim_IntList(view_273, [0], True);  view_273 = None
        reshape_default: "f32[30522]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:501 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        mul_tensor: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(view_275, mul_187);  mul_187 = None
        sum_dim_int_list_1: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_2: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_275, [0, 1]);  view_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:499 in forward, code: hidden_states = self.dense(generator_hidden_states)
        permute_default: "f32[128, 32768]" = torch.ops.aten.permute.default(view_276, [1, 0])
        sum_dim_int_list_3: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_276, [0], True);  view_276 = None
        reshape_default_1: "f32[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_1);  sum_dim_int_list_3 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_1: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(view_278, mul_182);  mul_182 = None
        sum_dim_int_list_4: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_5: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_278, [0, 1]);  view_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_1: "f32[256, 32768]" = torch.ops.aten.permute.default(view_279, [1, 0])
        sum_dim_int_list_6: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_279, [0], True);  view_279 = None
        reshape_default_2: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, _shape_param_2);  sum_dim_int_list_6 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_2: "f32[1024, 32768]" = torch.ops.aten.permute.default(view_282, [1, 0])
        sum_dim_int_list_7: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_282, [0], True);  view_282 = None
        reshape_default_3: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_3);  sum_dim_int_list_7 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_2: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_110, mul_175);  mul_175 = None
        sum_dim_int_list_8: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        sum_dim_int_list_9: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_110, [0, 1]);  add_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_3: "f32[256, 32768]" = torch.ops.aten.permute.default(view_285, [1, 0])
        sum_dim_int_list_10: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_285, [0], True);  view_285 = None
        reshape_default_4: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, _shape_param_4);  sum_dim_int_list_10 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_4: "f32[256, 32768]" = torch.ops.aten.permute.default(view_296, [1, 0])
        sum_dim_int_list_11: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_296, [0], True);  view_296 = None
        reshape_default_5: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, _shape_param_5);  sum_dim_int_list_11 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_5: "f32[256, 32768]" = torch.ops.aten.permute.default(view_300, [1, 0])
        sum_dim_int_list_12: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_300, [0], True);  view_300 = None
        reshape_default_6: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, _shape_param_6);  sum_dim_int_list_12 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_6: "f32[256, 32768]" = torch.ops.aten.permute.default(view_304, [1, 0])
        sum_dim_int_list_13: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_304, [0], True);  view_304 = None
        reshape_default_7: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_13, _shape_param_7);  sum_dim_int_list_13 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_3: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_113, mul_167);  mul_167 = None
        sum_dim_int_list_14: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        sum_dim_int_list_15: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_113, [0, 1]);  add_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_7: "f32[256, 32768]" = torch.ops.aten.permute.default(view_307, [1, 0])
        sum_dim_int_list_16: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_307, [0], True);  view_307 = None
        reshape_default_8: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, _shape_param_8);  sum_dim_int_list_16 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_8: "f32[1024, 32768]" = torch.ops.aten.permute.default(view_310, [1, 0])
        sum_dim_int_list_17: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_310, [0], True);  view_310 = None
        reshape_default_9: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_17, _shape_param_9);  sum_dim_int_list_17 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_4: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_116, mul_160);  mul_160 = None
        sum_dim_int_list_18: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_19: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_116, [0, 1]);  add_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_9: "f32[256, 32768]" = torch.ops.aten.permute.default(view_313, [1, 0])
        sum_dim_int_list_20: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_313, [0], True);  view_313 = None
        reshape_default_10: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_20, _shape_param_10);  sum_dim_int_list_20 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_10: "f32[256, 32768]" = torch.ops.aten.permute.default(view_324, [1, 0])
        sum_dim_int_list_21: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_324, [0], True);  view_324 = None
        reshape_default_11: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_21, _shape_param_11);  sum_dim_int_list_21 = _shape_param_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_11: "f32[256, 32768]" = torch.ops.aten.permute.default(view_328, [1, 0])
        sum_dim_int_list_22: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_328, [0], True);  view_328 = None
        reshape_default_12: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_22, _shape_param_12);  sum_dim_int_list_22 = _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_12: "f32[256, 32768]" = torch.ops.aten.permute.default(view_332, [1, 0])
        sum_dim_int_list_23: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_332, [0], True);  view_332 = None
        reshape_default_13: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, _shape_param_13);  sum_dim_int_list_23 = _shape_param_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_5: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_119, mul_152);  mul_152 = None
        sum_dim_int_list_24: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_25: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_119, [0, 1]);  add_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_13: "f32[256, 32768]" = torch.ops.aten.permute.default(view_335, [1, 0])
        sum_dim_int_list_26: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_335, [0], True);  view_335 = None
        reshape_default_14: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_26, _shape_param_14);  sum_dim_int_list_26 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_14: "f32[1024, 32768]" = torch.ops.aten.permute.default(view_338, [1, 0])
        sum_dim_int_list_27: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_338, [0], True);  view_338 = None
        reshape_default_15: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, _shape_param_15);  sum_dim_int_list_27 = _shape_param_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_6: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_122, mul_145);  mul_145 = None
        sum_dim_int_list_28: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_29: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_122, [0, 1]);  add_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_15: "f32[256, 32768]" = torch.ops.aten.permute.default(view_341, [1, 0])
        sum_dim_int_list_30: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_341, [0], True);  view_341 = None
        reshape_default_16: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_30, _shape_param_16);  sum_dim_int_list_30 = _shape_param_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_16: "f32[256, 32768]" = torch.ops.aten.permute.default(view_352, [1, 0])
        sum_dim_int_list_31: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_352, [0], True);  view_352 = None
        reshape_default_17: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_31, _shape_param_17);  sum_dim_int_list_31 = _shape_param_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_17: "f32[256, 32768]" = torch.ops.aten.permute.default(view_356, [1, 0])
        sum_dim_int_list_32: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_356, [0], True);  view_356 = None
        reshape_default_18: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_32, _shape_param_18);  sum_dim_int_list_32 = _shape_param_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_18: "f32[256, 32768]" = torch.ops.aten.permute.default(view_360, [1, 0])
        sum_dim_int_list_33: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_360, [0], True);  view_360 = None
        reshape_default_19: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_33, _shape_param_19);  sum_dim_int_list_33 = _shape_param_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_7: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_125, mul_137);  mul_137 = None
        sum_dim_int_list_34: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_35: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_125, [0, 1]);  add_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_19: "f32[256, 32768]" = torch.ops.aten.permute.default(view_363, [1, 0])
        sum_dim_int_list_36: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_363, [0], True);  view_363 = None
        reshape_default_20: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_36, _shape_param_20);  sum_dim_int_list_36 = _shape_param_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_20: "f32[1024, 32768]" = torch.ops.aten.permute.default(view_366, [1, 0])
        sum_dim_int_list_37: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_366, [0], True);  view_366 = None
        reshape_default_21: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_37, _shape_param_21);  sum_dim_int_list_37 = _shape_param_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_8: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_128, mul_130);  mul_130 = None
        sum_dim_int_list_38: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_39: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_128, [0, 1]);  add_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_21: "f32[256, 32768]" = torch.ops.aten.permute.default(view_369, [1, 0])
        sum_dim_int_list_40: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_369, [0], True);  view_369 = None
        reshape_default_22: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_40, _shape_param_22);  sum_dim_int_list_40 = _shape_param_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_22: "f32[256, 32768]" = torch.ops.aten.permute.default(view_380, [1, 0])
        sum_dim_int_list_41: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_380, [0], True);  view_380 = None
        reshape_default_23: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_41, _shape_param_23);  sum_dim_int_list_41 = _shape_param_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_23: "f32[256, 32768]" = torch.ops.aten.permute.default(view_384, [1, 0])
        sum_dim_int_list_42: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_384, [0], True);  view_384 = None
        reshape_default_24: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_42, _shape_param_24);  sum_dim_int_list_42 = _shape_param_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_24: "f32[256, 32768]" = torch.ops.aten.permute.default(view_388, [1, 0])
        sum_dim_int_list_43: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_388, [0], True);  view_388 = None
        reshape_default_25: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_43, _shape_param_25);  sum_dim_int_list_43 = _shape_param_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_9: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_131, mul_122);  mul_122 = None
        sum_dim_int_list_44: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_45: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_131, [0, 1]);  add_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_25: "f32[256, 32768]" = torch.ops.aten.permute.default(view_391, [1, 0])
        sum_dim_int_list_46: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_391, [0], True);  view_391 = None
        reshape_default_26: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_46, _shape_param_26);  sum_dim_int_list_46 = _shape_param_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_26: "f32[1024, 32768]" = torch.ops.aten.permute.default(view_394, [1, 0])
        sum_dim_int_list_47: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_394, [0], True);  view_394 = None
        reshape_default_27: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_47, _shape_param_27);  sum_dim_int_list_47 = _shape_param_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_10: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_134, mul_115);  mul_115 = None
        sum_dim_int_list_48: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_49: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_134, [0, 1]);  add_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_27: "f32[256, 32768]" = torch.ops.aten.permute.default(view_397, [1, 0])
        sum_dim_int_list_50: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_397, [0], True);  view_397 = None
        reshape_default_28: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_50, _shape_param_28);  sum_dim_int_list_50 = _shape_param_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_28: "f32[256, 32768]" = torch.ops.aten.permute.default(view_408, [1, 0])
        sum_dim_int_list_51: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_408, [0], True);  view_408 = None
        reshape_default_29: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_51, _shape_param_29);  sum_dim_int_list_51 = _shape_param_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_29: "f32[256, 32768]" = torch.ops.aten.permute.default(view_412, [1, 0])
        sum_dim_int_list_52: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_412, [0], True);  view_412 = None
        reshape_default_30: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_52, _shape_param_30);  sum_dim_int_list_52 = _shape_param_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_30: "f32[256, 32768]" = torch.ops.aten.permute.default(view_416, [1, 0])
        sum_dim_int_list_53: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_416, [0], True);  view_416 = None
        reshape_default_31: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_53, _shape_param_31);  sum_dim_int_list_53 = _shape_param_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_11: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_137, mul_107);  mul_107 = None
        sum_dim_int_list_54: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1]);  mul_tensor_11 = None
        sum_dim_int_list_55: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_137, [0, 1]);  add_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_31: "f32[256, 32768]" = torch.ops.aten.permute.default(view_419, [1, 0])
        sum_dim_int_list_56: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_419, [0], True);  view_419 = None
        reshape_default_32: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_56, _shape_param_32);  sum_dim_int_list_56 = _shape_param_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_32: "f32[1024, 32768]" = torch.ops.aten.permute.default(view_422, [1, 0])
        sum_dim_int_list_57: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_422, [0], True);  view_422 = None
        reshape_default_33: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_57, _shape_param_33);  sum_dim_int_list_57 = _shape_param_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_12: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_140, mul_100);  mul_100 = None
        sum_dim_int_list_58: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1]);  mul_tensor_12 = None
        sum_dim_int_list_59: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_140, [0, 1]);  add_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_33: "f32[256, 32768]" = torch.ops.aten.permute.default(view_425, [1, 0])
        sum_dim_int_list_60: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_425, [0], True);  view_425 = None
        reshape_default_34: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_60, _shape_param_34);  sum_dim_int_list_60 = _shape_param_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_34: "f32[256, 32768]" = torch.ops.aten.permute.default(view_436, [1, 0])
        sum_dim_int_list_61: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_436, [0], True);  view_436 = None
        reshape_default_35: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_61, _shape_param_35);  sum_dim_int_list_61 = _shape_param_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_35: "f32[256, 32768]" = torch.ops.aten.permute.default(view_440, [1, 0])
        sum_dim_int_list_62: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_440, [0], True);  view_440 = None
        reshape_default_36: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_62, _shape_param_36);  sum_dim_int_list_62 = _shape_param_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_36: "f32[256, 32768]" = torch.ops.aten.permute.default(view_444, [1, 0])
        sum_dim_int_list_63: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_444, [0], True);  view_444 = None
        reshape_default_37: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_63, _shape_param_37);  sum_dim_int_list_63 = _shape_param_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_13: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_143, mul_92);  mul_92 = None
        sum_dim_int_list_64: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1]);  mul_tensor_13 = None
        sum_dim_int_list_65: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_143, [0, 1]);  add_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_37: "f32[256, 32768]" = torch.ops.aten.permute.default(view_447, [1, 0])
        sum_dim_int_list_66: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_447, [0], True);  view_447 = None
        reshape_default_38: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_66, _shape_param_38);  sum_dim_int_list_66 = _shape_param_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_38: "f32[1024, 32768]" = torch.ops.aten.permute.default(view_450, [1, 0])
        sum_dim_int_list_67: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_450, [0], True);  view_450 = None
        reshape_default_39: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_67, _shape_param_39);  sum_dim_int_list_67 = _shape_param_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_14: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_146, mul_85);  mul_85 = None
        sum_dim_int_list_68: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1]);  mul_tensor_14 = None
        sum_dim_int_list_69: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_146, [0, 1]);  add_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_39: "f32[256, 32768]" = torch.ops.aten.permute.default(view_453, [1, 0])
        sum_dim_int_list_70: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_453, [0], True);  view_453 = None
        reshape_default_40: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_70, _shape_param_40);  sum_dim_int_list_70 = _shape_param_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_40: "f32[256, 32768]" = torch.ops.aten.permute.default(view_464, [1, 0])
        sum_dim_int_list_71: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_464, [0], True);  view_464 = None
        reshape_default_41: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_71, _shape_param_41);  sum_dim_int_list_71 = _shape_param_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_41: "f32[256, 32768]" = torch.ops.aten.permute.default(view_468, [1, 0])
        sum_dim_int_list_72: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_468, [0], True);  view_468 = None
        reshape_default_42: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_72, _shape_param_42);  sum_dim_int_list_72 = _shape_param_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_42: "f32[256, 32768]" = torch.ops.aten.permute.default(view_472, [1, 0])
        sum_dim_int_list_73: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_472, [0], True);  view_472 = None
        reshape_default_43: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_73, _shape_param_43);  sum_dim_int_list_73 = _shape_param_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_15: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_149, mul_77);  mul_77 = None
        sum_dim_int_list_74: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1]);  mul_tensor_15 = None
        sum_dim_int_list_75: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_149, [0, 1]);  add_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_43: "f32[256, 32768]" = torch.ops.aten.permute.default(view_475, [1, 0])
        sum_dim_int_list_76: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_475, [0], True);  view_475 = None
        reshape_default_44: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_76, _shape_param_44);  sum_dim_int_list_76 = _shape_param_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_44: "f32[1024, 32768]" = torch.ops.aten.permute.default(view_478, [1, 0])
        sum_dim_int_list_77: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_478, [0], True);  view_478 = None
        reshape_default_45: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_77, _shape_param_45);  sum_dim_int_list_77 = _shape_param_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_16: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_152, mul_70);  mul_70 = None
        sum_dim_int_list_78: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1]);  mul_tensor_16 = None
        sum_dim_int_list_79: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_152, [0, 1]);  add_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_45: "f32[256, 32768]" = torch.ops.aten.permute.default(view_481, [1, 0])
        sum_dim_int_list_80: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_481, [0], True);  view_481 = None
        reshape_default_46: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_80, _shape_param_46);  sum_dim_int_list_80 = _shape_param_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_46: "f32[256, 32768]" = torch.ops.aten.permute.default(view_492, [1, 0])
        sum_dim_int_list_81: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_492, [0], True);  view_492 = None
        reshape_default_47: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_81, _shape_param_47);  sum_dim_int_list_81 = _shape_param_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_47: "f32[256, 32768]" = torch.ops.aten.permute.default(view_496, [1, 0])
        sum_dim_int_list_82: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_496, [0], True);  view_496 = None
        reshape_default_48: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_82, _shape_param_48);  sum_dim_int_list_82 = _shape_param_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_48: "f32[256, 32768]" = torch.ops.aten.permute.default(view_500, [1, 0])
        sum_dim_int_list_83: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_500, [0], True);  view_500 = None
        reshape_default_49: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_83, _shape_param_49);  sum_dim_int_list_83 = _shape_param_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_17: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_155, mul_62);  mul_62 = None
        sum_dim_int_list_84: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1]);  mul_tensor_17 = None
        sum_dim_int_list_85: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_155, [0, 1]);  add_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_49: "f32[256, 32768]" = torch.ops.aten.permute.default(view_503, [1, 0])
        sum_dim_int_list_86: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_503, [0], True);  view_503 = None
        reshape_default_50: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_86, _shape_param_50);  sum_dim_int_list_86 = _shape_param_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_50: "f32[1024, 32768]" = torch.ops.aten.permute.default(view_506, [1, 0])
        sum_dim_int_list_87: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_506, [0], True);  view_506 = None
        reshape_default_51: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_87, _shape_param_51);  sum_dim_int_list_87 = _shape_param_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_18: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_158, mul_55);  mul_55 = None
        sum_dim_int_list_88: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1]);  mul_tensor_18 = None
        sum_dim_int_list_89: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_158, [0, 1]);  add_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_51: "f32[256, 32768]" = torch.ops.aten.permute.default(view_509, [1, 0])
        sum_dim_int_list_90: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_509, [0], True);  view_509 = None
        reshape_default_52: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_90, _shape_param_52);  sum_dim_int_list_90 = _shape_param_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_52: "f32[256, 32768]" = torch.ops.aten.permute.default(view_520, [1, 0])
        sum_dim_int_list_91: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_520, [0], True);  view_520 = None
        reshape_default_53: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_91, _shape_param_53);  sum_dim_int_list_91 = _shape_param_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_53: "f32[256, 32768]" = torch.ops.aten.permute.default(view_524, [1, 0])
        sum_dim_int_list_92: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_524, [0], True);  view_524 = None
        reshape_default_54: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_92, _shape_param_54);  sum_dim_int_list_92 = _shape_param_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_54: "f32[256, 32768]" = torch.ops.aten.permute.default(view_528, [1, 0])
        sum_dim_int_list_93: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_528, [0], True);  view_528 = None
        reshape_default_55: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_93, _shape_param_55);  sum_dim_int_list_93 = _shape_param_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_19: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_161, mul_47);  mul_47 = None
        sum_dim_int_list_94: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1]);  mul_tensor_19 = None
        sum_dim_int_list_95: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_161, [0, 1]);  add_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_55: "f32[256, 32768]" = torch.ops.aten.permute.default(view_531, [1, 0])
        sum_dim_int_list_96: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_531, [0], True);  view_531 = None
        reshape_default_56: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_96, _shape_param_56);  sum_dim_int_list_96 = _shape_param_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_56: "f32[1024, 32768]" = torch.ops.aten.permute.default(view_534, [1, 0])
        sum_dim_int_list_97: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_534, [0], True);  view_534 = None
        reshape_default_57: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_97, _shape_param_57);  sum_dim_int_list_97 = _shape_param_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_20: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_164, mul_40);  mul_40 = None
        sum_dim_int_list_98: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1]);  mul_tensor_20 = None
        sum_dim_int_list_99: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_164, [0, 1]);  add_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_57: "f32[256, 32768]" = torch.ops.aten.permute.default(view_537, [1, 0])
        sum_dim_int_list_100: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_537, [0], True);  view_537 = None
        reshape_default_58: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_100, _shape_param_58);  sum_dim_int_list_100 = _shape_param_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_58: "f32[256, 32768]" = torch.ops.aten.permute.default(view_548, [1, 0])
        sum_dim_int_list_101: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_548, [0], True);  view_548 = None
        reshape_default_59: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_101, _shape_param_59);  sum_dim_int_list_101 = _shape_param_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_59: "f32[256, 32768]" = torch.ops.aten.permute.default(view_552, [1, 0])
        sum_dim_int_list_102: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_552, [0], True);  view_552 = None
        reshape_default_60: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_102, _shape_param_60);  sum_dim_int_list_102 = _shape_param_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_60: "f32[256, 32768]" = torch.ops.aten.permute.default(view_556, [1, 0])
        sum_dim_int_list_103: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_556, [0], True);  view_556 = None
        reshape_default_61: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_103, _shape_param_61);  sum_dim_int_list_103 = _shape_param_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_21: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_167, mul_32);  mul_32 = None
        sum_dim_int_list_104: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1]);  mul_tensor_21 = None
        sum_dim_int_list_105: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_167, [0, 1]);  add_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_61: "f32[256, 32768]" = torch.ops.aten.permute.default(view_559, [1, 0])
        sum_dim_int_list_106: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_559, [0], True);  view_559 = None
        reshape_default_62: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_106, _shape_param_62);  sum_dim_int_list_106 = _shape_param_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_62: "f32[1024, 32768]" = torch.ops.aten.permute.default(view_562, [1, 0])
        sum_dim_int_list_107: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_562, [0], True);  view_562 = None
        reshape_default_63: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_107, _shape_param_63);  sum_dim_int_list_107 = _shape_param_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_22: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_170, mul_25);  mul_25 = None
        sum_dim_int_list_108: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0, 1]);  mul_tensor_22 = None
        sum_dim_int_list_109: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_170, [0, 1]);  add_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_63: "f32[256, 32768]" = torch.ops.aten.permute.default(view_565, [1, 0])
        sum_dim_int_list_110: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_565, [0], True);  view_565 = None
        reshape_default_64: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_110, _shape_param_64);  sum_dim_int_list_110 = _shape_param_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_64: "f32[256, 32768]" = torch.ops.aten.permute.default(view_576, [1, 0])
        sum_dim_int_list_111: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_576, [0], True);  view_576 = None
        reshape_default_65: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_111, _shape_param_65);  sum_dim_int_list_111 = _shape_param_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_65: "f32[256, 32768]" = torch.ops.aten.permute.default(view_580, [1, 0])
        sum_dim_int_list_112: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_580, [0], True);  view_580 = None
        reshape_default_66: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_112, _shape_param_66);  sum_dim_int_list_112 = _shape_param_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_66: "f32[256, 32768]" = torch.ops.aten.permute.default(view_584, [1, 0])
        sum_dim_int_list_113: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_584, [0], True);  view_584 = None
        reshape_default_67: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_113, _shape_param_67);  sum_dim_int_list_113 = _shape_param_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_23: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_173, mul_17);  mul_17 = None
        sum_dim_int_list_114: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1]);  mul_tensor_23 = None
        sum_dim_int_list_115: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_173, [0, 1]);  add_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_67: "f32[256, 32768]" = torch.ops.aten.permute.default(view_587, [1, 0])
        sum_dim_int_list_116: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_587, [0], True);  view_587 = None
        reshape_default_68: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_116, _shape_param_68);  sum_dim_int_list_116 = _shape_param_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_68: "f32[1024, 32768]" = torch.ops.aten.permute.default(view_590, [1, 0])
        sum_dim_int_list_117: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_590, [0], True);  view_590 = None
        reshape_default_69: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_117, _shape_param_69);  sum_dim_int_list_117 = _shape_param_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_24: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(add_176, mul_10);  mul_10 = None
        sum_dim_int_list_118: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_24, [0, 1]);  mul_tensor_24 = None
        sum_dim_int_list_119: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_176, [0, 1]);  add_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_69: "f32[256, 32768]" = torch.ops.aten.permute.default(view_593, [1, 0])
        sum_dim_int_list_120: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_593, [0], True);  view_593 = None
        reshape_default_70: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_120, _shape_param_70);  sum_dim_int_list_120 = _shape_param_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_70: "f32[256, 32768]" = torch.ops.aten.permute.default(view_604, [1, 0])
        sum_dim_int_list_121: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_604, [0], True);  view_604 = None
        reshape_default_71: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_121, _shape_param_71);  sum_dim_int_list_121 = _shape_param_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_71: "f32[256, 32768]" = torch.ops.aten.permute.default(view_608, [1, 0])
        sum_dim_int_list_122: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_608, [0], True);  view_608 = None
        reshape_default_72: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_122, _shape_param_72);  sum_dim_int_list_122 = _shape_param_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_72: "f32[256, 32768]" = torch.ops.aten.permute.default(view_612, [1, 0])
        sum_dim_int_list_123: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_612, [0], True);  view_612 = None
        reshape_default_73: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_123, _shape_param_73);  sum_dim_int_list_123 = _shape_param_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:610 in forward, code: embedding_output = self.embeddings_project(embedding_output)
        permute_default_73: "f32[256, 32768]" = torch.ops.aten.permute.default(view_615, [1, 0])
        sum_dim_int_list_124: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_615, [0], True);  view_615 = None
        reshape_default_74: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_124, _shape_param_74);  sum_dim_int_list_124 = _shape_param_74 = None
        reshape_default_75: "f32[64, 512, 128]" = torch.ops.aten.reshape.default(mm_148, _shape_param_75);  mm_148 = _shape_param_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:116 in forward, code: embeddings = self.dropout(embeddings)
        convert_element_type_default: "f32[64, 512, 128]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor_25: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_26: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(reshape_default_75, mul_tensor_25);  reshape_default_75 = mul_tensor_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:115 in forward, code: embeddings = self.LayerNorm(embeddings)
        mul_tensor_27: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_26, primals_8);  primals_8 = None
        mul_tensor_28: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_27, 128)
        sum_dim_int_list_125: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [2], True)
        mul_tensor_29: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_27, mul);  mul_tensor_27 = None
        sum_dim_int_list_126: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_29, [2], True);  mul_tensor_29 = None
        mul_tensor_30: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul, sum_dim_int_list_126);  sum_dim_int_list_126 = None
        sub_tensor: "f32[64, 512, 128]" = torch.ops.aten.sub.Tensor(mul_tensor_28, sum_dim_int_list_125);  mul_tensor_28 = sum_dim_int_list_125 = None
        sub_tensor_1: "f32[64, 512, 128]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_30);  sub_tensor = mul_tensor_30 = None
        mul_tensor_31: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(div_39, sub_tensor_1);  div_39 = sub_tensor_1 = None
        mul_tensor_32: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_26, mul);  mul = None
        sum_dim_int_list_127: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_32, [0, 1]);  mul_tensor_32 = None
        sum_dim_int_list_128: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_26, [0, 1]);  mul_tensor_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:113 in forward, code: embeddings = embeddings + position_embeddings
        sum_dim_int_list_129: "f32[1, 512, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_31, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:112 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        eq_scalar: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_3, -1)
        unsqueeze_default: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[1, 512, 128]" = torch.ops.aten.where.self(unsqueeze_default, full_default_1, sum_dim_int_list_129);  unsqueeze_default = sum_dim_int_list_129 = None
        full_default: "f32[512, 128]" = torch.ops.aten.full.default([512, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[512, 128]" = torch.ops.aten.index_put.default(full_default, [primals_3], where_self, True);  full_default = primals_3 = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:103 in forward, code: token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_default: "i64[64, 512]" = torch.ops.aten.expand.default(gather, _shape_param_76);  gather = _shape_param_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:109 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        eq_scalar_1: "b8[64, 512]" = torch.ops.aten.eq.Scalar(expand_default, -1)
        unsqueeze_default_1: "b8[64, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[64, 512, 128]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default_1, mul_tensor_31);  unsqueeze_default_1 = None
        full_default_2: "f32[2, 128]" = torch.ops.aten.full.default([2, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[2, 128]" = torch.ops.aten.index_put.default(full_default_2, [expand_default], where_self_1, True);  full_default_2 = expand_default = where_self_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:108 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        eq_scalar_2: "b8[64, 512]" = torch.ops.aten.eq.Scalar(primals_2, 0)
        unsqueeze_default_2: "b8[64, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_2, -1);  eq_scalar_2 = None
        where_self_2: "f32[64, 512, 128]" = torch.ops.aten.where.self(unsqueeze_default_2, full_default_1, mul_tensor_31);  unsqueeze_default_2 = full_default_1 = mul_tensor_31 = None
        full_default_3: "f32[30522, 128]" = torch.ops.aten.full.default([30522, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_2: "f32[30522, 128]" = torch.ops.aten.index_put.default(full_default_3, [primals_2], where_self_2, True);  full_default_3 = primals_2 = where_self_2 = None
        add_tensor: "f32[30522, 128]" = torch.ops.aten.add.Tensor(mm_1, index_put_default_2);  mm_1 = index_put_default_2 = None
        return (reshape_default, sum_dim_int_list_1, sum_dim_int_list_2, permute_default, reshape_default_1, sum_dim_int_list_4, sum_dim_int_list_5, permute_default_1, reshape_default_2, permute_default_2, reshape_default_3, sum_dim_int_list_8, sum_dim_int_list_9, permute_default_3, reshape_default_4, permute_default_4, reshape_default_5, permute_default_5, reshape_default_6, permute_default_6, reshape_default_7, sum_dim_int_list_14, sum_dim_int_list_15, permute_default_7, reshape_default_8, permute_default_8, reshape_default_9, sum_dim_int_list_18, sum_dim_int_list_19, permute_default_9, reshape_default_10, permute_default_10, reshape_default_11, permute_default_11, reshape_default_12, permute_default_12, reshape_default_13, sum_dim_int_list_24, sum_dim_int_list_25, permute_default_13, reshape_default_14, permute_default_14, reshape_default_15, sum_dim_int_list_28, sum_dim_int_list_29, permute_default_15, reshape_default_16, permute_default_16, reshape_default_17, permute_default_17, reshape_default_18, permute_default_18, reshape_default_19, sum_dim_int_list_34, sum_dim_int_list_35, permute_default_19, reshape_default_20, permute_default_20, reshape_default_21, sum_dim_int_list_38, sum_dim_int_list_39, permute_default_21, reshape_default_22, permute_default_22, reshape_default_23, permute_default_23, reshape_default_24, permute_default_24, reshape_default_25, sum_dim_int_list_44, sum_dim_int_list_45, permute_default_25, reshape_default_26, permute_default_26, reshape_default_27, sum_dim_int_list_48, sum_dim_int_list_49, permute_default_27, reshape_default_28, permute_default_28, reshape_default_29, permute_default_29, reshape_default_30, permute_default_30, reshape_default_31, sum_dim_int_list_54, sum_dim_int_list_55, permute_default_31, reshape_default_32, permute_default_32, reshape_default_33, sum_dim_int_list_58, sum_dim_int_list_59, permute_default_33, reshape_default_34, permute_default_34, reshape_default_35, permute_default_35, reshape_default_36, permute_default_36, reshape_default_37, sum_dim_int_list_64, sum_dim_int_list_65, permute_default_37, reshape_default_38, permute_default_38, reshape_default_39, sum_dim_int_list_68, sum_dim_int_list_69, permute_default_39, reshape_default_40, permute_default_40, reshape_default_41, permute_default_41, reshape_default_42, permute_default_42, reshape_default_43, sum_dim_int_list_74, sum_dim_int_list_75, permute_default_43, reshape_default_44, permute_default_44, reshape_default_45, sum_dim_int_list_78, sum_dim_int_list_79, permute_default_45, reshape_default_46, permute_default_46, reshape_default_47, permute_default_47, reshape_default_48, permute_default_48, reshape_default_49, sum_dim_int_list_84, sum_dim_int_list_85, permute_default_49, reshape_default_50, permute_default_50, reshape_default_51, sum_dim_int_list_88, sum_dim_int_list_89, permute_default_51, reshape_default_52, permute_default_52, reshape_default_53, permute_default_53, reshape_default_54, permute_default_54, reshape_default_55, sum_dim_int_list_94, sum_dim_int_list_95, permute_default_55, reshape_default_56, permute_default_56, reshape_default_57, sum_dim_int_list_98, sum_dim_int_list_99, permute_default_57, reshape_default_58, permute_default_58, reshape_default_59, permute_default_59, reshape_default_60, permute_default_60, reshape_default_61, sum_dim_int_list_104, sum_dim_int_list_105, permute_default_61, reshape_default_62, permute_default_62, reshape_default_63, sum_dim_int_list_108, sum_dim_int_list_109, permute_default_63, reshape_default_64, permute_default_64, reshape_default_65, permute_default_65, reshape_default_66, permute_default_66, reshape_default_67, sum_dim_int_list_114, sum_dim_int_list_115, permute_default_67, reshape_default_68, permute_default_68, reshape_default_69, sum_dim_int_list_118, sum_dim_int_list_119, permute_default_69, reshape_default_70, permute_default_70, reshape_default_71, permute_default_71, reshape_default_72, permute_default_72, reshape_default_73, permute_default_73, reshape_default_74, sum_dim_int_list_127, sum_dim_int_list_128, index_put_default, index_put_default_1, add_tensor)


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
