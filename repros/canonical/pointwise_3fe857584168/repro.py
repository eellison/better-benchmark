"""
Standalone repro captured via capture_hook.
Label: mistral_7b
Pattern hash: 3fe857584168
Shape hash: d360b309
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, _tensor_constant0: "i64[]", _tensor_constant1: "i64[]", _tensor_constant2: "i64[]", _tensor_constant3: "i64[]", _tensor_constant4: "i64[]", _tensor_constant5: "i64[]", _tensor_constant6: "i64[]", _tensor_constant7: "i64[]", _tensor_constant8: "i64[]", _tensor_constant9: "i64[]", _tensor_constant10: "i64[]", _tensor_constant11: "i64[]", _tensor_constant12: "i64[]", _tensor_constant13: "i64[]", _tensor_constant14: "i64[]", _tensor_constant15: "i64[]", _tensor_constant16: "i64[]", _tensor_constant17: "i64[]", _tensor_constant18: "i64[]", _tensor_constant19: "i64[]", _tensor_constant20: "i64[]", _tensor_constant21: "i64[]", _tensor_constant22: "i64[]", _tensor_constant23: "i64[]", _tensor_constant24: "i64[]", _tensor_constant25: "i64[]", _tensor_constant26: "i64[]", _tensor_constant27: "i64[]", _tensor_constant28: "i64[]", _tensor_constant29: "i64[]", _tensor_constant30: "i64[]", _tensor_constant31: "i64[]", permute_6: "f16[4, 8, 256, 128]", add_5: "f16[4, 8, 256, 128]", permute_17: "f16[4, 8, 256, 128]", add_12: "f16[4, 8, 256, 128]", permute_28: "f16[4, 8, 256, 128]", add_19: "f16[4, 8, 256, 128]", permute_39: "f16[4, 8, 256, 128]", add_26: "f16[4, 8, 256, 128]", permute_50: "f16[4, 8, 256, 128]", add_33: "f16[4, 8, 256, 128]", permute_61: "f16[4, 8, 256, 128]", add_40: "f16[4, 8, 256, 128]", permute_72: "f16[4, 8, 256, 128]", add_47: "f16[4, 8, 256, 128]", permute_83: "f16[4, 8, 256, 128]", add_54: "f16[4, 8, 256, 128]", permute_94: "f16[4, 8, 256, 128]", add_61: "f16[4, 8, 256, 128]", permute_105: "f16[4, 8, 256, 128]", add_68: "f16[4, 8, 256, 128]", permute_116: "f16[4, 8, 256, 128]", add_75: "f16[4, 8, 256, 128]", permute_127: "f16[4, 8, 256, 128]", add_82: "f16[4, 8, 256, 128]", permute_138: "f16[4, 8, 256, 128]", add_89: "f16[4, 8, 256, 128]", permute_149: "f16[4, 8, 256, 128]", add_96: "f16[4, 8, 256, 128]", permute_160: "f16[4, 8, 256, 128]", add_103: "f16[4, 8, 256, 128]", permute_171: "f16[4, 8, 256, 128]", add_110: "f16[4, 8, 256, 128]", permute_182: "f16[4, 8, 256, 128]", add_117: "f16[4, 8, 256, 128]", permute_193: "f16[4, 8, 256, 128]", add_124: "f16[4, 8, 256, 128]", permute_204: "f16[4, 8, 256, 128]", add_131: "f16[4, 8, 256, 128]", permute_215: "f16[4, 8, 256, 128]", add_138: "f16[4, 8, 256, 128]", permute_226: "f16[4, 8, 256, 128]", add_145: "f16[4, 8, 256, 128]", permute_237: "f16[4, 8, 256, 128]", add_152: "f16[4, 8, 256, 128]", permute_248: "f16[4, 8, 256, 128]", add_159: "f16[4, 8, 256, 128]", permute_259: "f16[4, 8, 256, 128]", add_166: "f16[4, 8, 256, 128]", permute_270: "f16[4, 8, 256, 128]", add_173: "f16[4, 8, 256, 128]", permute_281: "f16[4, 8, 256, 128]", add_180: "f16[4, 8, 256, 128]", permute_292: "f16[4, 8, 256, 128]", add_187: "f16[4, 8, 256, 128]", permute_303: "f16[4, 8, 256, 128]", add_194: "f16[4, 8, 256, 128]", permute_314: "f16[4, 8, 256, 128]", add_201: "f16[4, 8, 256, 128]", permute_325: "f16[4, 8, 256, 128]", add_208: "f16[4, 8, 256, 128]", permute_336: "f16[4, 8, 256, 128]", add_215: "f16[4, 8, 256, 128]", permute_347: "f16[4, 8, 256, 128]", add_222: "f16[4, 8, 256, 128]", mm_224: "f16[1024, 32000]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:208 in __init__, code: self._sliding_window_tensor = torch.tensor(self.sliding_window, dtype=torch.long)
        lift_fresh_copy_default: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        lift_fresh_copy_default_1: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
        lift_fresh_copy_default_2: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant2);  _tensor_constant2 = None
        lift_fresh_copy_default_3: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant3);  _tensor_constant3 = None
        lift_fresh_copy_default_4: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant4);  _tensor_constant4 = None
        lift_fresh_copy_default_5: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant5);  _tensor_constant5 = None
        lift_fresh_copy_default_6: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant6);  _tensor_constant6 = None
        lift_fresh_copy_default_7: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant7);  _tensor_constant7 = None
        lift_fresh_copy_default_8: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant8);  _tensor_constant8 = None
        lift_fresh_copy_default_9: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant9);  _tensor_constant9 = None
        lift_fresh_copy_default_10: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant10);  _tensor_constant10 = None
        lift_fresh_copy_default_11: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant11);  _tensor_constant11 = None
        lift_fresh_copy_default_12: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant12);  _tensor_constant12 = None
        lift_fresh_copy_default_13: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant13);  _tensor_constant13 = None
        lift_fresh_copy_default_14: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant14);  _tensor_constant14 = None
        lift_fresh_copy_default_15: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant15);  _tensor_constant15 = None
        lift_fresh_copy_default_16: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant16);  _tensor_constant16 = None
        lift_fresh_copy_default_17: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant17);  _tensor_constant17 = None
        lift_fresh_copy_default_18: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant18);  _tensor_constant18 = None
        lift_fresh_copy_default_19: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant19);  _tensor_constant19 = None
        lift_fresh_copy_default_20: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant20);  _tensor_constant20 = None
        lift_fresh_copy_default_21: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant21);  _tensor_constant21 = None
        lift_fresh_copy_default_22: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant22);  _tensor_constant22 = None
        lift_fresh_copy_default_23: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant23);  _tensor_constant23 = None
        lift_fresh_copy_default_24: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant24);  _tensor_constant24 = None
        lift_fresh_copy_default_25: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant25);  _tensor_constant25 = None
        lift_fresh_copy_default_26: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant26);  _tensor_constant26 = None
        lift_fresh_copy_default_27: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant27);  _tensor_constant27 = None
        lift_fresh_copy_default_28: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant28);  _tensor_constant28 = None
        lift_fresh_copy_default_29: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant29);  _tensor_constant29 = None
        lift_fresh_copy_default_30: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant30);  _tensor_constant30 = None
        lift_fresh_copy_default_31: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant31);  _tensor_constant31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_6, 2, -4095, 9223372036854775807);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_1: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_5, 2, -4095, 9223372036854775807);  add_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_1: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_2: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_17, 2, -4095, 9223372036854775807);  permute_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_3: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_12, 2, -4095, 9223372036854775807);  add_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_2: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_4: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_28, 2, -4095, 9223372036854775807);  permute_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_5: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_19, 2, -4095, 9223372036854775807);  add_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_3: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_6: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_39, 2, -4095, 9223372036854775807);  permute_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_7: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_26, 2, -4095, 9223372036854775807);  add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_4: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_8: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_50, 2, -4095, 9223372036854775807);  permute_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_9: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_33, 2, -4095, 9223372036854775807);  add_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_5: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_10: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_61, 2, -4095, 9223372036854775807);  permute_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_11: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_40, 2, -4095, 9223372036854775807);  add_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_6: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_12: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_72, 2, -4095, 9223372036854775807);  permute_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_13: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_47, 2, -4095, 9223372036854775807);  add_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_7: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_14: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_83, 2, -4095, 9223372036854775807);  permute_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_15: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_54, 2, -4095, 9223372036854775807);  add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_8: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_16: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_94, 2, -4095, 9223372036854775807);  permute_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_17: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_61, 2, -4095, 9223372036854775807);  add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_9: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_18: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_105, 2, -4095, 9223372036854775807);  permute_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_19: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_68, 2, -4095, 9223372036854775807);  add_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_10: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_20: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_116, 2, -4095, 9223372036854775807);  permute_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_21: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_75, 2, -4095, 9223372036854775807);  add_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_11: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_22: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_127, 2, -4095, 9223372036854775807);  permute_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_23: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_82, 2, -4095, 9223372036854775807);  add_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_12: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_24: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_138, 2, -4095, 9223372036854775807);  permute_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_25: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_89, 2, -4095, 9223372036854775807);  add_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_13: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_26: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_149, 2, -4095, 9223372036854775807);  permute_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_27: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_96, 2, -4095, 9223372036854775807);  add_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_14: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_28: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_160, 2, -4095, 9223372036854775807);  permute_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_29: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_103, 2, -4095, 9223372036854775807);  add_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_15: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_30: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_171, 2, -4095, 9223372036854775807);  permute_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_31: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_110, 2, -4095, 9223372036854775807);  add_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_16: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_32: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_182, 2, -4095, 9223372036854775807);  permute_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_33: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_117, 2, -4095, 9223372036854775807);  add_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_17: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_34: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_193, 2, -4095, 9223372036854775807);  permute_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_35: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_124, 2, -4095, 9223372036854775807);  add_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_18: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_36: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_204, 2, -4095, 9223372036854775807);  permute_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_37: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_131, 2, -4095, 9223372036854775807);  add_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_19: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_38: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_215, 2, -4095, 9223372036854775807);  permute_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_39: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_138, 2, -4095, 9223372036854775807);  add_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_20: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_40: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_226, 2, -4095, 9223372036854775807);  permute_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_41: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_145, 2, -4095, 9223372036854775807);  add_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_21: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_42: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_237, 2, -4095, 9223372036854775807);  permute_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_43: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_152, 2, -4095, 9223372036854775807);  add_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_22: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_44: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_248, 2, -4095, 9223372036854775807);  permute_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_45: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_159, 2, -4095, 9223372036854775807);  add_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_23: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_46: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_259, 2, -4095, 9223372036854775807);  permute_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_47: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_166, 2, -4095, 9223372036854775807);  add_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_24: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_48: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_270, 2, -4095, 9223372036854775807);  permute_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_49: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_173, 2, -4095, 9223372036854775807);  add_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_25: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_50: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_281, 2, -4095, 9223372036854775807);  permute_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_51: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_180, 2, -4095, 9223372036854775807);  add_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_26: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_52: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_292, 2, -4095, 9223372036854775807);  permute_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_53: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_187, 2, -4095, 9223372036854775807);  add_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_27: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_54: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_303, 2, -4095, 9223372036854775807);  permute_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_55: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_194, 2, -4095, 9223372036854775807);  add_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_28: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_56: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_314, 2, -4095, 9223372036854775807);  permute_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_57: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_201, 2, -4095, 9223372036854775807);  add_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_29: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_58: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_325, 2, -4095, 9223372036854775807);  permute_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_59: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_208, 2, -4095, 9223372036854775807);  add_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_30: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_60: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_336, 2, -4095, 9223372036854775807);  permute_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_61: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_215, 2, -4095, 9223372036854775807);  add_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:212 in lazy_initialization, code: self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)
        full_default_31: "i64[]" = torch.ops.aten.full.default([], 4096, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:238 in update, code: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_62: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(permute_347, 2, -4095, 9223372036854775807);  permute_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:237 in update, code: self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        slice_tensor_63: "f16[4, 8, 256, 128]" = torch.ops.aten.slice.Tensor(add_222, 2, -4095, 9223372036854775807);  add_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:460 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        reshape_default: "f16[4, 256, 32000]" = torch.ops.aten.reshape.default(mm_224, _shape_param_0);  mm_224 = _shape_param_0 = None
        return (lift_fresh_copy_default, lift_fresh_copy_default_1, lift_fresh_copy_default_2, lift_fresh_copy_default_3, lift_fresh_copy_default_4, lift_fresh_copy_default_5, lift_fresh_copy_default_6, lift_fresh_copy_default_7, lift_fresh_copy_default_8, lift_fresh_copy_default_9, lift_fresh_copy_default_10, lift_fresh_copy_default_11, lift_fresh_copy_default_12, lift_fresh_copy_default_13, lift_fresh_copy_default_14, lift_fresh_copy_default_15, lift_fresh_copy_default_16, lift_fresh_copy_default_17, lift_fresh_copy_default_18, lift_fresh_copy_default_19, lift_fresh_copy_default_20, lift_fresh_copy_default_21, lift_fresh_copy_default_22, lift_fresh_copy_default_23, lift_fresh_copy_default_24, lift_fresh_copy_default_25, lift_fresh_copy_default_26, lift_fresh_copy_default_27, lift_fresh_copy_default_28, lift_fresh_copy_default_29, lift_fresh_copy_default_30, lift_fresh_copy_default_31, full_default, slice_tensor, slice_tensor_1, full_default_1, slice_tensor_2, slice_tensor_3, full_default_2, slice_tensor_4, slice_tensor_5, full_default_3, slice_tensor_6, slice_tensor_7, full_default_4, slice_tensor_8, slice_tensor_9, full_default_5, slice_tensor_10, slice_tensor_11, full_default_6, slice_tensor_12, slice_tensor_13, full_default_7, slice_tensor_14, slice_tensor_15, full_default_8, slice_tensor_16, slice_tensor_17, full_default_9, slice_tensor_18, slice_tensor_19, full_default_10, slice_tensor_20, slice_tensor_21, full_default_11, slice_tensor_22, slice_tensor_23, full_default_12, slice_tensor_24, slice_tensor_25, full_default_13, slice_tensor_26, slice_tensor_27, full_default_14, slice_tensor_28, slice_tensor_29, full_default_15, slice_tensor_30, slice_tensor_31, full_default_16, slice_tensor_32, slice_tensor_33, full_default_17, slice_tensor_34, slice_tensor_35, full_default_18, slice_tensor_36, slice_tensor_37, full_default_19, slice_tensor_38, slice_tensor_39, full_default_20, slice_tensor_40, slice_tensor_41, full_default_21, slice_tensor_42, slice_tensor_43, full_default_22, slice_tensor_44, slice_tensor_45, full_default_23, slice_tensor_46, slice_tensor_47, full_default_24, slice_tensor_48, slice_tensor_49, full_default_25, slice_tensor_50, slice_tensor_51, full_default_26, slice_tensor_52, slice_tensor_53, full_default_27, slice_tensor_54, slice_tensor_55, full_default_28, slice_tensor_56, slice_tensor_57, full_default_29, slice_tensor_58, slice_tensor_59, full_default_30, slice_tensor_60, slice_tensor_61, full_default_31, slice_tensor_62, slice_tensor_63, reshape_default)


def _default_make_inputs():
    return [
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cpu'),
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_6
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_5
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_17
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_12
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_28
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_19
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_39
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_26
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_50
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_33
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_61
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_40
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_72
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_47
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_83
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_54
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_94
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_61
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_105
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_68
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_116
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_75
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_127
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_82
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_138
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_89
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_149
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_96
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_160
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_103
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_171
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_110
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_182
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_117
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_193
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_124
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_204
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_131
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_215
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_138
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_226
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_145
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_237
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_152
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_248
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_159
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_259
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_166
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_270
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_173
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_281
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_180
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_292
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_187
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_303
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_194
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_314
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_201
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_325
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_208
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_336
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_215
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # permute_347
    torch.randn(1048576, dtype=torch.float16, device='cuda').as_strided([4, 8, 256, 128], [262144, 128, 1024, 1]),  # add_222
    torch.randn([1024, 32000], dtype=torch.float16, device='cuda'),
    [4, 256, 32000],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
