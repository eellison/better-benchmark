"""
Standalone repro captured via capture_hook.
Label: torchbench_modded_nanogpt_train
Pattern hash: 494deec8b0a2
Shape hash: 5f1bc729
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([2], f32), T([1, 6144, 6, 128], bf16), T([6144, 768], bf16), T([1, 6144, 6, 128], bf16, stride=(14155776, 2304, 128, 1)), T([2], f32), T([1, 6144, 768], bf16), T([6144, 768], bf16), T([1, 6144, 1], f32), T([1, 6144, 768], bf16), T([1, 6144, 768], bf16), T([1, 6144, 768], bf16), T([2], f32), T([1, 6144, 6, 128], bf16), T([6144, 768], bf16), T([1, 6144, 6, 128], bf16, stride=(14155776, 2304, 128, 1)), T([2], f32), T([1, 6144, 768], bf16), T([1, 6144, 768], bf16), T([1, 6144, 768], bf16), T([1, 6144, 768], bf16), T([2], f32), T([1, 6144, 6, 128], bf16), T([6144, 768], bf16), T([1, 6144, 6, 128], bf16, stride=(14155776, 2304, 128, 1)), T([2], f32), T([1, 6144, 768], bf16), T([1, 6144, 768], bf16), T([1, 6144, 768], bf16), T([1, 6144, 768], bf16), T([1, 6144, 6, 128], bf16), T([1, 6144, 6, 128], bf16, stride=(14155776, 2304, 128, 1)), T([2], f32), T([1, 6144, 768], bf16), T([1, 6144, 768], bf16), T([1, 6144, 768], bf16), T([1, 6144, 768], bf16), T([2], f32), T([1, 6144, 768], bf16), T([1, 6144, 768], bf16), T([1, 6144, 768], bf16), T([1, 6144, 768], bf16), T([1, 6144, 6, 128], bf16), T([1, 6144, 6, 128], bf16, stride=(14155776, 2304, 128, 1)), T([2], f32), T([1, 6144, 768], bf16), T([], f32), T([1, 6144, 768], bf16), T([1, 6144, 768], bf16), T([1, 6144, 6, 128], bf16), T([1, 6144, 6, 128], bf16, stride=(14155776, 2304, 128, 1)), T([2], f32), T([1, 6144, 768], bf16), T([1, 6144, 6, 128], bf16), T([1, 6144, 6, 128], bf16, stride=(14155776, 2304, 128, 1)), T([2], f32), T([1, 6144, 768], bf16), T([1, 6144, 6, 128], bf16), T([1, 6144, 6, 128], bf16, stride=(14155776, 2304, 128, 1)), T([2], f32), T([1, 6144, 768], bf16), T([2], f32), T([1, 6144, 6, 128], bf16), T([1, 6144, 6, 128], bf16, stride=(14155776, 2304, 128, 1)), T([2], f32), T([1, 6144, 768], bf16), T([2], f32), T([1, 6144, 6, 128], bf16), T([1, 6144, 6, 128], bf16, stride=(14155776, 2304, 128, 1)), T([2], f32), T([1, 6144, 768], bf16), T([6144, 12], bf16), T([1, 6144, 768], bf16), T([2], f32), T([1, 6144, 6, 128], bf16), T([1, 6144, 6, 128], bf16, stride=(14155776, 2304, 128, 1)), T([6144, 768], bf16), T([12, 2], f32), T([1, 6144, 1], f32), T([1, 6144, 768], bf16), T([6144], i32), T([], f32), S([1, 6144, 6, 128]), S([6144, 768]), S([1, 6144, 6, 128]), S([6144, 768]), S([1, 6144, 6, 128]), S([6144, 768]), S([6144, 768]), S([6144, 768]), S([1, 6144, 12]), S([6144, 768]), S([1, 6144, 768]), S([24]), S([24]))"

class Repro(torch.nn.Module):
    def forward(self, select_77: "f32[2]", permute_134: "bf16[1, 6144, 6, 128]", embedding_2: "bf16[6144, 768]", getitem_114: "bf16[1, 6144, 6, 128]", select_76: "f32[2]", add_146: "bf16[1, 6144, 768]", embedding_3: "bf16[6144, 768]", rsqrt: "f32[1, 6144, 1]", add_124: "bf16[1, 6144, 768]", mul_239: "bf16[1, 6144, 768]", add_12: "bf16[1, 6144, 768]", select_69: "f32[2]", permute_158: "bf16[1, 6144, 6, 128]", embedding_1: "bf16[6144, 768]", getitem_104: "bf16[1, 6144, 6, 128]", select_68: "f32[2]", add_156: "bf16[1, 6144, 768]", add_111: "bf16[1, 6144, 768]", mul_279: "bf16[1, 6144, 768]", add_24: "bf16[1, 6144, 768]", select_61: "f32[2]", permute_182: "bf16[1, 6144, 6, 128]", embedding: "bf16[6144, 768]", getitem_94: "bf16[1, 6144, 6, 128]", select_60: "f32[2]", add_170: "bf16[1, 6144, 768]", add_98: "bf16[1, 6144, 768]", mul_319: "bf16[1, 6144, 768]", add_36: "bf16[1, 6144, 768]", permute_206: "bf16[1, 6144, 6, 128]", getitem_84: "bf16[1, 6144, 6, 128]", select_53: "f32[2]", add_183: "bf16[1, 6144, 768]", add_86: "bf16[1, 6144, 768]", mul_357: "bf16[1, 6144, 768]", add_47: "bf16[1, 6144, 768]", select_49: "f32[2]", add_189: "bf16[1, 6144, 768]", add_82: "bf16[1, 6144, 768]", mul_369: "bf16[1, 6144, 768]", add_58: "bf16[1, 6144, 768]", permute_238: "bf16[1, 6144, 6, 128]", getitem_74: "bf16[1, 6144, 6, 128]", select_42: "f32[2]", add_201: "bf16[1, 6144, 768]", select_41: "f32[]", add_69: "bf16[1, 6144, 768]", mul_407: "bf16[1, 6144, 768]", permute_262: "bf16[1, 6144, 6, 128]", getitem_64: "bf16[1, 6144, 6, 128]", select_35: "f32[2]", add_215: "bf16[1, 6144, 768]", permute_286: "bf16[1, 6144, 6, 128]", getitem_54: "bf16[1, 6144, 6, 128]", select_29: "f32[2]", add_228: "bf16[1, 6144, 768]", permute_310: "bf16[1, 6144, 6, 128]", getitem_44: "bf16[1, 6144, 6, 128]", select_23: "f32[2]", add_241: "bf16[1, 6144, 768]", select_17: "f32[2]", permute_334: "bf16[1, 6144, 6, 128]", getitem_34: "bf16[1, 6144, 6, 128]", select_16: "f32[2]", add_256: "bf16[1, 6144, 768]", select_10: "f32[2]", permute_358: "bf16[1, 6144, 6, 128]", getitem_24: "bf16[1, 6144, 6, 128]", select_9: "f32[2]", add_271: "bf16[1, 6144, 768]", mm_169: "bf16[6144, 12]", full_default_19: "bf16[1, 6144, 768]", select_3: "f32[2]", permute_382: "bf16[1, 6144, 6, 128]", getitem_14: "bf16[1, 6144, 6, 128]", mm_171: "bf16[6144, 768]", view_6: "f32[12, 2]", rsqrt_1: "f32[1, 6144, 1]", add_277: "bf16[1, 6144, 768]", primals_1: "i32[6144]", full_default_13: "f32[]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:685 in forward, code: v = lambdas[0] * v + lambdas[1] * ve.view_as(
        select_int: "f32[]" = torch.ops.aten.select.int(select_77, 0, 1);  select_77 = None
        mul_tensor: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_134, select_int);  select_int = None
        reshape_default: "bf16[1, 6144, 6, 128]" = torch.ops.aten.reshape.default(embedding_2, _shape_param_0);  embedding_2 = _shape_param_0 = None
        mul_tensor_1: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_134, reshape_default)
        sum_default: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_1);  mul_tensor_1 = None
        convert_element_type_default: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default, torch.float32);  sum_default = None
        reshape_default_1: "bf16[6144, 768]" = torch.ops.aten.reshape.default(mul_tensor, _shape_param_1);  mul_tensor = _shape_param_1 = None
        full_default: "f32[2]" = torch.ops.aten.full.default([2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_default: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default, 0, 1);  convert_element_type_default = None
        mul_tensor_2: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_134, getitem_114);  permute_134 = getitem_114 = None
        sum_default_1: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_2);  mul_tensor_2 = None
        convert_element_type_default_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_1, torch.float32);  sum_default_1 = None
        select_scatter_default_1: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_1, 0, 0);  convert_element_type_default_1 = None
        add_tensor: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_default, select_scatter_default_1);  select_scatter_default = select_scatter_default_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_int_1: "f32[]" = torch.ops.aten.select.int(select_76, 0, 1);  select_76 = None
        mul_tensor_3: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_146, select_int_1);  select_int_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:909 in forward, code: x = x0 = norm(self.embed(input_seq)[None])  # use of norm here by @Grad62304977
        unsqueeze_default: "bf16[1, 6144, 768]" = torch.ops.aten.unsqueeze.default(embedding_3, 0);  embedding_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_default_2: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(unsqueeze_default, torch.float32);  unsqueeze_default = None
        mul_tensor_4: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, rsqrt);  convert_element_type_default_2 = None
        convert_element_type_default_3: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        mul_tensor_5: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_146, convert_element_type_default_3)
        sum_default_2: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_5);  mul_tensor_5 = None
        convert_element_type_default_4: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_2, torch.float32);  sum_default_2 = None
        select_scatter_default_2: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_4, 0, 1);  convert_element_type_default_4 = None
        mul_tensor_6: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_146, add_124);  add_146 = add_124 = None
        sum_default_3: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_6);  mul_tensor_6 = None
        convert_element_type_default_5: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_3, torch.float32);  sum_default_3 = None
        select_scatter_default_3: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_5, 0, 0);  convert_element_type_default_5 = None
        add_tensor_1: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_default_2, select_scatter_default_3);  select_scatter_default_2 = select_scatter_default_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        full_default_14: "f32[12, 2]" = torch.ops.aten.full.default([12, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_default_4: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_14, add_tensor, 0, 11);  add_tensor = None
        select_scatter_default_5: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_14, add_tensor_1, 0, 11);  add_tensor_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:923 in forward, code: x = x + skip_weights[i - n] * skip_connections.pop()
        mul_tensor_7: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_239, add_12);  mul_239 = None
        sum_default_4: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_7);  mul_tensor_7 = None
        convert_element_type_default_6: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_4, torch.float32);  sum_default_4 = None
        full_default_15: "f32[6]" = torch.ops.aten.full.default([6], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_default_6: "f32[6]" = torch.ops.aten.select_scatter.default(full_default_15, convert_element_type_default_6, 0, 5);  convert_element_type_default_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:685 in forward, code: v = lambdas[0] * v + lambdas[1] * ve.view_as(
        select_int_2: "f32[]" = torch.ops.aten.select.int(select_69, 0, 1);  select_69 = None
        mul_tensor_8: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_158, select_int_2);  select_int_2 = None
        reshape_default_2: "bf16[1, 6144, 6, 128]" = torch.ops.aten.reshape.default(embedding_1, _shape_param_2);  embedding_1 = _shape_param_2 = None
        mul_tensor_9: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_158, reshape_default_2)
        sum_default_5: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_9);  mul_tensor_9 = None
        convert_element_type_default_7: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_5, torch.float32);  sum_default_5 = None
        reshape_default_3: "bf16[6144, 768]" = torch.ops.aten.reshape.default(mul_tensor_8, _shape_param_3);  mul_tensor_8 = _shape_param_3 = None
        select_scatter_default_7: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_7, 0, 1);  convert_element_type_default_7 = None
        mul_tensor_10: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_158, getitem_104);  permute_158 = getitem_104 = None
        sum_default_6: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_10);  mul_tensor_10 = None
        convert_element_type_default_8: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_6, torch.float32);  sum_default_6 = None
        select_scatter_default_8: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_8, 0, 0);  convert_element_type_default_8 = None
        add_tensor_2: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_default_7, select_scatter_default_8);  select_scatter_default_7 = select_scatter_default_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_int_3: "f32[]" = torch.ops.aten.select.int(select_68, 0, 1);  select_68 = None
        mul_tensor_11: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_156, select_int_3);  select_int_3 = None
        mul_tensor_12: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_156, convert_element_type_default_3)
        sum_default_7: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_12);  mul_tensor_12 = None
        convert_element_type_default_9: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_7, torch.float32);  sum_default_7 = None
        add_tensor_3: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(mul_tensor_3, mul_tensor_11);  mul_tensor_3 = mul_tensor_11 = None
        select_scatter_default_9: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_9, 0, 1);  convert_element_type_default_9 = None
        mul_tensor_13: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_156, add_111);  add_156 = add_111 = None
        sum_default_8: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_13);  mul_tensor_13 = None
        convert_element_type_default_10: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_8, torch.float32);  sum_default_8 = None
        select_scatter_default_10: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_10, 0, 0);  convert_element_type_default_10 = None
        add_tensor_4: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_default_9, select_scatter_default_10);  select_scatter_default_9 = select_scatter_default_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_scatter_default_11: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_14, add_tensor_2, 0, 10);  add_tensor_2 = None
        add_tensor_5: "f32[12, 2]" = torch.ops.aten.add.Tensor(select_scatter_default_4, select_scatter_default_11);  select_scatter_default_4 = select_scatter_default_11 = None
        select_scatter_default_12: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_14, add_tensor_4, 0, 10);  add_tensor_4 = None
        add_tensor_6: "f32[12, 2]" = torch.ops.aten.add.Tensor(select_scatter_default_5, select_scatter_default_12);  select_scatter_default_5 = select_scatter_default_12 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:923 in forward, code: x = x + skip_weights[i - n] * skip_connections.pop()
        mul_tensor_14: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_279, add_24);  mul_279 = None
        sum_default_9: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_14);  mul_tensor_14 = None
        convert_element_type_default_11: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_9, torch.float32);  sum_default_9 = None
        select_scatter_default_13: "f32[6]" = torch.ops.aten.select_scatter.default(full_default_15, convert_element_type_default_11, 0, 4);  convert_element_type_default_11 = None
        add_tensor_7: "f32[6]" = torch.ops.aten.add.Tensor(select_scatter_default_6, select_scatter_default_13);  select_scatter_default_6 = select_scatter_default_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:685 in forward, code: v = lambdas[0] * v + lambdas[1] * ve.view_as(
        select_int_4: "f32[]" = torch.ops.aten.select.int(select_61, 0, 1);  select_61 = None
        mul_tensor_15: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_182, select_int_4);  select_int_4 = None
        reshape_default_4: "bf16[1, 6144, 6, 128]" = torch.ops.aten.reshape.default(embedding, _shape_param_4);  embedding = _shape_param_4 = None
        mul_tensor_16: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_182, reshape_default_4)
        sum_default_10: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_16);  mul_tensor_16 = None
        convert_element_type_default_12: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_10, torch.float32);  sum_default_10 = None
        reshape_default_5: "bf16[6144, 768]" = torch.ops.aten.reshape.default(mul_tensor_15, _shape_param_5);  mul_tensor_15 = _shape_param_5 = None
        select_scatter_default_14: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_12, 0, 1);  convert_element_type_default_12 = None
        mul_tensor_17: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_182, getitem_94);  permute_182 = getitem_94 = None
        sum_default_11: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_17);  mul_tensor_17 = None
        convert_element_type_default_13: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_11, torch.float32);  sum_default_11 = None
        select_scatter_default_15: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_13, 0, 0);  convert_element_type_default_13 = None
        add_tensor_8: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_default_14, select_scatter_default_15);  select_scatter_default_14 = select_scatter_default_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_int_5: "f32[]" = torch.ops.aten.select.int(select_60, 0, 1);  select_60 = None
        mul_tensor_18: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_170, select_int_5);  select_int_5 = None
        mul_tensor_19: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_170, convert_element_type_default_3)
        sum_default_12: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_19);  mul_tensor_19 = None
        convert_element_type_default_14: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_12, torch.float32);  sum_default_12 = None
        add_tensor_9: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_tensor_3, mul_tensor_18);  add_tensor_3 = mul_tensor_18 = None
        select_scatter_default_16: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_14, 0, 1);  convert_element_type_default_14 = None
        mul_tensor_20: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_170, add_98);  add_170 = add_98 = None
        sum_default_13: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_20);  mul_tensor_20 = None
        convert_element_type_default_15: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_13, torch.float32);  sum_default_13 = None
        select_scatter_default_17: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_15, 0, 0);  convert_element_type_default_15 = None
        add_tensor_10: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_default_16, select_scatter_default_17);  select_scatter_default_16 = select_scatter_default_17 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_scatter_default_18: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_14, add_tensor_8, 0, 9);  add_tensor_8 = None
        add_tensor_11: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_tensor_5, select_scatter_default_18);  add_tensor_5 = select_scatter_default_18 = None
        select_scatter_default_19: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_14, add_tensor_10, 0, 9);  add_tensor_10 = None
        add_tensor_12: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_tensor_6, select_scatter_default_19);  add_tensor_6 = select_scatter_default_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:923 in forward, code: x = x + skip_weights[i - n] * skip_connections.pop()
        mul_tensor_21: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_319, add_36);  mul_319 = None
        sum_default_14: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_21);  mul_tensor_21 = None
        convert_element_type_default_16: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_14, torch.float32);  sum_default_14 = None
        select_scatter_default_20: "f32[6]" = torch.ops.aten.select_scatter.default(full_default_15, convert_element_type_default_16, 0, 3);  convert_element_type_default_16 = None
        add_tensor_13: "f32[6]" = torch.ops.aten.add.Tensor(add_tensor_7, select_scatter_default_20);  add_tensor_7 = select_scatter_default_20 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:689 in forward, code: v = lambdas[0] * v
        mul_tensor_22: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_206, getitem_84);  permute_206 = getitem_84 = None
        sum_default_15: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_22);  mul_tensor_22 = None
        convert_element_type_default_17: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_15, torch.float32);  sum_default_15 = None
        select_scatter_default_21: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_17, 0, 0);  convert_element_type_default_17 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_int_6: "f32[]" = torch.ops.aten.select.int(select_53, 0, 1);  select_53 = None
        mul_tensor_23: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_183, select_int_6);  select_int_6 = None
        mul_tensor_24: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_183, convert_element_type_default_3)
        sum_default_16: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_24);  mul_tensor_24 = None
        convert_element_type_default_18: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_16, torch.float32);  sum_default_16 = None
        add_tensor_14: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_tensor_9, mul_tensor_23);  add_tensor_9 = mul_tensor_23 = None
        select_scatter_default_22: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_18, 0, 1);  convert_element_type_default_18 = None
        mul_tensor_25: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_183, add_86);  add_183 = add_86 = None
        sum_default_17: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_25);  mul_tensor_25 = None
        convert_element_type_default_19: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_17, torch.float32);  sum_default_17 = None
        select_scatter_default_23: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_19, 0, 0);  convert_element_type_default_19 = None
        add_tensor_15: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_default_22, select_scatter_default_23);  select_scatter_default_22 = select_scatter_default_23 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_scatter_default_24: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_14, select_scatter_default_21, 0, 8);  select_scatter_default_21 = None
        add_tensor_16: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_tensor_11, select_scatter_default_24);  add_tensor_11 = select_scatter_default_24 = None
        select_scatter_default_25: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_14, add_tensor_15, 0, 8);  add_tensor_15 = None
        add_tensor_17: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_tensor_12, select_scatter_default_25);  add_tensor_12 = select_scatter_default_25 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:923 in forward, code: x = x + skip_weights[i - n] * skip_connections.pop()
        mul_tensor_26: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_357, add_47);  mul_357 = None
        sum_default_18: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_26);  mul_tensor_26 = None
        convert_element_type_default_20: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_18, torch.float32);  sum_default_18 = None
        select_scatter_default_26: "f32[6]" = torch.ops.aten.select_scatter.default(full_default_15, convert_element_type_default_20, 0, 2);  convert_element_type_default_20 = None
        add_tensor_18: "f32[6]" = torch.ops.aten.add.Tensor(add_tensor_13, select_scatter_default_26);  add_tensor_13 = select_scatter_default_26 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_int_7: "f32[]" = torch.ops.aten.select.int(select_49, 0, 1);  select_49 = None
        mul_tensor_27: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_189, select_int_7);  select_int_7 = None
        mul_tensor_28: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_189, convert_element_type_default_3)
        sum_default_19: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_28);  mul_tensor_28 = None
        convert_element_type_default_21: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_19, torch.float32);  sum_default_19 = None
        add_tensor_19: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_tensor_14, mul_tensor_27);  add_tensor_14 = mul_tensor_27 = None
        select_scatter_default_27: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_21, 0, 1);  convert_element_type_default_21 = None
        mul_tensor_29: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_189, add_82);  add_189 = add_82 = None
        sum_default_20: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_29);  mul_tensor_29 = None
        convert_element_type_default_22: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_20, torch.float32);  sum_default_20 = None
        select_scatter_default_28: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_22, 0, 0);  convert_element_type_default_22 = None
        add_tensor_20: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_default_27, select_scatter_default_28);  select_scatter_default_27 = select_scatter_default_28 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_scatter_default_29: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_14, add_tensor_20, 0, 7);  add_tensor_20 = None
        add_tensor_21: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_tensor_17, select_scatter_default_29);  add_tensor_17 = select_scatter_default_29 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:923 in forward, code: x = x + skip_weights[i - n] * skip_connections.pop()
        mul_tensor_30: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_369, add_58);  mul_369 = None
        sum_default_21: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_30);  mul_tensor_30 = None
        convert_element_type_default_23: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_21, torch.float32);  sum_default_21 = None
        select_scatter_default_30: "f32[6]" = torch.ops.aten.select_scatter.default(full_default_15, convert_element_type_default_23, 0, 1);  convert_element_type_default_23 = None
        add_tensor_22: "f32[6]" = torch.ops.aten.add.Tensor(add_tensor_18, select_scatter_default_30);  add_tensor_18 = select_scatter_default_30 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:689 in forward, code: v = lambdas[0] * v
        mul_tensor_31: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_238, getitem_74);  permute_238 = getitem_74 = None
        sum_default_22: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_31);  mul_tensor_31 = None
        convert_element_type_default_24: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_22, torch.float32);  sum_default_22 = None
        select_scatter_default_31: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_24, 0, 0);  convert_element_type_default_24 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_int_8: "f32[]" = torch.ops.aten.select.int(select_42, 0, 1);  select_42 = None
        mul_tensor_32: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_201, select_int_8);  select_int_8 = None
        mul_tensor_33: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_201, convert_element_type_default_3)
        sum_default_23: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_33);  mul_tensor_33 = None
        convert_element_type_default_25: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_23, torch.float32);  sum_default_23 = None
        add_tensor_23: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_tensor_19, mul_tensor_32);  add_tensor_19 = mul_tensor_32 = None
        select_scatter_default_32: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_25, 0, 1);  convert_element_type_default_25 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:923 in forward, code: x = x + skip_weights[i - n] * skip_connections.pop()
        mul_tensor_34: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_41, add_69);  select_41 = None
        add_tensor_24: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_69, mul_tensor_34);  mul_tensor_34 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        mul_tensor_35: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_201, add_tensor_24);  add_201 = add_tensor_24 = None
        sum_default_24: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_35);  mul_tensor_35 = None
        convert_element_type_default_26: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_24, torch.float32);  sum_default_24 = None
        select_scatter_default_33: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_26, 0, 0);  convert_element_type_default_26 = None
        add_tensor_25: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_default_32, select_scatter_default_33);  select_scatter_default_32 = select_scatter_default_33 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_scatter_default_34: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_14, select_scatter_default_31, 0, 6);  select_scatter_default_31 = None
        add_tensor_26: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_tensor_16, select_scatter_default_34);  add_tensor_16 = select_scatter_default_34 = None
        select_scatter_default_35: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_14, add_tensor_25, 0, 6);  add_tensor_25 = None
        add_tensor_27: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_tensor_21, select_scatter_default_35);  add_tensor_21 = select_scatter_default_35 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:923 in forward, code: x = x + skip_weights[i - n] * skip_connections.pop()
        mul_tensor_36: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_407, add_69);  mul_407 = add_69 = None
        sum_default_25: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_36);  mul_tensor_36 = None
        convert_element_type_default_27: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_25, torch.float32);  sum_default_25 = None
        select_scatter_default_36: "f32[6]" = torch.ops.aten.select_scatter.default(full_default_15, convert_element_type_default_27, 0, 0);  full_default_15 = convert_element_type_default_27 = None
        add_tensor_28: "f32[6]" = torch.ops.aten.add.Tensor(add_tensor_22, select_scatter_default_36);  add_tensor_22 = select_scatter_default_36 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:689 in forward, code: v = lambdas[0] * v
        mul_tensor_37: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_262, getitem_64);  permute_262 = getitem_64 = None
        sum_default_26: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_37);  mul_tensor_37 = None
        convert_element_type_default_28: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_26, torch.float32);  sum_default_26 = None
        select_scatter_default_37: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_28, 0, 0);  convert_element_type_default_28 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_int_9: "f32[]" = torch.ops.aten.select.int(select_35, 0, 1);  select_35 = None
        mul_tensor_38: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_215, select_int_9);  select_int_9 = None
        mul_tensor_39: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_215, convert_element_type_default_3)
        sum_default_27: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_39);  mul_tensor_39 = None
        convert_element_type_default_29: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_27, torch.float32);  sum_default_27 = None
        add_tensor_29: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_tensor_23, mul_tensor_38);  add_tensor_23 = mul_tensor_38 = None
        select_scatter_default_38: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_29, 0, 1);  convert_element_type_default_29 = None
        mul_tensor_40: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_215, add_58);  add_215 = add_58 = None
        sum_default_28: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_40);  mul_tensor_40 = None
        convert_element_type_default_30: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_28, torch.float32);  sum_default_28 = None
        select_scatter_default_39: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_30, 0, 0);  convert_element_type_default_30 = None
        add_tensor_30: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_default_38, select_scatter_default_39);  select_scatter_default_38 = select_scatter_default_39 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_scatter_default_40: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_14, select_scatter_default_37, 0, 5);  select_scatter_default_37 = None
        add_tensor_31: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_tensor_26, select_scatter_default_40);  add_tensor_26 = select_scatter_default_40 = None
        select_scatter_default_41: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_14, add_tensor_30, 0, 5);  add_tensor_30 = None
        add_tensor_32: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_tensor_27, select_scatter_default_41);  add_tensor_27 = select_scatter_default_41 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:689 in forward, code: v = lambdas[0] * v
        mul_tensor_41: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_286, getitem_54);  permute_286 = getitem_54 = None
        sum_default_29: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_41);  mul_tensor_41 = None
        convert_element_type_default_31: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_29, torch.float32);  sum_default_29 = None
        select_scatter_default_42: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_31, 0, 0);  convert_element_type_default_31 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_int_10: "f32[]" = torch.ops.aten.select.int(select_29, 0, 1);  select_29 = None
        mul_tensor_42: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_228, select_int_10);  select_int_10 = None
        mul_tensor_43: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_228, convert_element_type_default_3)
        sum_default_30: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_43);  mul_tensor_43 = None
        convert_element_type_default_32: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_30, torch.float32);  sum_default_30 = None
        add_tensor_33: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_tensor_29, mul_tensor_42);  add_tensor_29 = mul_tensor_42 = None
        select_scatter_default_43: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_32, 0, 1);  convert_element_type_default_32 = None
        mul_tensor_44: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_228, add_47);  add_228 = add_47 = None
        sum_default_31: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_44);  mul_tensor_44 = None
        convert_element_type_default_33: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_31, torch.float32);  sum_default_31 = None
        select_scatter_default_44: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_33, 0, 0);  convert_element_type_default_33 = None
        add_tensor_34: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_default_43, select_scatter_default_44);  select_scatter_default_43 = select_scatter_default_44 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_scatter_default_45: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_14, select_scatter_default_42, 0, 4);  select_scatter_default_42 = None
        add_tensor_35: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_tensor_31, select_scatter_default_45);  add_tensor_31 = select_scatter_default_45 = None
        select_scatter_default_46: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_14, add_tensor_34, 0, 4);  add_tensor_34 = None
        add_tensor_36: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_tensor_32, select_scatter_default_46);  add_tensor_32 = select_scatter_default_46 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:689 in forward, code: v = lambdas[0] * v
        mul_tensor_45: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_310, getitem_44);  permute_310 = getitem_44 = None
        sum_default_32: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_45);  mul_tensor_45 = None
        convert_element_type_default_34: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_32, torch.float32);  sum_default_32 = None
        select_scatter_default_47: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_34, 0, 0);  convert_element_type_default_34 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_int_11: "f32[]" = torch.ops.aten.select.int(select_23, 0, 1);  select_23 = None
        mul_tensor_46: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_241, select_int_11);  select_int_11 = None
        mul_tensor_47: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_241, convert_element_type_default_3)
        sum_default_33: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_47);  mul_tensor_47 = None
        convert_element_type_default_35: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_33, torch.float32);  sum_default_33 = None
        add_tensor_37: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_tensor_33, mul_tensor_46);  add_tensor_33 = mul_tensor_46 = None
        select_scatter_default_48: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_35, 0, 1);  convert_element_type_default_35 = None
        mul_tensor_48: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_241, add_36);  add_241 = add_36 = None
        sum_default_34: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_48);  mul_tensor_48 = None
        convert_element_type_default_36: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_34, torch.float32);  sum_default_34 = None
        select_scatter_default_49: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_36, 0, 0);  convert_element_type_default_36 = None
        add_tensor_38: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_default_48, select_scatter_default_49);  select_scatter_default_48 = select_scatter_default_49 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_scatter_default_50: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_14, select_scatter_default_47, 0, 3);  select_scatter_default_47 = None
        add_tensor_39: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_tensor_35, select_scatter_default_50);  add_tensor_35 = select_scatter_default_50 = None
        select_scatter_default_51: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_14, add_tensor_38, 0, 3);  add_tensor_38 = None
        add_tensor_40: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_tensor_36, select_scatter_default_51);  add_tensor_36 = select_scatter_default_51 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:685 in forward, code: v = lambdas[0] * v + lambdas[1] * ve.view_as(
        select_int_12: "f32[]" = torch.ops.aten.select.int(select_17, 0, 1);  select_17 = None
        mul_tensor_49: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_334, select_int_12);  select_int_12 = None
        mul_tensor_50: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_334, reshape_default);  reshape_default = None
        sum_default_35: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_50);  mul_tensor_50 = None
        convert_element_type_default_37: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_35, torch.float32);  sum_default_35 = None
        reshape_default_6: "bf16[6144, 768]" = torch.ops.aten.reshape.default(mul_tensor_49, _shape_param_6);  mul_tensor_49 = _shape_param_6 = None
        add_tensor_41: "bf16[6144, 768]" = torch.ops.aten.add.Tensor(reshape_default_1, reshape_default_6);  reshape_default_1 = reshape_default_6 = None
        select_scatter_default_52: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_37, 0, 1);  convert_element_type_default_37 = None
        mul_tensor_51: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_334, getitem_34);  permute_334 = getitem_34 = None
        sum_default_36: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_51);  mul_tensor_51 = None
        convert_element_type_default_38: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_36, torch.float32);  sum_default_36 = None
        select_scatter_default_53: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_38, 0, 0);  convert_element_type_default_38 = None
        add_tensor_42: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_default_52, select_scatter_default_53);  select_scatter_default_52 = select_scatter_default_53 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_int_13: "f32[]" = torch.ops.aten.select.int(select_16, 0, 1);  select_16 = None
        mul_tensor_52: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_256, select_int_13);  select_int_13 = None
        mul_tensor_53: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_256, convert_element_type_default_3)
        sum_default_37: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_53);  mul_tensor_53 = None
        convert_element_type_default_39: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_37, torch.float32);  sum_default_37 = None
        add_tensor_43: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_tensor_37, mul_tensor_52);  add_tensor_37 = mul_tensor_52 = None
        select_scatter_default_54: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_39, 0, 1);  convert_element_type_default_39 = None
        mul_tensor_54: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_256, add_24);  add_256 = add_24 = None
        sum_default_38: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_54);  mul_tensor_54 = None
        convert_element_type_default_40: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_38, torch.float32);  sum_default_38 = None
        select_scatter_default_55: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_40, 0, 0);  convert_element_type_default_40 = None
        add_tensor_44: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_default_54, select_scatter_default_55);  select_scatter_default_54 = select_scatter_default_55 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_scatter_default_56: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_14, add_tensor_42, 0, 2);  add_tensor_42 = None
        add_tensor_45: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_tensor_39, select_scatter_default_56);  add_tensor_39 = select_scatter_default_56 = None
        select_scatter_default_57: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_14, add_tensor_44, 0, 2);  add_tensor_44 = None
        add_tensor_46: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_tensor_40, select_scatter_default_57);  add_tensor_40 = select_scatter_default_57 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:685 in forward, code: v = lambdas[0] * v + lambdas[1] * ve.view_as(
        select_int_14: "f32[]" = torch.ops.aten.select.int(select_10, 0, 1);  select_10 = None
        mul_tensor_55: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_358, select_int_14);  select_int_14 = None
        mul_tensor_56: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_358, reshape_default_2);  reshape_default_2 = None
        sum_default_39: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_56);  mul_tensor_56 = None
        convert_element_type_default_41: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_39, torch.float32);  sum_default_39 = None
        reshape_default_7: "bf16[6144, 768]" = torch.ops.aten.reshape.default(mul_tensor_55, _shape_param_7);  mul_tensor_55 = _shape_param_7 = None
        add_tensor_47: "bf16[6144, 768]" = torch.ops.aten.add.Tensor(reshape_default_3, reshape_default_7);  reshape_default_3 = reshape_default_7 = None
        select_scatter_default_58: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_41, 0, 1);  convert_element_type_default_41 = None
        mul_tensor_57: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_358, getitem_24);  permute_358 = getitem_24 = None
        sum_default_40: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_57);  mul_tensor_57 = None
        convert_element_type_default_42: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_40, torch.float32);  sum_default_40 = None
        select_scatter_default_59: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_42, 0, 0);  convert_element_type_default_42 = None
        add_tensor_48: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_default_58, select_scatter_default_59);  select_scatter_default_58 = select_scatter_default_59 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_int_15: "f32[]" = torch.ops.aten.select.int(select_9, 0, 1);  select_9 = None
        mul_tensor_58: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_271, select_int_15);  select_int_15 = None
        mul_tensor_59: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_271, convert_element_type_default_3)
        sum_default_41: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_59);  mul_tensor_59 = None
        convert_element_type_default_43: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_41, torch.float32);  sum_default_41 = None
        add_tensor_49: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_tensor_43, mul_tensor_58);  add_tensor_43 = mul_tensor_58 = None
        select_scatter_default_60: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_43, 0, 1);  convert_element_type_default_43 = None
        mul_tensor_60: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_271, add_12);  add_271 = add_12 = None
        sum_default_42: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_60);  mul_tensor_60 = None
        convert_element_type_default_44: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_42, torch.float32);  sum_default_42 = None
        select_scatter_default_61: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_44, 0, 0);  convert_element_type_default_44 = None
        add_tensor_50: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_default_60, select_scatter_default_61);  select_scatter_default_60 = select_scatter_default_61 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_scatter_default_62: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_14, add_tensor_48, 0, 1);  add_tensor_48 = None
        add_tensor_51: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_tensor_45, select_scatter_default_62);  add_tensor_45 = select_scatter_default_62 = None
        select_scatter_default_63: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_14, add_tensor_50, 0, 1);  add_tensor_50 = None
        add_tensor_52: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_tensor_46, select_scatter_default_63);  add_tensor_46 = select_scatter_default_63 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        reshape_default_8: "bf16[1, 6144, 12]" = torch.ops.aten.reshape.default(mm_169, _shape_param_8);  mm_169 = _shape_param_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        slice_scatter_default: "bf16[1, 6144, 768]" = torch.ops.aten.slice_scatter.default(full_default_19, reshape_default_8, 2, 0, 12);  full_default_19 = reshape_default_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:685 in forward, code: v = lambdas[0] * v + lambdas[1] * ve.view_as(
        select_int_16: "f32[]" = torch.ops.aten.select.int(select_3, 0, 1);  select_3 = None
        mul_tensor_61: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_382, select_int_16);  select_int_16 = None
        mul_tensor_62: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_382, reshape_default_4);  reshape_default_4 = None
        sum_default_43: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_62);  mul_tensor_62 = None
        convert_element_type_default_45: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_43, torch.float32);  sum_default_43 = None
        reshape_default_9: "bf16[6144, 768]" = torch.ops.aten.reshape.default(mul_tensor_61, _shape_param_9);  mul_tensor_61 = _shape_param_9 = None
        add_tensor_53: "bf16[6144, 768]" = torch.ops.aten.add.Tensor(reshape_default_5, reshape_default_9);  reshape_default_5 = reshape_default_9 = None
        select_scatter_default_64: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_45, 0, 1);  convert_element_type_default_45 = None
        mul_tensor_63: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(permute_382, getitem_14);  permute_382 = getitem_14 = None
        sum_default_44: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_63);  mul_tensor_63 = None
        convert_element_type_default_46: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_44, torch.float32);  sum_default_44 = None
        select_scatter_default_65: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_46, 0, 0);  convert_element_type_default_46 = None
        add_tensor_54: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_default_64, select_scatter_default_65);  select_scatter_default_64 = select_scatter_default_65 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        reshape_default_10: "bf16[1, 6144, 768]" = torch.ops.aten.reshape.default(mm_171, _shape_param_10);  mm_171 = _shape_param_10 = None
        add_tensor_55: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(slice_scatter_default, reshape_default_10);  slice_scatter_default = reshape_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_default_47: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_55, torch.float32);  add_tensor_55 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_int_17: "f32[2]" = torch.ops.aten.select.int(view_6, 0, 0);  view_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        select_int_18: "f32[]" = torch.ops.aten.select.int(select_int_17, 0, 0)
        mul_tensor_64: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_int_18, convert_element_type_default_3)
        select_int_19: "f32[]" = torch.ops.aten.select.int(select_int_17, 0, 1);  select_int_17 = None
        mul_tensor_65: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(select_int_19, convert_element_type_default_3)
        add_tensor_56: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(mul_tensor_64, mul_tensor_65);  mul_tensor_64 = mul_tensor_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_default_48: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_56, torch.float32);  add_tensor_56 = None
        mul_tensor_66: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default_48, rsqrt_1);  convert_element_type_default_48 = None
        mul_tensor_67: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_66, convert_element_type_default_47)
        sum_dim_int_list: "f32[1, 6144, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_67, [2], True);  mul_tensor_67 = None
        div_tensor: "f32[1, 6144, 768]" = torch.ops.aten.div.Tensor(mul_tensor_66, 768);  mul_tensor_66 = None
        mul_tensor_68: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sum_dim_int_list);  div_tensor = sum_dim_int_list = None
        sub_tensor: "f32[1, 6144, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_default_47, mul_tensor_68);  convert_element_type_default_47 = mul_tensor_68 = None
        mul_tensor_69: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_1);  sub_tensor = rsqrt_1 = None
        convert_element_type_default_49: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_tensor_69, torch.bfloat16);  mul_tensor_69 = None
        add_tensor_57: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_277, convert_element_type_default_49);  add_277 = convert_element_type_default_49 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:749 in forward, code: x = lambdas[0] * x + lambdas[1] * x0
        mul_tensor_70: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_tensor_57, select_int_19);  select_int_19 = None
        mul_tensor_71: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_tensor_57, convert_element_type_default_3);  convert_element_type_default_3 = None
        sum_default_45: "bf16[]" = torch.ops.aten.sum.default(mul_tensor_71);  mul_tensor_71 = None
        convert_element_type_default_50: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_45, torch.float32);  sum_default_45 = None
        add_tensor_58: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_tensor_49, mul_tensor_70);  add_tensor_49 = mul_tensor_70 = None
        select_scatter_default_66: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_50, 0, 1)
        mul_tensor_72: "bf16[1, 6144, 768]" = torch.ops.aten.mul.Tensor(add_tensor_57, select_int_18);  add_tensor_57 = select_int_18 = None
        add_tensor_59: "bf16[1, 6144, 768]" = torch.ops.aten.add.Tensor(add_tensor_58, mul_tensor_72);  add_tensor_58 = mul_tensor_72 = None
        select_scatter_default_67: "f32[2]" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_50, 0, 0);  full_default = convert_element_type_default_50 = None
        add_tensor_60: "f32[2]" = torch.ops.aten.add.Tensor(select_scatter_default_66, select_scatter_default_67);  select_scatter_default_66 = select_scatter_default_67 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_scatter_default_68: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_14, add_tensor_54, 0, 0);  add_tensor_54 = None
        add_tensor_61: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_tensor_51, select_scatter_default_68);  add_tensor_51 = select_scatter_default_68 = None
        select_scatter_default_69: "f32[12, 2]" = torch.ops.aten.select_scatter.default(full_default_14, add_tensor_60, 0, 0);  full_default_14 = add_tensor_60 = None
        add_tensor_62: "f32[12, 2]" = torch.ops.aten.add.Tensor(add_tensor_52, select_scatter_default_69);  add_tensor_52 = select_scatter_default_69 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:915 in forward, code: sa_lambdas = self.scalars[3 * len(self.blocks) : 5 * len(self.blocks)].view(
        reshape_default_11: "f32[24]" = torch.ops.aten.reshape.default(add_tensor_61, _shape_param_11);  add_tensor_61 = _shape_param_11 = None
        full_default_16: "f32[64]" = torch.ops.aten.full.default([64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default_1: "f32[64]" = torch.ops.aten.slice_scatter.default(full_default_16, reshape_default_11, 0, 36, 60);  reshape_default_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:914 in forward, code: lambdas = self.scalars[1 * len(self.blocks) : 3 * len(self.blocks)].view(-1, 2)
        reshape_default_12: "f32[24]" = torch.ops.aten.reshape.default(add_tensor_62, _shape_param_12);  add_tensor_62 = _shape_param_12 = None
        slice_scatter_default_2: "f32[64]" = torch.ops.aten.slice_scatter.default(full_default_16, reshape_default_12, 0, 12, 36);  reshape_default_12 = None
        add_tensor_63: "f32[64]" = torch.ops.aten.add.Tensor(slice_scatter_default_1, slice_scatter_default_2);  slice_scatter_default_1 = slice_scatter_default_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:913 in forward, code: skip_weights = self.scalars[: (len(self.blocks) // 2)]
        slice_scatter_default_3: "f32[64]" = torch.ops.aten.slice_scatter.default(full_default_16, add_tensor_28, 0, 0, 6);  full_default_16 = add_tensor_28 = None
        add_tensor_64: "f32[64]" = torch.ops.aten.add.Tensor(add_tensor_63, slice_scatter_default_3);  add_tensor_63 = slice_scatter_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_default_51: "f32[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_59, torch.float32);  add_tensor_59 = None
        mul_tensor_73: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_4, convert_element_type_default_51)
        sum_dim_int_list_1: "f32[1, 6144, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_73, [2], True);  mul_tensor_73 = None
        div_tensor_1: "f32[1, 6144, 768]" = torch.ops.aten.div.Tensor(mul_tensor_4, 768);  mul_tensor_4 = None
        mul_tensor_74: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(div_tensor_1, sum_dim_int_list_1);  div_tensor_1 = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[1, 6144, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_default_51, mul_tensor_74);  convert_element_type_default_51 = mul_tensor_74 = None
        mul_tensor_75: "f32[1, 6144, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt);  sub_tensor_1 = rsqrt = None
        convert_element_type_default_52: "bf16[1, 6144, 768]" = torch.ops.prims.convert_element_type.default(mul_tensor_75, torch.bfloat16);  mul_tensor_75 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:909 in forward, code: x = x0 = norm(self.embed(input_seq)[None])  # use of norm here by @Grad62304977
        squeeze_dim: "bf16[6144, 768]" = torch.ops.aten.squeeze.dim(convert_element_type_default_52, 0);  convert_element_type_default_52 = None
        convert_element_type_default_53: "f32[6144, 768]" = torch.ops.prims.convert_element_type.default(squeeze_dim, torch.float32);  squeeze_dim = None
        convert_element_type_default_54: "i64[6144]" = torch.ops.prims.convert_element_type.default(primals_1, torch.int64);  primals_1 = None
        eq_scalar: "b8[6144]" = torch.ops.aten.eq.Scalar(convert_element_type_default_54, -1)
        unsqueeze_default_1: "b8[6144, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[6144, 768]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default_13, convert_element_type_default_53);  convert_element_type_default_53 = None
        full_default_17: "f32[50304, 768]" = torch.ops.aten.full.default([50304, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[50304, 768]" = torch.ops.aten.index_put.default(full_default_17, [convert_element_type_default_54], where_self, True);  where_self = None
        convert_element_type_default_55: "bf16[50304, 768]" = torch.ops.prims.convert_element_type.default(index_put_default, torch.bfloat16);  index_put_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:883 in forward, code: ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        convert_element_type_default_56: "f32[6144, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_41, torch.float32);  add_tensor_41 = None
        where_self_1: "f32[6144, 768]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default_13, convert_element_type_default_56);  convert_element_type_default_56 = None
        index_put_default_1: "f32[50304, 768]" = torch.ops.aten.index_put.default(full_default_17, [convert_element_type_default_54], where_self_1, True);  where_self_1 = None
        convert_element_type_default_57: "bf16[50304, 768]" = torch.ops.prims.convert_element_type.default(index_put_default_1, torch.bfloat16);  index_put_default_1 = None
        convert_element_type_default_58: "f32[6144, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_47, torch.float32);  add_tensor_47 = None
        where_self_2: "f32[6144, 768]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default_13, convert_element_type_default_58);  convert_element_type_default_58 = None
        index_put_default_2: "f32[50304, 768]" = torch.ops.aten.index_put.default(full_default_17, [convert_element_type_default_54], where_self_2, True);  where_self_2 = None
        convert_element_type_default_59: "bf16[50304, 768]" = torch.ops.prims.convert_element_type.default(index_put_default_2, torch.bfloat16);  index_put_default_2 = None
        convert_element_type_default_60: "f32[6144, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_53, torch.float32);  add_tensor_53 = None
        where_self_3: "f32[6144, 768]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default_13, convert_element_type_default_60);  unsqueeze_default_1 = full_default_13 = convert_element_type_default_60 = None
        index_put_default_3: "f32[50304, 768]" = torch.ops.aten.index_put.default(full_default_17, [convert_element_type_default_54], where_self_3, True);  full_default_17 = convert_element_type_default_54 = where_self_3 = None
        convert_element_type_default_61: "bf16[50304, 768]" = torch.ops.prims.convert_element_type.default(index_put_default_3, torch.bfloat16);  index_put_default_3 = None
        return (convert_element_type_default_57, convert_element_type_default_61, convert_element_type_default_59, convert_element_type_default_55, add_tensor_64)



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
