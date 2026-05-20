"""
Standalone repro captured via capture_hook.
Label: hf_DebertaV2ForMaskedLM_infer
Pattern hash: 7026b642e8fd
Shape hash: 4cacdb7a
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([], f32), T([], f32), T([], f32), T([], f32), T([], f32), T([], f32), T([], f32), T([], f32), T([], f32), T([], f32), T([], f32), T([], f32), T([], f32), T([], f32), T([], f32), T([], f32), T([], f32), T([], f32), T([], f32), T([], f32), T([], f32), T([], f32), T([], f32), T([], f32), T([8, 512], i64, max=512), T([4096, 128100], f32), S([8, 512, 128100]), S([-1, 128100]))"

class Repro(torch.nn.Module):
    def forward(self, _tensor_constant0: "f32[]", _tensor_constant1: "f32[]", _tensor_constant2: "f32[]", _tensor_constant3: "f32[]", _tensor_constant4: "f32[]", _tensor_constant5: "f32[]", _tensor_constant6: "f32[]", _tensor_constant7: "f32[]", _tensor_constant8: "f32[]", _tensor_constant9: "f32[]", _tensor_constant10: "f32[]", _tensor_constant11: "f32[]", _tensor_constant12: "f32[]", _tensor_constant13: "f32[]", _tensor_constant14: "f32[]", _tensor_constant15: "f32[]", _tensor_constant16: "f32[]", _tensor_constant17: "f32[]", _tensor_constant18: "f32[]", _tensor_constant19: "f32[]", _tensor_constant20: "f32[]", _tensor_constant21: "f32[]", _tensor_constant22: "f32[]", _tensor_constant23: "f32[]", arg395_1: "i64[8, 512]", addmm_145: "f32[4096, 128100]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:558 in forward, code: mask = mask.unsqueeze(2)
        full_default: "f32[8, 512, 1]" = torch.ops.aten.full.default([8, 512, 1], 1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        lift_fresh_copy_default: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        mul_tensor: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_default, 1);  lift_fresh_copy_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:753 in forward, code: attention_mask = torch.ones(input_shape, device=device)
        full_default_1: "f32[8, 512]" = torch.ops.aten.full.default([8, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:605 in get_attention_mask, code: extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        unsqueeze_default: "f32[8, 1, 512]" = torch.ops.aten.unsqueeze.default(full_default_1, 1);  full_default_1 = None
        unsqueeze_default_1: "f32[8, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:606 in get_attention_mask, code: attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
        squeeze_dim: "f32[8, 1, 512]" = torch.ops.aten.squeeze.dim(unsqueeze_default_1, -2)
        unsqueeze_default_2: "f32[8, 1, 512, 1]" = torch.ops.aten.unsqueeze.default(squeeze_dim, -1);  squeeze_dim = None
        mul_tensor_1: "f32[8, 1, 512, 512]" = torch.ops.aten.mul.Tensor(unsqueeze_default_1, unsqueeze_default_2);  unsqueeze_default_1 = unsqueeze_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_default: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bool)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        lift_fresh_copy_default_1: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
        mul_tensor_2: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_default_1, 1);  lift_fresh_copy_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_default_1: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bool)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        lift_fresh_copy_default_2: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant2);  _tensor_constant2 = None
        mul_tensor_3: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_default_2, 1);  lift_fresh_copy_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_default_2: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bool)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        lift_fresh_copy_default_3: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant3);  _tensor_constant3 = None
        mul_tensor_4: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_default_3, 1);  lift_fresh_copy_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_default_3: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bool)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        lift_fresh_copy_default_4: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant4);  _tensor_constant4 = None
        mul_tensor_5: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_default_4, 1);  lift_fresh_copy_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_default_4: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bool)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        lift_fresh_copy_default_5: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant5);  _tensor_constant5 = None
        mul_tensor_6: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_default_5, 1);  lift_fresh_copy_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_default_5: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bool)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        lift_fresh_copy_default_6: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant6);  _tensor_constant6 = None
        mul_tensor_7: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_default_6, 1);  lift_fresh_copy_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_default_6: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bool)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        lift_fresh_copy_default_7: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant7);  _tensor_constant7 = None
        mul_tensor_8: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_default_7, 1);  lift_fresh_copy_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_default_7: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bool)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        lift_fresh_copy_default_8: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant8);  _tensor_constant8 = None
        mul_tensor_9: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_default_8, 1);  lift_fresh_copy_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_default_8: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bool)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        lift_fresh_copy_default_9: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant9);  _tensor_constant9 = None
        mul_tensor_10: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_default_9, 1);  lift_fresh_copy_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_default_9: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bool)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        lift_fresh_copy_default_10: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant10);  _tensor_constant10 = None
        mul_tensor_11: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_default_10, 1);  lift_fresh_copy_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_default_10: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bool)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        lift_fresh_copy_default_11: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant11);  _tensor_constant11 = None
        mul_tensor_12: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_default_11, 1);  lift_fresh_copy_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_default_11: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bool)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        lift_fresh_copy_default_12: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant12);  _tensor_constant12 = None
        mul_tensor_13: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_default_12, 1);  lift_fresh_copy_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_default_12: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bool)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        lift_fresh_copy_default_13: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant13);  _tensor_constant13 = None
        mul_tensor_14: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_default_13, 1);  lift_fresh_copy_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_default_13: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bool)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        lift_fresh_copy_default_14: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant14);  _tensor_constant14 = None
        mul_tensor_15: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_default_14, 1);  lift_fresh_copy_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_default_14: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bool)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        lift_fresh_copy_default_15: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant15);  _tensor_constant15 = None
        mul_tensor_16: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_default_15, 1);  lift_fresh_copy_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_default_15: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bool)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        lift_fresh_copy_default_16: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant16);  _tensor_constant16 = None
        mul_tensor_17: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_default_16, 1);  lift_fresh_copy_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_default_16: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bool)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        lift_fresh_copy_default_17: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant17);  _tensor_constant17 = None
        mul_tensor_18: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_default_17, 1);  lift_fresh_copy_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_default_17: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bool)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        lift_fresh_copy_default_18: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant18);  _tensor_constant18 = None
        mul_tensor_19: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_default_18, 1);  lift_fresh_copy_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_default_18: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bool)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        lift_fresh_copy_default_19: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant19);  _tensor_constant19 = None
        mul_tensor_20: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_default_19, 1);  lift_fresh_copy_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_default_19: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bool)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        lift_fresh_copy_default_20: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant20);  _tensor_constant20 = None
        mul_tensor_21: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_default_20, 1);  lift_fresh_copy_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_default_20: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bool)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        lift_fresh_copy_default_21: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant21);  _tensor_constant21 = None
        mul_tensor_22: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_default_21, 1);  lift_fresh_copy_default_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_default_21: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bool)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        lift_fresh_copy_default_22: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant22);  _tensor_constant22 = None
        mul_tensor_23: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_default_22, 1);  lift_fresh_copy_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_default_22: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bool)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        lift_fresh_copy_default_23: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant23);  _tensor_constant23 = None
        mul_tensor_24: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_default_23, 1);  lift_fresh_copy_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_mask = attention_mask.bool()
        convert_element_type_default_23: "b8[8, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bool);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:968 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        reshape_default: "i64[4096]" = torch.ops.aten.reshape.default(arg395_1, [-1]);  arg395_1 = None
        ne_scalar: "b8[4096]" = torch.ops.aten.ne.Scalar(reshape_default, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:838 in forward, code: hidden_states = self.decoder(hidden_states)
        reshape_default_1: "f32[8, 512, 128100]" = torch.ops.aten.reshape.default(addmm_145, _shape_param_0);  addmm_145 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:968 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        reshape_default_2: "f32[4096, 128100]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_1);  reshape_default_1 = _shape_param_1 = None
        amax_default: "f32[4096, 1]" = torch.ops.aten.amax.default(reshape_default_2, [1], True)
        sub_tensor: "f32[4096, 128100]" = torch.ops.aten.sub.Tensor(reshape_default_2, amax_default);  reshape_default_2 = amax_default = None
        exp_default: "f32[4096, 128100]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[4096, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[4096, 128100]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        ne_scalar_1: "b8[4096]" = torch.ops.aten.ne.Scalar(reshape_default, -100)
        full_default_2: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[4096]" = torch.ops.aten.where.self(ne_scalar_1, reshape_default, full_default_2);  ne_scalar_1 = full_default_2 = None
        unsqueeze_default_3: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[4096, 1]" = torch.ops.aten.gather.default(sub_tensor_1, 1, unsqueeze_default_3);  sub_tensor_1 = unsqueeze_default_3 = None
        squeeze_dim_1: "f32[4096]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[4096]" = torch.ops.aten.neg.default(squeeze_dim_1);  squeeze_dim_1 = None
        full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[4096]" = torch.ops.aten.where.self(ne_scalar, neg_default, full_default_3);  ne_scalar = neg_default = full_default_3 = None
        sum_default: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        ne_scalar_2: "b8[4096]" = torch.ops.aten.ne.Scalar(reshape_default, -100);  reshape_default = None
        sum_default_1: "i64[]" = torch.ops.aten.sum.default(ne_scalar_2);  ne_scalar_2 = None
        convert_element_type_default_24: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_1, torch.float32);  sum_default_1 = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default, convert_element_type_default_24);  sum_default = convert_element_type_default_24 = None
        return (full_default, mul_tensor, convert_element_type_default, mul_tensor_2, convert_element_type_default_1, mul_tensor_3, convert_element_type_default_2, mul_tensor_4, convert_element_type_default_3, mul_tensor_5, convert_element_type_default_4, mul_tensor_6, convert_element_type_default_5, mul_tensor_7, convert_element_type_default_6, mul_tensor_8, convert_element_type_default_7, mul_tensor_9, convert_element_type_default_8, mul_tensor_10, convert_element_type_default_9, mul_tensor_11, convert_element_type_default_10, mul_tensor_12, convert_element_type_default_11, mul_tensor_13, convert_element_type_default_12, mul_tensor_14, convert_element_type_default_13, mul_tensor_15, convert_element_type_default_14, mul_tensor_16, convert_element_type_default_15, mul_tensor_17, convert_element_type_default_16, mul_tensor_18, convert_element_type_default_17, mul_tensor_19, convert_element_type_default_18, mul_tensor_20, convert_element_type_default_19, mul_tensor_21, convert_element_type_default_20, mul_tensor_22, convert_element_type_default_21, mul_tensor_23, convert_element_type_default_22, mul_tensor_24, convert_element_type_default_23, div_tensor)


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
