"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train
Pattern hash: cd47d7ca4703
Shape hash: 6e66b7cd
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([8, 1024, 768], b8), T([8, 1024, 768], f32), T([768], f32), T([8, 1024, 768], f32), T([8, 1024, 1], f32), T([8, 1024], i64, max=1), T([8, 1024], i64, max=4098), T([8, 1024], i64, max=50265))"

class Repro(torch.nn.Module):
    def forward(self, gt: "b8[8, 1024, 768]", tangents_1: "f32[8, 1024, 768]", primals_5: "f32[768]", mul_2: "f32[8, 1024, 768]", div: "f32[8, 1024, 1]", full_default: "i64[8, 1024]", add: "i64[8, 1024]", primals_1: "i64[8, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:424 in forward, code: embeddings = self.dropout(embeddings)
        convert_element_type_default: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.0);  convert_element_type_default = None
        mul_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(tangents_1, mul_tensor);  tangents_1 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:423 in forward, code: embeddings = self.LayerNorm(embeddings)
        mul_tensor_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, primals_5);  primals_5 = None
        mul_tensor_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 768)
        sum_dim_int_list: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True)
        mul_tensor_4: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_2);  mul_tensor_2 = None
        sum_dim_int_list_1: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [2], True);  mul_tensor_4 = None
        mul_tensor_5: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_3, sum_dim_int_list);  mul_tensor_3 = sum_dim_int_list = None
        sub_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_5);  sub_tensor = mul_tensor_5 = None
        mul_tensor_6: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div, sub_tensor_1);  div = sub_tensor_1 = None
        mul_tensor_7: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, mul_2);  mul_2 = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:420 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        full_default_1: "b8[8, 1024, 1]" = torch.ops.aten.full.default([8, 1024, 1], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[8, 1024, 768]" = torch.ops.aten.where.self(full_default_1, full_default_2, mul_tensor_6);  full_default_1 = None
        full_default_3: "f32[1, 768]" = torch.ops.aten.full.default([1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[1, 768]" = torch.ops.aten.index_put.default(full_default_3, [full_default], where_self, True);  full_default_3 = full_default = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:419 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        eq_scalar: "b8[8, 1024]" = torch.ops.aten.eq.Scalar(add, 1)
        unsqueeze_default: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self_1: "f32[8, 1024, 768]" = torch.ops.aten.where.self(unsqueeze_default, full_default_2, mul_tensor_6);  unsqueeze_default = None
        full_default_4: "f32[4098, 768]" = torch.ops.aten.full.default([4098, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[4098, 768]" = torch.ops.aten.index_put.default(full_default_4, [add], where_self_1, True);  full_default_4 = add = where_self_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:418 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        eq_scalar_1: "b8[8, 1024]" = torch.ops.aten.eq.Scalar(primals_1, 1)
        unsqueeze_default_1: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_2: "f32[8, 1024, 768]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default_2, mul_tensor_6);  unsqueeze_default_1 = full_default_2 = mul_tensor_6 = None
        full_default_5: "f32[50265, 768]" = torch.ops.aten.full.default([50265, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_2: "f32[50265, 768]" = torch.ops.aten.index_put.default(full_default_5, [primals_1], where_self_2, True);  full_default_5 = primals_1 = where_self_2 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, index_put_default, index_put_default_1, index_put_default_2)


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
