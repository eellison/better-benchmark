"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_T5_train
Pattern hash: 2444a8eb389c
Shape hash: d512cf78
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
_shapes_config = "(T([8, 8, 1024, 1024], f32), T([8, 8, 1024, 1024], f32), T([8, 8, 1024, 1024], f32), T([8, 8, 1024, 1024], f32), T([8, 8, 1024, 1024], f32), T([8, 8, 1024, 1024], f32), T([1024, 1024], i64, gen=Index(32)), T([], f32), T([8, 8, 1024, 1024], f32), T([8, 8, 1024, 1024], f32), T([8, 8, 1024, 1024], f32), T([8, 8, 1024, 1024], f32), T([8, 8, 1024, 1024], f32), T([8, 8, 1024, 1024], f32), T([1024, 1024], i64, gen=Index(32)))"

class Repro(torch.nn.Module):
    def forward(self, view_473: "f32[8, 8, 1024, 1024]", view_524: "f32[8, 8, 1024, 1024]", view_575: "f32[8, 8, 1024, 1024]", view_626: "f32[8, 8, 1024, 1024]", view_677: "f32[8, 8, 1024, 1024]", view_728: "f32[8, 8, 1024, 1024]", add_45: "i64[1024, 1024]", full_default: "f32[]", view_757: "f32[8, 8, 1024, 1024]", view_785: "f32[8, 8, 1024, 1024]", view_813: "f32[8, 8, 1024, 1024]", view_841: "f32[8, 8, 1024, 1024]", view_869: "f32[8, 8, 1024, 1024]", view_897: "f32[8, 8, 1024, 1024]", add_6: "i64[1024, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_473, view_524);  view_473 = view_524 = None
        add_tensor_1: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor, view_575);  add_tensor = view_575 = None
        add_tensor_2: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_1, view_626);  add_tensor_1 = view_626 = None
        add_tensor_3: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_2, view_677);  add_tensor_2 = view_677 = None
        add_tensor_4: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_3, view_728);  add_tensor_3 = view_728 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        sum_dim_int_list: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sum.dim_IntList(add_tensor_4, [0], True);  add_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        squeeze_dim: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sum_dim_int_list, 0);  sum_dim_int_list = None
        permute_default: "f32[1024, 1024, 8]" = torch.ops.aten.permute.default(squeeze_dim, [1, 2, 0]);  squeeze_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:236 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        eq_scalar: "b8[1024, 1024]" = torch.ops.aten.eq.Scalar(add_45, -1)
        unsqueeze_default: "b8[1024, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[1024, 1024, 8]" = torch.ops.aten.where.self(unsqueeze_default, full_default, permute_default);  unsqueeze_default = permute_default = None
        clone_default: "f32[1024, 1024, 8]" = torch.ops.aten.clone.default(where_self, memory_format = torch.contiguous_format);  where_self = None
        full_default_1: "f32[32, 8]" = torch.ops.aten.full.default([32, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[32, 8]" = torch.ops.aten.index_put.default(full_default_1, [add_45], clone_default, True);  add_45 = clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_5: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_757, view_785);  view_757 = view_785 = None
        add_tensor_6: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_5, view_813);  add_tensor_5 = view_813 = None
        add_tensor_7: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_6, view_841);  add_tensor_6 = view_841 = None
        add_tensor_8: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_7, view_869);  add_tensor_7 = view_869 = None
        add_tensor_9: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_8, view_897);  add_tensor_8 = view_897 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        sum_dim_int_list_1: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sum.dim_IntList(add_tensor_9, [0], True);  add_tensor_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        squeeze_dim_1: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_1, 0);  sum_dim_int_list_1 = None
        permute_default_1: "f32[1024, 1024, 8]" = torch.ops.aten.permute.default(squeeze_dim_1, [1, 2, 0]);  squeeze_dim_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:236 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        eq_scalar_1: "b8[1024, 1024]" = torch.ops.aten.eq.Scalar(add_6, -1)
        unsqueeze_default_1: "b8[1024, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[1024, 1024, 8]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default, permute_default_1);  unsqueeze_default_1 = full_default = permute_default_1 = None
        clone_default_1: "f32[1024, 1024, 8]" = torch.ops.aten.clone.default(where_self_1, memory_format = torch.contiguous_format);  where_self_1 = None
        index_put_default_1: "f32[32, 8]" = torch.ops.aten.index_put.default(full_default_1, [add_6], clone_default_1, True);  full_default_1 = add_6 = clone_default_1 = None
        return (index_put_default, index_put_default_1)



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
