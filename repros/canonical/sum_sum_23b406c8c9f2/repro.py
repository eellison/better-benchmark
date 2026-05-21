"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_T5_base_train
Pattern hash: 23b406c8c9f2
Shape hash: 17712799
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
_shapes_config = "(T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([1024, 1024], i64, gen=Index(32)), T([], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([8, 12, 1024, 1024], f32), T([1024, 1024], i64, gen=Index(32)))"

class Repro(torch.nn.Module):
    def forward(self, view_899: "f32[8, 12, 1024, 1024]", view_950: "f32[8, 12, 1024, 1024]", view_1001: "f32[8, 12, 1024, 1024]", view_1052: "f32[8, 12, 1024, 1024]", view_1103: "f32[8, 12, 1024, 1024]", view_1154: "f32[8, 12, 1024, 1024]", view_1205: "f32[8, 12, 1024, 1024]", view_1256: "f32[8, 12, 1024, 1024]", view_1307: "f32[8, 12, 1024, 1024]", view_1358: "f32[8, 12, 1024, 1024]", view_1409: "f32[8, 12, 1024, 1024]", view_1460: "f32[8, 12, 1024, 1024]", add_75: "i64[1024, 1024]", full_default: "f32[]", view_1489: "f32[8, 12, 1024, 1024]", view_1517: "f32[8, 12, 1024, 1024]", view_1545: "f32[8, 12, 1024, 1024]", view_1573: "f32[8, 12, 1024, 1024]", view_1601: "f32[8, 12, 1024, 1024]", view_1629: "f32[8, 12, 1024, 1024]", view_1657: "f32[8, 12, 1024, 1024]", view_1685: "f32[8, 12, 1024, 1024]", view_1713: "f32[8, 12, 1024, 1024]", view_1741: "f32[8, 12, 1024, 1024]", view_1769: "f32[8, 12, 1024, 1024]", view_1797: "f32[8, 12, 1024, 1024]", add_6: "i64[1024, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_899, view_950);  view_899 = view_950 = None
        add_tensor_1: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor, view_1001);  add_tensor = view_1001 = None
        add_tensor_2: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_1, view_1052);  add_tensor_1 = view_1052 = None
        add_tensor_3: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_2, view_1103);  add_tensor_2 = view_1103 = None
        add_tensor_4: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_3, view_1154);  add_tensor_3 = view_1154 = None
        add_tensor_5: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_4, view_1205);  add_tensor_4 = view_1205 = None
        add_tensor_6: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_5, view_1256);  add_tensor_5 = view_1256 = None
        add_tensor_7: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_6, view_1307);  add_tensor_6 = view_1307 = None
        add_tensor_8: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_7, view_1358);  add_tensor_7 = view_1358 = None
        add_tensor_9: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_8, view_1409);  add_tensor_8 = view_1409 = None
        add_tensor_10: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_9, view_1460);  add_tensor_9 = view_1460 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        sum_dim_int_list: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sum.dim_IntList(add_tensor_10, [0], True);  add_tensor_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        squeeze_dim: "f32[12, 1024, 1024]" = torch.ops.aten.squeeze.dim(sum_dim_int_list, 0);  sum_dim_int_list = None
        permute_default: "f32[1024, 1024, 12]" = torch.ops.aten.permute.default(squeeze_dim, [1, 2, 0]);  squeeze_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:236 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        eq_scalar: "b8[1024, 1024]" = torch.ops.aten.eq.Scalar(add_75, -1)
        unsqueeze_default: "b8[1024, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[1024, 1024, 12]" = torch.ops.aten.where.self(unsqueeze_default, full_default, permute_default);  unsqueeze_default = permute_default = None
        clone_default: "f32[1024, 1024, 12]" = torch.ops.aten.clone.default(where_self, memory_format = torch.contiguous_format);  where_self = None
        full_default_1: "f32[32, 12]" = torch.ops.aten.full.default([32, 12], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[32, 12]" = torch.ops.aten.index_put.default(full_default_1, [add_75], clone_default, True);  add_75 = clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_11: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_1489, view_1517);  view_1489 = view_1517 = None
        add_tensor_12: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_11, view_1545);  add_tensor_11 = view_1545 = None
        add_tensor_13: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_12, view_1573);  add_tensor_12 = view_1573 = None
        add_tensor_14: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_13, view_1601);  add_tensor_13 = view_1601 = None
        add_tensor_15: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_14, view_1629);  add_tensor_14 = view_1629 = None
        add_tensor_16: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_15, view_1657);  add_tensor_15 = view_1657 = None
        add_tensor_17: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_16, view_1685);  add_tensor_16 = view_1685 = None
        add_tensor_18: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_17, view_1713);  add_tensor_17 = view_1713 = None
        add_tensor_19: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_18, view_1741);  add_tensor_18 = view_1741 = None
        add_tensor_20: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_19, view_1769);  add_tensor_19 = view_1769 = None
        add_tensor_21: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_20, view_1797);  add_tensor_20 = view_1797 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        sum_dim_int_list_1: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sum.dim_IntList(add_tensor_21, [0], True);  add_tensor_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        squeeze_dim_1: "f32[12, 1024, 1024]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_1, 0);  sum_dim_int_list_1 = None
        permute_default_1: "f32[1024, 1024, 12]" = torch.ops.aten.permute.default(squeeze_dim_1, [1, 2, 0]);  squeeze_dim_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:236 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        eq_scalar_1: "b8[1024, 1024]" = torch.ops.aten.eq.Scalar(add_6, -1)
        unsqueeze_default_1: "b8[1024, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[1024, 1024, 12]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default, permute_default_1);  unsqueeze_default_1 = full_default = permute_default_1 = None
        clone_default_1: "f32[1024, 1024, 12]" = torch.ops.aten.clone.default(where_self_1, memory_format = torch.contiguous_format);  where_self_1 = None
        index_put_default_1: "f32[32, 12]" = torch.ops.aten.index_put.default(full_default_1, [add_6], clone_default_1, True);  full_default_1 = add_6 = clone_default_1 = None
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
