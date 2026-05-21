"""
Standalone repro captured via capture_hook.
Label: torchbench_modded_nanogpt_train
Pattern hash: 4658780a79e9
Shape hash: 6a0cabf6
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
_shapes_config = "(T([], i32), T([1, 1, 48], i32), T([1, 1, 48], i32), T([48], i32), T([1, 1, 48, 48], i32, gen=Index(1)), T([], i32, gen=Index(1)), T([1, 1, 48, 49], i32, gen=Index(1)), T([1, 1, 1, 1], i64, gen=Index(1)), T([1, 1, 1], i64, gen=Index(1)), T([48, 1], i32, gen=Index(1)), T([1, 1, 1, 1], i32, gen=Index(1)), T([1, 1, 48, 48], i32, gen=Index(1)))"

class Repro(torch.nn.Module):
    def forward(self, primals_5: "i32[]", unsqueeze_11: "i32[1, 1, 48]", unsqueeze_7: "i32[1, 1, 48]", iota_1: "i32[48]", unsqueeze_9: "i32[1, 1, 48, 48]", full_default_1: "i32[]", full_default: "i32[1, 1, 48, 49]", unsqueeze_20: "i64[1, 1, 1, 1]", unsqueeze_17: "i64[1, 1, 1]", unsqueeze_14: "i32[48, 1]", full_default_2: "i32[1, 1, 1, 1]", unsqueeze_13: "i32[1, 1, 48, 48]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:875 in create_blockmasks, code: sliding_window_num_blocks // 2
        div_tensor_mode: "i32[]" = torch.ops.aten.div.Tensor_mode(primals_5, 2, rounding_mode = 'floor');  primals_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:864 in build_bm, code: torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1),
        sub_tensor: "i32[1, 1, 48]" = torch.ops.aten.sub.Tensor(div_tensor_mode, unsqueeze_11)
        clamp_min_default: "i32[1, 1, 48]" = torch.ops.aten.clamp_min.default(sub_tensor, 1);  sub_tensor = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:862 in build_bm, code: torch.clamp_max(
        clamp_max_tensor: "i32[1, 1, 48]" = torch.ops.aten.clamp_max.Tensor(unsqueeze_7, clamp_min_default);  unsqueeze_7 = clamp_min_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:867 in build_bm, code: torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
        sub_tensor_1: "i32[]" = torch.ops.aten.sub.Tensor(div_tensor_mode, 1);  div_tensor_mode = None
        clamp_max_tensor_1: "i32[1, 1, 48]" = torch.ops.aten.clamp_max.Tensor(unsqueeze_11, sub_tensor_1);  unsqueeze_11 = sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:406 in create_dense_one, code: index_mask = col_range < kv_num_blocks.unsqueeze(-1)
        unsqueeze_default: "i32[1, 1, 48, 1]" = torch.ops.aten.unsqueeze.default(clamp_max_tensor, 3);  clamp_max_tensor = None
        lt_tensor: "b8[1, 1, 48, 48]" = torch.ops.aten.lt.Tensor(iota_1, unsqueeze_default);  unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:409 in create_dense_one, code: valid_indices = torch.where(index_mask, kv_indices, num_cols)
        where_self: "i32[1, 1, 48, 48]" = torch.ops.aten.where.self(lt_tensor, unsqueeze_9, full_default_1);  lt_tensor = unsqueeze_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:412 in create_dense_one, code: dense_mask[row_indices, valid_indices] = dense_mask.new_ones(())
        index_put_default: "i32[1, 1, 48, 49]" = torch.ops.aten.index_put.default(full_default, [unsqueeze_20, unsqueeze_17, unsqueeze_14, where_self], full_default_2);  where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:413 in create_dense_one, code: return dense_mask[:, :num_cols].contiguous()
        slice_tensor: "i32[1, 1, 48, 48]" = torch.ops.aten.slice.Tensor(index_put_default, 3, 0, 48);  index_put_default = None
        clone_default: "i32[1, 1, 48, 48]" = torch.ops.aten.clone.default(slice_tensor, memory_format = torch.contiguous_format);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:437 in _transpose_ordered, code: return _dense_to_ordered(dense.transpose(-2, -1))
        permute_default: "i32[1, 1, 48, 48]" = torch.ops.aten.permute.default(clone_default, [0, 1, 3, 2]);  clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:425 in _dense_to_ordered, code: num_blocks_in_row = dense_mask.sum(dim=-1)
        sum_dim_int_list: "i64[1, 1, 48]" = torch.ops.aten.sum.dim_IntList(permute_default, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:426 in _dense_to_ordered, code: col_indices = torch.argsort(dense_mask, dim=-1, descending=True, stable=True)
        sort_stable = torch.ops.aten.sort.stable(permute_default, stable = True, descending = True);  permute_default = None
        getitem: "i64[1, 1, 48, 48]" = sort_stable[1];  sort_stable = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:428 in _dense_to_ordered, code: num_blocks_in_row.to(torch.int32, memory_format=torch.contiguous_format),
        convert_element_type_default: "i32[1, 1, 48]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list, torch.int32);  sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:429 in _dense_to_ordered, code: col_indices.to(torch.int32, memory_format=torch.contiguous_format),
        convert_element_type_default_1: "i32[1, 1, 48, 48]" = torch.ops.prims.convert_element_type.default(getitem, torch.int32);  getitem = None
        clone_default_1: "i32[1, 1, 48, 48]" = torch.ops.aten.clone.default(convert_element_type_default_1, memory_format = torch.contiguous_format);  convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:406 in create_dense_one, code: index_mask = col_range < kv_num_blocks.unsqueeze(-1)
        unsqueeze_default_1: "i32[1, 1, 48, 1]" = torch.ops.aten.unsqueeze.default(clamp_max_tensor_1, 3);  clamp_max_tensor_1 = None
        lt_tensor_1: "b8[1, 1, 48, 48]" = torch.ops.aten.lt.Tensor(iota_1, unsqueeze_default_1);  iota_1 = unsqueeze_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:409 in create_dense_one, code: valid_indices = torch.where(index_mask, kv_indices, num_cols)
        where_self_1: "i32[1, 1, 48, 48]" = torch.ops.aten.where.self(lt_tensor_1, unsqueeze_13, full_default_1);  lt_tensor_1 = unsqueeze_13 = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:412 in create_dense_one, code: dense_mask[row_indices, valid_indices] = dense_mask.new_ones(())
        index_put_default_1: "i32[1, 1, 48, 49]" = torch.ops.aten.index_put.default(full_default, [unsqueeze_20, unsqueeze_17, unsqueeze_14, where_self_1], full_default_2);  full_default = unsqueeze_20 = unsqueeze_17 = unsqueeze_14 = where_self_1 = full_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:413 in create_dense_one, code: return dense_mask[:, :num_cols].contiguous()
        slice_tensor_1: "i32[1, 1, 48, 48]" = torch.ops.aten.slice.Tensor(index_put_default_1, 3, 0, 48);  index_put_default_1 = None
        clone_default_2: "i32[1, 1, 48, 48]" = torch.ops.aten.clone.default(slice_tensor_1, memory_format = torch.contiguous_format);  slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:437 in _transpose_ordered, code: return _dense_to_ordered(dense.transpose(-2, -1))
        permute_default_1: "i32[1, 1, 48, 48]" = torch.ops.aten.permute.default(clone_default_2, [0, 1, 3, 2]);  clone_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:425 in _dense_to_ordered, code: num_blocks_in_row = dense_mask.sum(dim=-1)
        sum_dim_int_list_1: "i64[1, 1, 48]" = torch.ops.aten.sum.dim_IntList(permute_default_1, [-1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:426 in _dense_to_ordered, code: col_indices = torch.argsort(dense_mask, dim=-1, descending=True, stable=True)
        sort_stable_1 = torch.ops.aten.sort.stable(permute_default_1, stable = True, descending = True);  permute_default_1 = None
        getitem_1: "i64[1, 1, 48, 48]" = sort_stable_1[1];  sort_stable_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:428 in _dense_to_ordered, code: num_blocks_in_row.to(torch.int32, memory_format=torch.contiguous_format),
        convert_element_type_default_2: "i32[1, 1, 48]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_1, torch.int32);  sum_dim_int_list_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:429 in _dense_to_ordered, code: col_indices.to(torch.int32, memory_format=torch.contiguous_format),
        convert_element_type_default_3: "i32[1, 1, 48, 48]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.int32);  getitem_1 = None
        clone_default_3: "i32[1, 1, 48, 48]" = torch.ops.aten.clone.default(convert_element_type_default_3, memory_format = torch.contiguous_format);  convert_element_type_default_3 = None
        return (convert_element_type_default_2, convert_element_type_default, clone_default_3, clone_default_1)



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
