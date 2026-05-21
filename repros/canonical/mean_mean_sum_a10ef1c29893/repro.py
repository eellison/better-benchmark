"""
Standalone repro captured via capture_hook.
Label: torchbench_modded_nanogpt_infer
Pattern hash: a10ef1c29893
Shape hash: 7bf65cf8
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
_shapes_config = "(T([], i32), T([1, 1, 48], i32), T([1, 1, 48], i32), T([1, 1, 48, 48], i32, gen=Index(1)), T([1, 1, 48, 48], i32, gen=Index(1)), T([6144, 2304], bf16), T([262144, 64], f32), T([262144, 64], f32), T([12, 2], f32), T([50304, 768], bf16), T([6144], i32, gen=Index(50304)), T([1, 6144, 768], bf16), T([6, 12], f32), S([1, 6144, 2304]), S([1, 6144, 18, 128]), S([1, 6144, 6, 128]), S([1, 6144, 12]), S([1, 12, 6]))"

class Repro(torch.nn.Module):
    def forward(self, arg4_1: "i32[]", unsqueeze_11: "i32[1, 1, 48]", unsqueeze_7: "i32[1, 1, 48]", unsqueeze_9: "i32[1, 1, 48, 48]", unsqueeze_13: "i32[1, 1, 48, 48]", mm_4: "bf16[6144, 2304]", arg14_1: "f32[262144, 64]", arg15_1: "f32[262144, 64]", view_7: "f32[12, 2]", arg2_1: "bf16[50304, 768]", arg0_1: "i32[6144]", convert_element_type_40: "bf16[1, 6144, 768]", arg16_1: "f32[6, 12]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:400 in create_dense_one, code: dense_mask = kv_indices.new_zeros(num_rows, num_cols + 1, dtype=torch.int32)
        full_default: "i32[1, 1, 48, 49]" = torch.ops.aten.full.default([1, 1, 48, 49], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:412 in create_dense_one, code: dense_mask[row_indices, valid_indices] = dense_mask.new_ones(())
        iota_default: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_default, -1);  iota_default = None
        unsqueeze_default_1: "i64[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        unsqueeze_default_2: "i64[1, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, -1);  unsqueeze_default_1 = None
        iota_default_1: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_3: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_default_1, -1);  iota_default_1 = None
        unsqueeze_default_4: "i64[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, -1);  unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:402 in create_dense_one, code: row_indices = torch.arange(num_rows, dtype=torch.int, device=device).unsqueeze(
        iota_default_2: "i32[48]" = torch.ops.prims.iota.default(48, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_5: "i32[48, 1]" = torch.ops.aten.unsqueeze.default(iota_default_2, -1);  iota_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:405 in create_dense_one, code: col_range = torch.arange(num_cols, dtype=torch.int, device=device)
        iota_default_3: "i32[48]" = torch.ops.prims.iota.default(48, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:875 in create_blockmasks, code: sliding_window_num_blocks // 2
        div_tensor_mode: "i32[]" = torch.ops.aten.div.Tensor_mode(arg4_1, 2, rounding_mode = 'floor');  arg4_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:864 in build_bm, code: torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1),
        sub_tensor: "i32[1, 1, 48]" = torch.ops.aten.sub.Tensor(div_tensor_mode, unsqueeze_11)
        clamp_min_default: "i32[1, 1, 48]" = torch.ops.aten.clamp_min.default(sub_tensor, 1);  sub_tensor = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:862 in build_bm, code: torch.clamp_max(
        clamp_max_tensor: "i32[1, 1, 48]" = torch.ops.aten.clamp_max.Tensor(unsqueeze_7, clamp_min_default);  unsqueeze_7 = clamp_min_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:406 in create_dense_one, code: index_mask = col_range < kv_num_blocks.unsqueeze(-1)
        unsqueeze_default_6: "i32[1, 1, 48, 1]" = torch.ops.aten.unsqueeze.default(clamp_max_tensor, 3);  clamp_max_tensor = None
        lt_tensor: "b8[1, 1, 48, 48]" = torch.ops.aten.lt.Tensor(iota_default_3, unsqueeze_default_6);  iota_default_3 = unsqueeze_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:409 in create_dense_one, code: valid_indices = torch.where(index_mask, kv_indices, num_cols)
        full_default_1: "i32[]" = torch.ops.aten.full.default([], 48, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i32[1, 1, 48, 48]" = torch.ops.aten.where.self(lt_tensor, unsqueeze_9, full_default_1);  lt_tensor = unsqueeze_9 = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:412 in create_dense_one, code: dense_mask[row_indices, valid_indices] = dense_mask.new_ones(())
        full_default_2: "i32[1, 1, 1, 1]" = torch.ops.aten.full.default([1, 1, 1, 1], 1, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "i32[1, 1, 48, 49]" = torch.ops.aten.index_put.default(full_default, [unsqueeze_default_2, unsqueeze_default_4, unsqueeze_default_5, where_self], full_default_2);  full_default = unsqueeze_default_2 = unsqueeze_default_4 = unsqueeze_default_5 = where_self = full_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:413 in create_dense_one, code: return dense_mask[:, :num_cols].contiguous()
        slice_tensor: "i32[1, 1, 48, 48]" = torch.ops.aten.slice.Tensor(index_put_default, 3, 0, 48);  index_put_default = None
        clone_default: "i32[1, 1, 48, 48]" = torch.ops.aten.clone.default(slice_tensor, memory_format = torch.contiguous_format);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:437 in _transpose_ordered, code: return _dense_to_ordered(dense.transpose(-2, -1))
        permute_default: "i32[1, 1, 48, 48]" = torch.ops.aten.permute.default(clone_default, [0, 1, 3, 2]);  clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:426 in _dense_to_ordered, code: col_indices = torch.argsort(dense_mask, dim=-1, descending=True, stable=True)
        sort_stable = torch.ops.aten.sort.stable(permute_default, stable = True, descending = True)
        getitem: "i64[1, 1, 48, 48]" = sort_stable[1];  sort_stable = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:400 in create_dense_one, code: dense_mask = kv_indices.new_zeros(num_rows, num_cols + 1, dtype=torch.int32)
        full_default_3: "i32[1, 1, 48, 49]" = torch.ops.aten.full.default([1, 1, 48, 49], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:412 in create_dense_one, code: dense_mask[row_indices, valid_indices] = dense_mask.new_ones(())
        iota_default_4: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_7: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_default_4, -1);  iota_default_4 = None
        unsqueeze_default_8: "i64[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, -1);  unsqueeze_default_7 = None
        unsqueeze_default_9: "i64[1, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, -1);  unsqueeze_default_8 = None
        iota_default_5: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_10: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_default_5, -1);  iota_default_5 = None
        unsqueeze_default_11: "i64[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, -1);  unsqueeze_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:402 in create_dense_one, code: row_indices = torch.arange(num_rows, dtype=torch.int, device=device).unsqueeze(
        iota_default_6: "i32[48]" = torch.ops.prims.iota.default(48, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_12: "i32[48, 1]" = torch.ops.aten.unsqueeze.default(iota_default_6, -1);  iota_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:405 in create_dense_one, code: col_range = torch.arange(num_cols, dtype=torch.int, device=device)
        iota_default_7: "i32[48]" = torch.ops.prims.iota.default(48, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:867 in build_bm, code: torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
        sub_tensor_1: "i32[]" = torch.ops.aten.sub.Tensor(div_tensor_mode, 1);  div_tensor_mode = None
        clamp_max_tensor_1: "i32[1, 1, 48]" = torch.ops.aten.clamp_max.Tensor(unsqueeze_11, sub_tensor_1);  unsqueeze_11 = sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:406 in create_dense_one, code: index_mask = col_range < kv_num_blocks.unsqueeze(-1)
        unsqueeze_default_13: "i32[1, 1, 48, 1]" = torch.ops.aten.unsqueeze.default(clamp_max_tensor_1, 3);  clamp_max_tensor_1 = None
        lt_tensor_1: "b8[1, 1, 48, 48]" = torch.ops.aten.lt.Tensor(iota_default_7, unsqueeze_default_13);  iota_default_7 = unsqueeze_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:409 in create_dense_one, code: valid_indices = torch.where(index_mask, kv_indices, num_cols)
        full_default_4: "i32[]" = torch.ops.aten.full.default([], 48, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "i32[1, 1, 48, 48]" = torch.ops.aten.where.self(lt_tensor_1, unsqueeze_13, full_default_4);  lt_tensor_1 = unsqueeze_13 = full_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:412 in create_dense_one, code: dense_mask[row_indices, valid_indices] = dense_mask.new_ones(())
        full_default_5: "i32[1, 1, 1, 1]" = torch.ops.aten.full.default([1, 1, 1, 1], 1, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "i32[1, 1, 48, 49]" = torch.ops.aten.index_put.default(full_default_3, [unsqueeze_default_9, unsqueeze_default_11, unsqueeze_default_12, where_self_1], full_default_5);  full_default_3 = unsqueeze_default_9 = unsqueeze_default_11 = unsqueeze_default_12 = where_self_1 = full_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:413 in create_dense_one, code: return dense_mask[:, :num_cols].contiguous()
        slice_tensor_1: "i32[1, 1, 48, 48]" = torch.ops.aten.slice.Tensor(index_put_default_1, 3, 0, 48);  index_put_default_1 = None
        clone_default_1: "i32[1, 1, 48, 48]" = torch.ops.aten.clone.default(slice_tensor_1, memory_format = torch.contiguous_format);  slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:437 in _transpose_ordered, code: return _dense_to_ordered(dense.transpose(-2, -1))
        permute_default_1: "i32[1, 1, 48, 48]" = torch.ops.aten.permute.default(clone_default_1, [0, 1, 3, 2]);  clone_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:426 in _dense_to_ordered, code: col_indices = torch.argsort(dense_mask, dim=-1, descending=True, stable=True)
        sort_stable_1 = torch.ops.aten.sort.stable(permute_default_1, stable = True, descending = True)
        getitem_1: "i64[1, 1, 48, 48]" = sort_stable_1[1];  sort_stable_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:678 in forward, code: F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
        reshape_default: "bf16[1, 6144, 2304]" = torch.ops.aten.reshape.default(mm_4, _shape_param_0);  mm_4 = _shape_param_0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:679 in forward, code: .view(B, T, 3 * self.num_heads, self.head_dim)
        reshape_default_1: "bf16[1, 6144, 18, 128]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:680 in forward, code: .chunk(3, dim=-2)
        split_tensor = torch.ops.aten.split.Tensor(reshape_default_1, 6, -2);  reshape_default_1 = None
        getitem_2: "bf16[1, 6144, 6, 128]" = split_tensor[0]
        getitem_3: "bf16[1, 6144, 6, 128]" = split_tensor[1]
        getitem_4: "bf16[1, 6144, 6, 128]" = split_tensor[2];  split_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_default: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_2, torch.float32);  getitem_2 = None
        pow_tensor_scalar: "f32[1, 6144, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2)
        mean_dim: "f32[1, 6144, 6, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [3], True);  pow_tensor_scalar = None
        add_scalar: "f32[1, 6144, 6, 1]" = torch.ops.aten.add.Scalar(mean_dim, 1.1920928955078125e-07);  mean_dim = None
        rsqrt_default: "f32[1, 6144, 6, 1]" = torch.ops.aten.rsqrt.default(add_scalar);  add_scalar = None
        mul_tensor: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default, rsqrt_default);  convert_element_type_default = rsqrt_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        convert_element_type_default_1: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float32);  mul_tensor = None
        split_tensor_1 = torch.ops.aten.split.Tensor(convert_element_type_default_1, 64, -1);  convert_element_type_default_1 = None
        getitem_5: "f32[1, 6144, 6, 64]" = split_tensor_1[0]
        getitem_6: "f32[1, 6144, 6, 64]" = split_tensor_1[1];  split_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:2958 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type_default_2: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
        pow_tensor_scalar_1: "f32[1, 6144, 6, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_2, 2)
        mean_dim_1: "f32[1, 6144, 6, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar_1, [3], True);  pow_tensor_scalar_1 = None
        add_scalar_1: "f32[1, 6144, 6, 1]" = torch.ops.aten.add.Scalar(mean_dim_1, 1.1920928955078125e-07);  mean_dim_1 = None
        rsqrt_default_1: "f32[1, 6144, 6, 1]" = torch.ops.aten.rsqrt.default(add_scalar_1);  add_scalar_1 = None
        mul_tensor_1: "f32[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, rsqrt_default_1);  convert_element_type_default_2 = rsqrt_default_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:640 in forward, code: x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        convert_element_type_default_3: "f32[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.float32);  mul_tensor_1 = None
        split_tensor_2 = torch.ops.aten.split.Tensor(convert_element_type_default_3, 64, -1);  convert_element_type_default_3 = None
        getitem_7: "f32[1, 6144, 6, 64]" = split_tensor_2[0]
        getitem_8: "f32[1, 6144, 6, 64]" = split_tensor_2[1];  split_tensor_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_default_14: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg14_1, 0)
        slice_tensor_2: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_default_14, 1, 0, 6144);  unsqueeze_default_14 = None
        unsqueeze_default_15: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_tensor_2, 2);  slice_tensor_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_tensor_2: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_5, unsqueeze_default_15)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_default_16: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg15_1, 0)
        slice_tensor_3: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_default_16, 1, 0, 6144);  unsqueeze_default_16 = None
        unsqueeze_default_17: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_tensor_3, 2);  slice_tensor_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_tensor_3: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_6, unsqueeze_default_17)
        add_tensor: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_default: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_default_17);  unsqueeze_default_17 = None
        mul_tensor_4: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_5, neg_default);  getitem_5 = neg_default = None
        mul_tensor_5: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_6, unsqueeze_default_15);  getitem_6 = unsqueeze_default_15 = None
        add_tensor_1: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        cat_default: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_tensor, add_tensor_1], 3);  add_tensor = add_tensor_1 = None
        convert_element_type_default_4: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_default, torch.bfloat16);  cat_default = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:691 in forward, code: q.transpose(1, 2),
        permute_default_2: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(convert_element_type_default_4, [0, 2, 1, 3]);  convert_element_type_default_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:637 in forward, code: self.cos[None, : x_BTHD.size(-3), None, :],
        unsqueeze_default_18: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg14_1, 0);  arg14_1 = None
        slice_tensor_4: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_default_18, 1, 0, 6144);  unsqueeze_default_18 = None
        unsqueeze_default_19: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_tensor_4, 2);  slice_tensor_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_tensor_6: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_7, unsqueeze_default_19)

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:638 in forward, code: self.sin[None, : x_BTHD.size(-3), None, :],
        unsqueeze_default_20: "f32[1, 262144, 64]" = torch.ops.aten.unsqueeze.default(arg15_1, 0);  arg15_1 = None
        slice_tensor_5: "f32[1, 6144, 64]" = torch.ops.aten.slice.Tensor(unsqueeze_default_20, 1, 0, 6144);  unsqueeze_default_20 = None
        unsqueeze_default_21: "f32[1, 6144, 1, 64]" = torch.ops.aten.unsqueeze.default(slice_tensor_5, 2);  slice_tensor_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:641 in forward, code: y1 = x1 * cos + x2 * sin
        mul_tensor_7: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_8, unsqueeze_default_21)
        add_tensor_2: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_tensor_6, mul_tensor_7);  mul_tensor_6 = mul_tensor_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:642 in forward, code: y2 = x1 * (-sin) + x2 * cos
        neg_default_1: "f32[1, 6144, 1, 64]" = torch.ops.aten.neg.default(unsqueeze_default_21);  unsqueeze_default_21 = None
        mul_tensor_8: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_7, neg_default_1);  getitem_7 = neg_default_1 = None
        mul_tensor_9: "f32[1, 6144, 6, 64]" = torch.ops.aten.mul.Tensor(getitem_8, unsqueeze_default_19);  getitem_8 = unsqueeze_default_19 = None
        add_tensor_3: "f32[1, 6144, 6, 64]" = torch.ops.aten.add.Tensor(mul_tensor_8, mul_tensor_9);  mul_tensor_8 = mul_tensor_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:643 in forward, code: return torch.cat((y1, y2), 3).type_as(x_BTHD)
        cat_default_1: "f32[1, 6144, 6, 128]" = torch.ops.aten.cat.default([add_tensor_2, add_tensor_3], 3);  add_tensor_2 = add_tensor_3 = None
        convert_element_type_default_5: "bf16[1, 6144, 6, 128]" = torch.ops.prims.convert_element_type.default(cat_default_1, torch.bfloat16);  cat_default_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:692 in forward, code: k.transpose(1, 2),
        permute_default_3: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(convert_element_type_default_5, [0, 2, 1, 3]);  convert_element_type_default_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:924 in forward, code: x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
        select_int: "f32[2]" = torch.ops.aten.select.int(view_7, 0, 1);  view_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:685 in forward, code: v = lambdas[0] * v + lambdas[1] * ve.view_as(
        select_int_1: "f32[]" = torch.ops.aten.select.int(select_int, 0, 0)
        mul_tensor_10: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(select_int_1, getitem_4);  select_int_1 = getitem_4 = None
        select_int_2: "f32[]" = torch.ops.aten.select.int(select_int, 0, 1);  select_int = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:883 in forward, code: ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        embedding_default: "bf16[6144, 768]" = torch.ops.aten.embedding.default(arg2_1, arg0_1);  arg2_1 = arg0_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:685 in forward, code: v = lambdas[0] * v + lambdas[1] * ve.view_as(
        reshape_default_2: "bf16[1, 6144, 6, 128]" = torch.ops.aten.reshape.default(embedding_default, _shape_param_2);  embedding_default = _shape_param_2 = None
        mul_tensor_11: "bf16[1, 6144, 6, 128]" = torch.ops.aten.mul.Tensor(select_int_2, reshape_default_2);  select_int_2 = reshape_default_2 = None
        add_tensor_4: "bf16[1, 6144, 6, 128]" = torch.ops.aten.add.Tensor(mul_tensor_10, mul_tensor_11);  mul_tensor_10 = mul_tensor_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:693 in forward, code: v.transpose(1, 2),
        permute_default_4: "bf16[1, 6, 6144, 128]" = torch.ops.aten.permute.default(add_tensor_4, [0, 2, 1, 3]);  add_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:425 in _dense_to_ordered, code: num_blocks_in_row = dense_mask.sum(dim=-1)
        sum_dim_int_list: "i64[1, 1, 48]" = torch.ops.aten.sum.dim_IntList(permute_default, [-1]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:428 in _dense_to_ordered, code: num_blocks_in_row.to(torch.int32, memory_format=torch.contiguous_format),
        convert_element_type_default_6: "i32[1, 1, 48]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list, torch.int32);  sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:429 in _dense_to_ordered, code: col_indices.to(torch.int32, memory_format=torch.contiguous_format),
        convert_element_type_default_7: "i32[1, 1, 48, 48]" = torch.ops.prims.convert_element_type.default(getitem, torch.int32);  getitem = None
        clone_default_2: "i32[1, 1, 48, 48]" = torch.ops.aten.clone.default(convert_element_type_default_7, memory_format = torch.contiguous_format);  convert_element_type_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:425 in _dense_to_ordered, code: num_blocks_in_row = dense_mask.sum(dim=-1)
        sum_dim_int_list_1: "i64[1, 1, 48]" = torch.ops.aten.sum.dim_IntList(permute_default_1, [-1]);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:428 in _dense_to_ordered, code: num_blocks_in_row.to(torch.int32, memory_format=torch.contiguous_format),
        convert_element_type_default_8: "i32[1, 1, 48]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_1, torch.int32);  sum_dim_int_list_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:429 in _dense_to_ordered, code: col_indices.to(torch.int32, memory_format=torch.contiguous_format),
        convert_element_type_default_9: "i32[1, 1, 48, 48]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.int32);  getitem_1 = None
        clone_default_3: "i32[1, 1, 48, 48]" = torch.ops.aten.clone.default(convert_element_type_default_9, memory_format = torch.contiguous_format);  convert_element_type_default_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:698 in forward, code: y = y * torch.sigmoid(self.attn_gate(x[..., : self.attn_gate_dim])).view(
        slice_tensor_6: "bf16[1, 6144, 12]" = torch.ops.aten.slice.Tensor(convert_element_type_40, 2, 0, 12);  convert_element_type_40 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:618 in forward, code: return F.linear(x, self.weight.type_as(x))
        expand_default: "bf16[1, 6144, 12]" = torch.ops.aten.expand.default(slice_tensor_6, _shape_param_3);  slice_tensor_6 = _shape_param_3 = None
        squeeze_dim: "bf16[6144, 12]" = torch.ops.aten.squeeze.dim(expand_default, 0);  expand_default = None
        convert_element_type_default_10: "bf16[6, 12]" = torch.ops.prims.convert_element_type.default(arg16_1, torch.bfloat16);  arg16_1 = None
        permute_default_5: "bf16[12, 6]" = torch.ops.aten.permute.default(convert_element_type_default_10, [1, 0]);  convert_element_type_default_10 = None
        expand_default_1: "bf16[1, 12, 6]" = torch.ops.aten.expand.default(permute_default_5, _shape_param_4);  permute_default_5 = _shape_param_4 = None
        squeeze_dim_1: "bf16[12, 6]" = torch.ops.aten.squeeze.dim(expand_default_1, 0);  expand_default_1 = None
        return (permute_default_2, permute_default_3, permute_default_4, convert_element_type_default_6, clone_default_2, convert_element_type_default_8, clone_default_3, squeeze_dim, squeeze_dim_1)



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
