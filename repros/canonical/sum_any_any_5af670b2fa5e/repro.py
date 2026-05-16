"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=any, ranges=[], reduction_ranges=[]
#   origins: ['aten.any.dims']
#   type=any, ranges=[], reduction_ranges=[]
#   origins: ['aten.any.dims']
"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_1: "i64[2048, 8]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:246 in forward, code: expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        ge_scalar: "b8[2048, 8]" = torch.ops.aten.ge.Scalar(getitem_1, 0)
        logical_not_default: "b8[2048, 8]" = torch.ops.aten.logical_not.default(ge_scalar);  ge_scalar = None
        any_dims: "b8[]" = torch.ops.aten.any.dims(logical_not_default);  logical_not_default = None
        logical_not_default_1: "b8[]" = torch.ops.aten.logical_not.default(any_dims);  any_dims = None
        _assert_async_msg = torch.ops.aten._assert_async.msg(logical_not_default_1, 'one_hot: Class values must be non-negative.');  logical_not_default_1 = None
        lt_scalar: "b8[2048, 8]" = torch.ops.aten.lt.Scalar(getitem_1, 128)
        logical_not_default_2: "b8[2048, 8]" = torch.ops.aten.logical_not.default(lt_scalar);  lt_scalar = None
        any_dims_1: "b8[]" = torch.ops.aten.any.dims(logical_not_default_2);  logical_not_default_2 = None
        logical_not_default_3: "b8[]" = torch.ops.aten.logical_not.default(any_dims_1);  any_dims_1 = None
        _assert_async_msg_1 = torch.ops.aten._assert_async.msg(logical_not_default_3, 'one_hot: Class values must be smaller than num_classes.');  logical_not_default_3 = None
        unsqueeze_default: "i64[2048, 8, 1]" = torch.ops.aten.unsqueeze.default(getitem_1, -1);  getitem_1 = None
        iota_default: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        eq_tensor: "b8[2048, 8, 128]" = torch.ops.aten.eq.Tensor(unsqueeze_default, iota_default);  unsqueeze_default = iota_default = None
        convert_element_type_default: "i64[2048, 8, 128]" = torch.ops.prims.convert_element_type.default(eq_tensor, torch.int64);  eq_tensor = None
        permute_default: "i64[128, 8, 2048]" = torch.ops.aten.permute.default(convert_element_type_default, [2, 1, 0]);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:249 in forward, code: expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        sum_dim_int_list: "i64[128]" = torch.ops.aten.sum.dim_IntList(permute_default, [-1, -2]);  permute_default = None
        gt_scalar: "b8[128]" = torch.ops.aten.gt.Scalar(sum_dim_int_list, 0);  sum_dim_int_list = None
        return (gt_scalar, _assert_async_msg_1, _assert_async_msg)


def _default_make_inputs():
    return [
    torch.randint(0, 2, [2048, 8], dtype=torch.int64, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
