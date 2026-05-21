"""
Standalone repro captured via capture_hook.
Label: vllm_Qwen_Qwen3-30B-A3B_000
Pattern hash: 8718bc60001a
Shape hash: d7bccd87
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
_shapes_config = "(T([2048, 128], bf16), T([2048, 2048], bf16))"

class Repro(torch.nn.Module):
    def forward(self, mm_19: "bf16[2048, 128]", view_72: "bf16[2048, 2048]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[2048, 128]" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None
        amax_default: "f32[2048, 1]" = torch.ops.aten.amax.default(convert_element_type_default, [-1], True)
        sub_tensor: "f32[2048, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default);  convert_element_type_default = amax_default = None
        exp_default: "f32[2048, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[2048, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        topk_default = torch.ops.aten.topk.default(div_tensor, 8);  div_tensor = None
        getitem: "f32[2048, 8]" = topk_default[0]
        getitem_1: "i64[2048, 8]" = topk_default[1];  topk_default = None
        view_default: "i64[16384]" = torch.ops.aten.view.default(getitem_1, [-1]);  getitem_1 = None
        sort_default = torch.ops.aten.sort.default(view_default);  view_default = None
        getitem_2: "i64[16384]" = sort_default[0]
        getitem_3: "i64[16384]" = sort_default[1];  sort_default = None
        ge_scalar: "b8[16384]" = torch.ops.aten.ge.Scalar(getitem_2, 128)
        unsqueeze_default: "b8[16384, 1]" = torch.ops.aten.unsqueeze.default(ge_scalar, -1);  ge_scalar = None
        full_default: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        div_tensor_mode: "i64[16384]" = torch.ops.aten.div.Tensor_mode(getitem_3, 8, rounding_mode = 'floor');  getitem_3 = None
        index_tensor: "bf16[16384, 2048]" = torch.ops.aten.index.Tensor(view_72, [div_tensor_mode]);  view_72 = div_tensor_mode = None
        where_self: "bf16[16384, 2048]" = torch.ops.aten.where.self(unsqueeze_default, full_default, index_tensor);  unsqueeze_default = full_default = index_tensor = None
        convert_element_type_default_1: "i32[16384]" = torch.ops.prims.convert_element_type.default(getitem_2, torch.int32);  getitem_2 = None
        return (getitem, convert_element_type_default_1, where_self)



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
