"""
Standalone repro captured via capture_hook.
Label: genai_ce_fwd_32768x256
Pattern hash: f0661488d68c
Shape hash: c483969c
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([32768], i64, max=256), T([32768, 256], bf16))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "i64[32768]", arg0_1: "bf16[32768, 256]"):
        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:165 in ce_fwd, code: return F.cross_entropy(x, target, reduction="none")
        ne_scalar: "b8[32768]" = torch.ops.aten.ne.Scalar(arg1_1, -100)
        convert_element_type_default: "f32[32768, 256]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        amax_default: "f32[32768, 1]" = torch.ops.aten.amax.default(convert_element_type_default, [1], True)
        sub_tensor: "f32[32768, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default);  convert_element_type_default = amax_default = None
        exp_default: "f32[32768, 256]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[32768, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[32768, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[32768, 256]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        convert_element_type_default_1: "bf16[32768, 256]" = torch.ops.prims.convert_element_type.default(sub_tensor_1, torch.bfloat16);  sub_tensor_1 = None
        ne_scalar_1: "b8[32768]" = torch.ops.aten.ne.Scalar(arg1_1, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[32768]" = torch.ops.aten.where.self(ne_scalar_1, arg1_1, full_default);  ne_scalar_1 = arg1_1 = full_default = None
        unsqueeze_default: "i64[32768, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "bf16[32768, 1]" = torch.ops.aten.gather.default(convert_element_type_default_1, 1, unsqueeze_default);  convert_element_type_default_1 = unsqueeze_default = None
        squeeze_dim: "bf16[32768]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "bf16[32768]" = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        full_default_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "bf16[32768]" = torch.ops.aten.where.self(ne_scalar, neg_default, full_default_1);  ne_scalar = neg_default = full_default_1 = None
        return where_self_1


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
