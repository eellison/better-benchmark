"""
Standalone repro captured via capture_hook.
Label: genai_ce_bwd_32768x256
Pattern hash: ac902649310d
Shape hash: fc42ab9a
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([], bf16), T([32768], i64, gen=Index(32768)), T([32768, 256], bf16), T([32768, 1], f32), T([32768, 1], f32), S([32768]), S([1, 256]), S([32768, 256]))"

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "bf16[]", primals_2: "i64[32768]", primals_1: "bf16[32768, 256]", amax: "f32[32768, 1]", log: "f32[32768, 1]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:192 in ce_bwd, code: return loss.sum()
        expand_default: "bf16[32768]" = torch.ops.aten.expand.default(tangents_1, _shape_param_0);  tangents_1 = _shape_param_0 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:191 in ce_bwd, code: loss = F.cross_entropy(x, target, reduction="none")
        unsqueeze_default: "i64[32768, 1]" = torch.ops.aten.unsqueeze.default(primals_2, 1);  primals_2 = None
        ne_scalar: "b8[32768, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_default, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[32768, 1]" = torch.ops.aten.where.self(ne_scalar, unsqueeze_default, full_default);  unsqueeze_default = full_default = None

        # No stacktrace found for following nodes
        iota_default: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        reshape_default: "i64[1, 256]" = torch.ops.aten.reshape.default(iota_default, _shape_param_1);  iota_default = _shape_param_1 = None
        expand_default_1: "i64[32768, 256]" = torch.ops.aten.expand.default(where_self, _shape_param_2);  where_self = _shape_param_2 = None
        eq_tensor: "b8[32768, 256]" = torch.ops.aten.eq.Tensor(expand_default_1, reshape_default);  expand_default_1 = reshape_default = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:191 in ce_bwd, code: loss = F.cross_entropy(x, target, reduction="none")
        where_self_1: "f32[32768, 256]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        unsqueeze_default_1: "bf16[32768, 1]" = torch.ops.aten.unsqueeze.default(expand_default, 1);  expand_default = None
        full_default_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_2: "bf16[32768, 1]" = torch.ops.aten.where.self(ne_scalar, unsqueeze_default_1, full_default_1);  ne_scalar = unsqueeze_default_1 = full_default_1 = None
        mul_tensor: "f32[32768, 256]" = torch.ops.aten.mul.Tensor(where_self_1, where_self_2);  where_self_1 = where_self_2 = None
        convert_element_type_default: "f32[32768, 256]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float32);  mul_tensor = None
        convert_element_type_default_1: "f32[32768, 256]" = torch.ops.prims.convert_element_type.default(primals_1, torch.float32);  primals_1 = None
        sub_tensor: "f32[32768, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_default_1, amax);  convert_element_type_default_1 = amax = None
        sub_tensor_1: "f32[32768, 256]" = torch.ops.aten.sub.Tensor(sub_tensor, log);  sub_tensor = log = None
        convert_element_type_default_2: "bf16[32768, 256]" = torch.ops.prims.convert_element_type.default(sub_tensor_1, torch.bfloat16);  sub_tensor_1 = None
        convert_element_type_default_3: "f32[32768, 256]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_2, torch.float32);  convert_element_type_default_2 = None
        exp_default: "f32[32768, 256]" = torch.ops.aten.exp.default(convert_element_type_default_3);  convert_element_type_default_3 = None
        sum_dim_int_list: "f32[32768, 1]" = torch.ops.aten.sum.dim_IntList(convert_element_type_default, [1], True)
        mul_tensor_1: "f32[32768, 256]" = torch.ops.aten.mul.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        sub_tensor_2: "f32[32768, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_default, mul_tensor_1);  convert_element_type_default = mul_tensor_1 = None
        convert_element_type_default_4: "bf16[32768, 256]" = torch.ops.prims.convert_element_type.default(sub_tensor_2, torch.bfloat16);  sub_tensor_2 = None
        return convert_element_type_default_4


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
