"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-5-5-linux.aws.a100_graph11
Pattern hash: c118f656672c
Shape hash: 618c5f21
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([1, 512], i64, max=512), T([32000, 1024], bf16), S([1, 512, 1024]), S([1, 512, 1024]), S([1, 1024, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[1, 512]", _param_constant0: "bf16[32000, 1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        permute_default: "i64[512, 1]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        embedding_default: "bf16[512, 1, 1024]" = torch.ops.aten.embedding.default(_param_constant0, permute_default);  _param_constant0 = permute_default = None
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_default: "f32[512]" = torch.ops.prims.convert_element_type.default(iota_default, torch.float32);  iota_default = None
        div_tensor: "f32[512]" = torch.ops.aten.div.Tensor(convert_element_type_default, 1024);  convert_element_type_default = None
        pow_scalar: "f32[512]" = torch.ops.aten.pow.Scalar(10000, div_tensor);  div_tensor = None
        reciprocal_default: "f32[512]" = torch.ops.aten.reciprocal.default(pow_scalar);  pow_scalar = None
        mul_tensor: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        iota_default_1: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 512, step = -1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_default_1: "f32[1024]" = torch.ops.prims.convert_element_type.default(iota_default_1, torch.float32);  iota_default_1 = None
        unsqueeze_default: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_1, 1);  convert_element_type_default_1 = None
        permute_default_1: "f32[1024, 1]" = torch.ops.aten.permute.default(unsqueeze_default, [0, 1]);  unsqueeze_default = None
        unsqueeze_default_1: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, 1);  mul_tensor = None
        permute_default_2: "f32[1, 512]" = torch.ops.aten.permute.default(unsqueeze_default_1, [1, 0]);  unsqueeze_default_1 = None
        mul_tensor_1: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(permute_default_1, permute_default_2);  permute_default_1 = permute_default_2 = None
        sin_default: "f32[1024, 512]" = torch.ops.aten.sin.default(mul_tensor_1)
        cos_default: "f32[1024, 512]" = torch.ops.aten.cos.default(mul_tensor_1);  mul_tensor_1 = None
        cat_default: "f32[1024, 1024]" = torch.ops.aten.cat.default([sin_default, cos_default], -1);  sin_default = cos_default = None
        unsqueeze_default_2: "f32[1024, 1, 1024]" = torch.ops.aten.unsqueeze.default(cat_default, 1);  cat_default = None
        expand_default: "f32[1024, 1, 1024]" = torch.ops.aten.expand.default(unsqueeze_default_2, [-1, 1, -1]);  unsqueeze_default_2 = None
        unsqueeze_default_3: "bf16[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(embedding_default, 3)
        unsqueeze_default_4: "bf16[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 4);  unsqueeze_default_3 = None
        permute_default_3: "bf16[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_default_4, [0, 1, 3, 4, 2]);  unsqueeze_default_4 = None
        permute_default_4: "bf16[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_default_3, [0, 4, 1, 2, 3]);  permute_default_3 = None
        view_default: "bf16[1, 512, 1024]" = torch.ops.aten.view.default(permute_default_4, _shape_param_0);  permute_default_4 = _shape_param_0 = None
        squeeze_dim: "bf16[512, 1024]" = torch.ops.aten.squeeze.dim(view_default, 0);  view_default = None
        unsqueeze_default_5: "bf16[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(embedding_default, 3);  embedding_default = None
        unsqueeze_default_6: "bf16[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 4);  unsqueeze_default_5 = None
        permute_default_5: "bf16[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_default_6, [0, 1, 3, 4, 2]);  unsqueeze_default_6 = None
        permute_default_6: "bf16[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_default_5, [0, 4, 1, 2, 3]);  permute_default_5 = None
        view_default_1: "bf16[1, 512, 1024]" = torch.ops.aten.view.default(permute_default_6, _shape_param_1);  permute_default_6 = _shape_param_1 = None
        squeeze_dim_1: "bf16[512, 1024]" = torch.ops.aten.squeeze.dim(view_default_1, 0);  view_default_1 = None
        convert_element_type_default_2: "bf16[1024, 1, 1024]" = torch.ops.prims.convert_element_type.default(expand_default, torch.bfloat16);  expand_default = None
        unsqueeze_default_7: "bf16[1024, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 3);  convert_element_type_default_2 = None
        unsqueeze_default_8: "bf16[1024, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 4);  unsqueeze_default_7 = None
        permute_default_7: "bf16[1024, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_default_8, [0, 1, 3, 4, 2]);  unsqueeze_default_8 = None
        permute_default_8: "bf16[1024, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_default_7, [0, 4, 1, 2, 3]);  permute_default_7 = None
        view_default_2: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(permute_default_8, _shape_param_2);  permute_default_8 = _shape_param_2 = None
        squeeze_dim_2: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_default_2, 0);  view_default_2 = None
        return (squeeze_dim, squeeze_dim_1, squeeze_dim_2)


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
