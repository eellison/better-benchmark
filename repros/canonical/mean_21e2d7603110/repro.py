"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_train_000
Pattern hash: 21e2d7603110
Shape hash: 0840feba
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
_shapes_config = "(T([250112, 512], f32), T([32, 128], i64, gen=Index(250112)), T([512], f32), S([4096, 512]))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[250112, 512]", arg0_1: "i64[32, 128]", arg2_1: "f32[512]", _shape_param_0):
        # No stacktrace found for following nodes
        embedding_default: "f32[32, 128, 512]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        inductor_seeds_default: "i64[84]" = torch.ops.prims.inductor_seeds.default(84, device(type='cuda', index=0))
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_scalar, embedding_default);  gt_scalar = embedding_default = None
        mul_tensor_1: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        pow_tensor_scalar: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_tensor_1, 2)
        mean_dim: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None
        add_tensor: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor_2: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, rsqrt_default);  mul_tensor_1 = rsqrt_default = None
        mul_tensor_3: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg2_1, mul_tensor_2);  arg2_1 = mul_tensor_2 = None
        view_default: "f32[4096, 512]" = torch.ops.aten.view.default(mul_tensor_3, _shape_param_0);  mul_tensor_3 = _shape_param_0 = None
        return view_default



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
