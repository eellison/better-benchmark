"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_infer_005
Pattern hash: 9cf9e9271ff1
Shape hash: e5f82c88
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([12, 4096, 64], f16), T([1, 12, 4096, 64], f16, stride=(3145728, 64, 768, 1)), T([4096, 768], f16), S([12, 4096, 1, 1, 1, 64]), S([1, 12, 1, 4096, 64]), S([1, 12, 1, 1]), S([1, 12, 4096]), S([1, 12, 4096]), S([-1, -1, -1, 64]), S([1, 12, -1, 64, 64]), S([1, 12, 64, 64, 64]), S([768, 64, 64]), S([1, 12, 64, 64, 128]), S([768, 64, 128]), S([1, 4096, 768]), S([1, 4096, 12, 64]), S([-1, -1, -1, 64]), S([1, 12, -1, 64, 64]), S([1, 12, 64, 128, 64]), S([768, 128, 64]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_12: "f16[12, 4096, 64]", permute_62: "f16[1, 12, 4096, 64]", mm_19: "f16[4096, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16):
        # No stacktrace found for following nodes
        view_default: "f16[12, 4096, 1, 1, 1, 64]" = torch.ops.aten.view.default(bmm_12, _shape_param_0);  bmm_12 = _shape_param_0 = None
        permute_default: "f16[1, 12, 1, 4096, 64, 1]" = torch.ops.aten.permute.default(view_default, [3, 0, 4, 1, 5, 2]);  view_default = None
        view_default_1: "f16[1, 12, 1, 4096, 64]" = torch.ops.aten.view.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None
        neg_default: "f16[1, 12, 1, 4096, 64]" = torch.ops.aten.neg.default(view_default_1)
        cat_default: "f16[1, 12, 1, 4096, 128]" = torch.ops.aten.cat.default([view_default_1, neg_default], -1);  view_default_1 = neg_default = None
        argmax_default: "i64[1, 12, 1, 4096]" = torch.ops.aten.argmax.default(cat_default, -1);  cat_default = None
        iota_default: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_tensor: "i64[1]" = torch.ops.aten.mul.Tensor(iota_default, 128);  iota_default = None
        view_default_2: "i64[1, 1, 1, 1]" = torch.ops.aten.view.default(mul_tensor, [1, 1, -1, 1]);  mul_tensor = None
        expand_default: "i64[1, 12, 1, 1]" = torch.ops.aten.expand.default(view_default_2, _shape_param_2);  view_default_2 = _shape_param_2 = None
        add_tensor: "i64[1, 12, 1, 4096]" = torch.ops.aten.add.Tensor(argmax_default, expand_default);  argmax_default = expand_default = None
        view_default_3: "i64[1, 12, 4096]" = torch.ops.aten.view.default(add_tensor, _shape_param_3);  add_tensor = _shape_param_3 = None
        mul_tensor_1: "i64[1, 12, 4096]" = torch.ops.aten.mul.Tensor(view_default_3, 4096);  view_default_3 = None
        iota_default_1: "i64[4096]" = torch.ops.prims.iota.default(4096, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default_4: "i64[1, 1, 4096]" = torch.ops.aten.view.default(iota_default_1, [1, 1, -1]);  iota_default_1 = None
        expand_default_1: "i64[1, 12, 4096]" = torch.ops.aten.expand.default(view_default_4, _shape_param_4);  view_default_4 = _shape_param_4 = None
        remainder_scalar: "i64[1, 12, 4096]" = torch.ops.aten.remainder.Scalar(expand_default_1, 4096);  expand_default_1 = None
        add_tensor_1: "i64[1, 12, 4096]" = torch.ops.aten.add.Tensor(mul_tensor_1, remainder_scalar);  mul_tensor_1 = remainder_scalar = None
        sort_default = torch.ops.aten.sort.default(add_tensor_1);  add_tensor_1 = None
        getitem: "i64[1, 12, 4096]" = sort_default[0]
        getitem_1: "i64[1, 12, 4096]" = sort_default[1];  sort_default = None
        remainder_scalar_1: "i64[1, 12, 4096]" = torch.ops.aten.remainder.Scalar(getitem_1, 4096);  getitem_1 = None
        repeat_default: "f16[1, 12, 4096, 64]" = torch.ops.aten.repeat.default(permute_62, [1, 1, 1, 1]);  permute_62 = None
        unsqueeze_default: "i64[1, 12, 4096, 1]" = torch.ops.aten.unsqueeze.default(remainder_scalar_1, -1)
        expand_default_2: "i64[1, 12, 4096, 64]" = torch.ops.aten.expand.default(unsqueeze_default, _shape_param_5);  unsqueeze_default = _shape_param_5 = None
        gather_default: "f16[1, 12, 4096, 64]" = torch.ops.aten.gather.default(repeat_default, 2, expand_default_2);  repeat_default = expand_default_2 = None
        view_default_5: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.view.default(gather_default, _shape_param_6);  gather_default = _shape_param_6 = None
        expand_default_3: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.expand.default(view_default_5, _shape_param_7);  _shape_param_7 = None
        view_default_6: "f16[768, 64, 64]" = torch.ops.aten.view.default(expand_default_3, _shape_param_8);  expand_default_3 = _shape_param_8 = None
        pow_tensor_scalar: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.pow.Tensor_Scalar(view_default_5, 2)
        mean_dim: "f16[1, 12, 64, 64, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None
        add_tensor_2: "f16[1, 12, 64, 64, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f16[1, 12, 64, 64, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor_2: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.mul.Tensor(view_default_5, rsqrt_default);  view_default_5 = rsqrt_default = None
        full_default: "f64[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_tensor: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.div.Tensor(mul_tensor_2, full_default);  mul_tensor_2 = full_default = None
        slice_tensor: "f16[1, 12, 1, 64, 64]" = torch.ops.aten.slice.Tensor(div_tensor, 2, -1, 9223372036854775807)
        slice_tensor_1: "f16[1, 12, 63, 64, 64]" = torch.ops.aten.slice.Tensor(div_tensor, 2, 0, -1)
        cat_default_1: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.cat.default([slice_tensor, slice_tensor_1], 2);  slice_tensor = slice_tensor_1 = None
        cat_default_2: "f16[1, 12, 64, 128, 64]" = torch.ops.aten.cat.default([cat_default_1, div_tensor], 3);  cat_default_1 = div_tensor = None
        permute_default_1: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.permute.default(cat_default_2, [0, 1, 2, 4, 3]);  cat_default_2 = None
        expand_default_4: "f16[1, 12, 64, 64, 128]" = torch.ops.aten.expand.default(permute_default_1, _shape_param_9);  permute_default_1 = _shape_param_9 = None
        view_default_7: "f16[768, 64, 128]" = torch.ops.aten.view.default(expand_default_4, _shape_param_10);  expand_default_4 = _shape_param_10 = None
        view_default_8: "f16[1, 4096, 768]" = torch.ops.aten.view.default(mm_19, _shape_param_11);  mm_19 = _shape_param_11 = None
        view_default_9: "f16[1, 4096, 12, 64]" = torch.ops.aten.view.default(view_default_8, _shape_param_12);  view_default_8 = _shape_param_12 = None
        permute_default_2: "f16[1, 12, 4096, 64]" = torch.ops.aten.permute.default(view_default_9, [0, 2, 1, 3]);  view_default_9 = None
        repeat_default_1: "f16[1, 12, 4096, 64]" = torch.ops.aten.repeat.default(permute_default_2, [1, 1, 1, 1]);  permute_default_2 = None
        unsqueeze_default_1: "i64[1, 12, 4096, 1]" = torch.ops.aten.unsqueeze.default(remainder_scalar_1, -1);  remainder_scalar_1 = None
        expand_default_5: "i64[1, 12, 4096, 64]" = torch.ops.aten.expand.default(unsqueeze_default_1, _shape_param_13);  unsqueeze_default_1 = _shape_param_13 = None
        gather_default_1: "f16[1, 12, 4096, 64]" = torch.ops.aten.gather.default(repeat_default_1, 2, expand_default_5);  repeat_default_1 = expand_default_5 = None
        view_default_10: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.view.default(gather_default_1, _shape_param_14);  gather_default_1 = _shape_param_14 = None
        slice_tensor_2: "f16[1, 12, 1, 64, 64]" = torch.ops.aten.slice.Tensor(view_default_10, 2, -1, 9223372036854775807)
        slice_tensor_3: "f16[1, 12, 63, 64, 64]" = torch.ops.aten.slice.Tensor(view_default_10, 2, 0, -1)
        cat_default_3: "f16[1, 12, 64, 64, 64]" = torch.ops.aten.cat.default([slice_tensor_2, slice_tensor_3], 2);  slice_tensor_2 = slice_tensor_3 = None
        cat_default_4: "f16[1, 12, 64, 128, 64]" = torch.ops.aten.cat.default([cat_default_3, view_default_10], 3);  cat_default_3 = view_default_10 = None
        expand_default_6: "f16[1, 12, 64, 128, 64]" = torch.ops.aten.expand.default(cat_default_4, _shape_param_15);  cat_default_4 = _shape_param_15 = None
        view_default_11: "f16[768, 128, 64]" = torch.ops.aten.view.default(expand_default_6, _shape_param_16);  expand_default_6 = _shape_param_16 = None
        return (getitem, view_default_6, view_default_7, view_default_11)

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
