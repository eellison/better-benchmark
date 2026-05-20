"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf_cuda_h100-2-5-linux.aws.h100_graph36
Pattern hash: 1008dda9a2ed
Shape hash: 9b15f1cf
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([1, 512], i64, gen=Index(1)), T([512, 2304], bf16), S([1, -1, 512, 512]), S([1, 512, 2304]), S([1, 512, -1, 64]), S([1, 512, -1, 64]), S([1, 512, -1, 64]), S([1, 12, 512, 512]))"

class Repro(torch.nn.Module):
    def forward(self, cumsum: "i64[1, 512]", addmm: "bf16[512, 2304]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        iota_default: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        iota_default_1: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[512]" = torch.ops.aten.add.Tensor(iota_default_1, 0);  iota_default_1 = None
        iota_default_2: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_1: "i64[512]" = torch.ops.aten.add.Tensor(iota_default_2, 0);  iota_default_2 = None
        unsqueeze_default: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_default, 1);  iota_default = None
        unsqueeze_default_1: "i64[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "i64[1, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        unsqueeze_default_3: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None
        unsqueeze_default_4: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 1);  unsqueeze_default_3 = None
        unsqueeze_default_5: "i64[1, 1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        unsqueeze_default_6: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_tensor_1, 0);  add_tensor_1 = None
        unsqueeze_default_7: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 1);  unsqueeze_default_6 = None
        unsqueeze_default_8: "i64[1, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        full_default: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        le_tensor: "b8[1, 1, 512, 512]" = torch.ops.aten.le.Tensor(unsqueeze_default_8, unsqueeze_default_5)
        bitwise_and_tensor: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_and.Tensor(full_default, le_tensor);  full_default = le_tensor = None
        index_tensor: "i64[1, 1, 512, 1]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_default_2, unsqueeze_default_5]);  unsqueeze_default_5 = None
        index_tensor_1: "i64[1, 1, 1, 512]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_default_2, unsqueeze_default_8]);  cumsum = unsqueeze_default_2 = unsqueeze_default_8 = None
        eq_tensor: "b8[1, 1, 512, 512]" = torch.ops.aten.eq.Tensor(index_tensor, index_tensor_1);  index_tensor = index_tensor_1 = None
        bitwise_and_tensor_1: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_tensor, eq_tensor);  bitwise_and_tensor = eq_tensor = None
        expand_default: "b8[1, 1, 512, 512]" = torch.ops.aten.expand.default(bitwise_and_tensor_1, _shape_param_0);  bitwise_and_tensor_1 = _shape_param_0 = None
        view_default: "bf16[1, 512, 2304]" = torch.ops.aten.view.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None
        split_tensor = torch.ops.aten.split.Tensor(view_default, 768, 2);  view_default = None
        getitem: "bf16[1, 512, 768]" = split_tensor[0]
        getitem_1: "bf16[1, 512, 768]" = split_tensor[1]
        getitem_2: "bf16[1, 512, 768]" = split_tensor[2];  split_tensor = None
        view_default_1: "bf16[1, 512, 12, 64]" = torch.ops.aten.view.default(getitem_1, _shape_param_2);  getitem_1 = _shape_param_2 = None
        permute_default: "bf16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3]);  view_default_1 = None
        view_default_2: "bf16[1, 512, 12, 64]" = torch.ops.aten.view.default(getitem_2, _shape_param_3);  getitem_2 = _shape_param_3 = None
        permute_default_1: "bf16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        view_default_3: "bf16[1, 512, 12, 64]" = torch.ops.aten.view.default(getitem, _shape_param_4);  getitem = _shape_param_4 = None
        permute_default_2: "bf16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_default_3, [0, 2, 1, 3]);  view_default_3 = None
        full_default_1: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "bf16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_2, full_default_1);  expand_default = full_default_2 = full_default_1 = None
        expand_default_1: "bf16[1, 12, 512, 512]" = torch.ops.aten.expand.default(where_self, _shape_param_5);  where_self = _shape_param_5 = None
        return (permute_default, permute_default_1, permute_default_2, expand_default_1)


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
