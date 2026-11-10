"""Curated MM repro. Pattern hash: 9dbea60f5ee3."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[384]", arg1_1: "bf16[401408, 128]", arg2_1: "bf16[128, 384]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10):
        # No stacktrace found for following nodes
        addmm: "bf16[401408, 384]" = torch.ops.aten.addmm.default(arg0_1, arg1_1, arg2_1);  arg0_1 = arg1_1 = arg2_1 = None
        view: "bf16[8192, 49, 384]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        view_1: "bf16[8192, 49, 3, 4, 32]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute: "bf16[3, 8192, 4, 49, 32]" = torch.ops.aten.permute.default(view_1, [2, 0, 3, 1, 4]);  view_1 = None
        unbind = torch.ops.aten.unbind.int(permute);  permute = None
        getitem: "bf16[8192, 4, 49, 32]" = unbind[0]
        getitem_1: "bf16[8192, 4, 49, 32]" = unbind[1]
        getitem_2: "bf16[8192, 4, 49, 32]" = unbind[2];  unbind = None
        mul: "bf16[8192, 4, 49, 32]" = torch.ops.aten.mul.Tensor(getitem, 0.1767766952966369);  getitem = None
        permute_1: "bf16[8192, 4, 32, 49]" = torch.ops.aten.permute.default(getitem_1, [0, 1, 3, 2]);  getitem_1 = None
        expand: "bf16[8192, 4, 49, 32]" = torch.ops.aten.expand.default(mul, _shape_param_2);  mul = _shape_param_2 = None
        clone: "bf16[8192, 4, 49, 32]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_2: "bf16[32768, 49, 32]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        expand_1: "bf16[8192, 4, 32, 49]" = torch.ops.aten.expand.default(permute_1, _shape_param_4);  permute_1 = _shape_param_4 = None
        clone_1: "bf16[8192, 4, 32, 49]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_3: "bf16[32768, 32, 49]" = torch.ops.aten.view.default(clone_1, _shape_param_5);  clone_1 = _shape_param_5 = None
        constant_pad_nd: "bf16[32768, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_2, _shape_param_6);  _shape_param_6 = None
        constant_pad_nd_1: "bf16[32768, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_3, _shape_param_7);  _shape_param_7 = None
        expand_2: "bf16[8192, 4, 49, 32]" = torch.ops.aten.expand.default(getitem_2, _shape_param_8);  getitem_2 = _shape_param_8 = None
        clone_2: "bf16[8192, 4, 49, 32]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_4: "bf16[32768, 49, 32]" = torch.ops.aten.view.default(clone_2, _shape_param_9);  clone_2 = _shape_param_9 = None
        constant_pad_nd_2: "bf16[32768, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_4, _shape_param_10);  _shape_param_10 = None
        permute_2: "bf16[32768, 32, 49]" = torch.ops.aten.permute.default(view_4, [0, 2, 1]);  view_4 = None
        permute_3: "bf16[32768, 32, 49]" = torch.ops.aten.permute.default(view_2, [0, 2, 1]);  view_2 = None
        permute_4: "bf16[32768, 49, 32]" = torch.ops.aten.permute.default(view_3, [0, 2, 1]);  view_3 = None
        return (constant_pad_nd, constant_pad_nd_1, constant_pad_nd_2, permute_2, permute_3, permute_4)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
