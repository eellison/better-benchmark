"""Curated MM repro. Pattern hash: 86e9ceaf1cb3."""
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
    def forward(self, arg0_1: "bf16[128]", arg1_1: "bf16[401408, 512]", arg2_1: "bf16[512, 128]", arg3_1: "bf16[128, 3136, 128]", arg4_1: "bf16[128]", arg5_1: "bf16[128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        addmm: "bf16[401408, 128]" = torch.ops.aten.addmm.default(arg0_1, arg1_1, arg2_1);  arg0_1 = arg1_1 = arg2_1 = None
        view: "bf16[128, 3136, 128]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        add: "bf16[128, 3136, 128]" = torch.ops.aten.add.Tensor(arg3_1, view);  arg3_1 = view = None
        view_1: "bf16[128, 56, 56, 128]" = torch.ops.aten.view.default(add, _shape_param_1);  add = _shape_param_1 = None
        convert_element_type: "f32[128, 56, 56, 128]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 56, 56, 1]" = var_mean[0]
        getitem_1: "f32[128, 56, 56, 1]" = var_mean[1];  var_mean = None
        sub: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_1: "f32[128, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[128, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        add_2: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        convert_element_type_1: "bf16[128, 56, 56, 128]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        iota: "i64[56]" = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_3: "i64[56]" = torch.ops.aten.add.Tensor(iota, 3);  iota = None
        fmod: "i64[56]" = torch.ops.aten.fmod.Scalar(add_3, 56);  add_3 = None
        index: "bf16[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(convert_element_type_1, [None, fmod]);  convert_element_type_1 = fmod = None
        iota_1: "i64[56]" = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_4: "i64[56]" = torch.ops.aten.add.Tensor(iota_1, 3);  iota_1 = None
        fmod_1: "i64[56]" = torch.ops.aten.fmod.Scalar(add_4, 56);  add_4 = None
        index_1: "bf16[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(index, [None, None, fmod_1]);  index = fmod_1 = None
        view_2: "bf16[128, 8, 7, 8, 7, 128]" = torch.ops.aten.view.default(index_1, _shape_param_2);  index_1 = _shape_param_2 = None
        permute: "bf16[128, 8, 8, 7, 7, 128]" = torch.ops.aten.permute.default(view_2, [0, 1, 3, 2, 4, 5]);  view_2 = None
        clone: "bf16[128, 8, 8, 7, 7, 128]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_3: "bf16[8192, 7, 7, 128]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        view_4: "bf16[8192, 49, 128]" = torch.ops.aten.view.default(view_3, _shape_param_4);  view_3 = _shape_param_4 = None
        view_5: "bf16[401408, 128]" = torch.ops.aten.view.default(view_4, _shape_param_5);  view_4 = _shape_param_5 = None
        return (view_1, view_5)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
