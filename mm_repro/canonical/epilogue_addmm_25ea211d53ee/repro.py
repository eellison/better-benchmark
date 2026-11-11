"""Curated MM repro. Pattern hash: 25ea211d53ee."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[512]", arg1_1: "bf16[25088, 512]", arg2_1: "bf16[512, 512]", arg3_1: "i64[14]", arg4_1: "i64[46]", arg5_1: "bf16[128, 14, 14, 512]", arg6_1: "f32[512]", arg7_1: "f32[512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        addmm: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg0_1, arg1_1, arg2_1);  arg0_1 = arg1_1 = arg2_1 = None
        view: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        view_1: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        view_2: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        permute: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_2, [0, 1, 3, 2, 4, 5]);  view_2 = None
        clone: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_3: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        add: "i64[14]" = torch.ops.aten.add.Tensor(arg3_1, 11);  arg3_1 = None
        fmod: "i64[14]" = torch.ops.aten.fmod.Scalar(add, 14);  add = None
        index: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_3, [None, fmod]);  view_3 = None
        index_1: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index, [None, None, fmod]);  index = fmod = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg4_1, 0);  arg4_1 = None
        inductor_random: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_4, inductor_lookup_seed, 'rand');  _shape_param_4 = inductor_lookup_seed = None
        lt: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random, 0.9782608672976494);  inductor_random = None
        convert_element_type: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt, torch.bfloat16);  lt = None
        div: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type, 0.9782608672976494);  convert_element_type = None
        mul: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_1, div);  index_1 = div = None
        add_1: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(arg5_1, mul);  arg5_1 = mul = None
        view_4: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_1, _shape_param_5);  _shape_param_5 = None
        convert_element_type_1: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_1, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 196, 1]" = var_mean[0]
        getitem_1: "f32[128, 196, 1]" = var_mean[1];  var_mean = None
        add_2: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_1, getitem_1);  convert_element_type_1 = None
        mul_1: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_2: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_1, arg6_1);  mul_1 = arg6_1 = None
        add_3: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_2, arg7_1);  mul_2 = arg7_1 = None
        convert_element_type_2: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        return (add_1, getitem_1, rsqrt, convert_element_type_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
