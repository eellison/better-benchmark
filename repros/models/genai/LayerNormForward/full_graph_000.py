import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[1152000, 512]", arg1_1: "f32[512]"):
        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:354 in layernorm_fwd, code: x_f32 = x.float()
        convert_element_type: "f32[1152000, 512]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:355 in layernorm_fwd, code: return F.layer_norm(x_f32, w.shape, w, None, 1e-6).to(x.dtype)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [1], correction = 0, keepdim = True)
        getitem: "f32[1152000, 1]" = var_mean[0]
        getitem_1: "f32[1152000, 1]" = var_mean[1];  var_mean = None
        sub: "f32[1152000, 512]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add: "f32[1152000, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt: "f32[1152000, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(mul, arg1_1);  mul = arg1_1 = None
        convert_element_type_1: "bf16[1152000, 512]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        return (convert_element_type_1,)
