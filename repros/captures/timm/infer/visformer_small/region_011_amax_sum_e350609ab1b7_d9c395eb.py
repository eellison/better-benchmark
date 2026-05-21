"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_infer
Pattern hash: e350609ab1b7
Shape hash: d9c395eb
"""
_shapes_config = "(T([768, 196, 196], f32), T([128, 6, 196, 64], f32, stride=(225792, 64, 1152, 1)), S([128, 6, 196, 196]), S([128, 6, 196, 196]), S([768, 196, 196]), S([128, 6, 196, 64]), S([768, 196, 64]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, bmm_6: "f32[768, 196, 196]", getitem_11: "f32[128, 6, 196, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        reshape_default: "f32[128, 6, 196, 196]" = torch.ops.aten.reshape.default(bmm_6, _shape_param_0);  bmm_6 = _shape_param_0 = None

        # No stacktrace found for following nodes
        mul_tensor: "f32[128, 6, 196, 196]" = torch.ops.aten.mul.Tensor(reshape_default, 1);  reshape_default = None
        amax_default: "f32[128, 6, 196, 1]" = torch.ops.aten.amax.default(mul_tensor, [-1], True)
        sub_tensor: "f32[128, 6, 196, 196]" = torch.ops.aten.sub.Tensor(mul_tensor, amax_default);  mul_tensor = amax_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:117 in forward, code: attn = attn.softmax(dim=-1)
        mul_tensor_1: "f32[128, 6, 196, 196]" = torch.ops.aten.mul.Tensor(sub_tensor, 0.125);  sub_tensor = None
        exp_default: "f32[128, 6, 196, 196]" = torch.ops.aten.exp.default(mul_tensor_1);  mul_tensor_1 = None
        sum_dim_int_list: "f32[128, 6, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[128, 6, 196, 196]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        expand_default: "f32[128, 6, 196, 196]" = torch.ops.aten.expand.default(div_tensor, _shape_param_1);  div_tensor = _shape_param_1 = None
        reshape_default_1: "f32[768, 196, 196]" = torch.ops.aten.reshape.default(expand_default, _shape_param_2);  expand_default = _shape_param_2 = None
        expand_default_1: "f32[128, 6, 196, 64]" = torch.ops.aten.expand.default(getitem_11, _shape_param_3);  getitem_11 = _shape_param_3 = None
        clone_default: "f32[128, 6, 196, 64]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_2: "f32[768, 196, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None
        return (reshape_default_1, reshape_default_2)



def make_inputs():
    return [
    torch.randn([768, 196, 196], dtype=torch.float32, device='cuda'),
    torch.randn(28900608, dtype=torch.float32, device='cuda').as_strided([128, 6, 196, 64], [225792, 64, 1152, 1]),  # getitem_11
    [128, 6, 196, 196],  # _shape_param_0
    [128, 6, 196, 196],  # _shape_param_1
    [768, 196, 196],  # _shape_param_2
    [128, 6, 196, 64],  # _shape_param_3
    [768, 196, 64],  # _shape_param_4
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
