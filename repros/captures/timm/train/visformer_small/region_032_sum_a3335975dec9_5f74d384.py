"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_train
Pattern hash: a3335975dec9
Shape hash: 5f74d384
"""
_shapes_config = "(T([128, 6, 49, 49], f32), T([768, 49, 49], f32), S([128, 6, 49, 49]), S([768, 49, 49]), S([128, 6, 49, 49]), S([768, 49, 49]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, div_4: "f32[128, 6, 49, 49]", bmm_29: "f32[768, 49, 49]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        expand_default: "f32[128, 6, 49, 49]" = torch.ops.aten.expand.default(div_4, _shape_param_0);  _shape_param_0 = None
        reshape_default: "f32[768, 49, 49]" = torch.ops.aten.reshape.default(expand_default, _shape_param_1);  expand_default = _shape_param_1 = None
        permute_default: "f32[768, 49, 49]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1]);  reshape_default = None
        reshape_default_1: "f32[128, 6, 49, 49]" = torch.ops.aten.reshape.default(bmm_29, _shape_param_2);  bmm_29 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:117 in forward, code: attn = attn.softmax(dim=-1)
        mul_tensor: "f32[128, 6, 49, 49]" = torch.ops.aten.mul.Tensor(reshape_default_1, div_4);  reshape_default_1 = None
        sum_dim_int_list: "f32[128, 6, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True)
        neg_default: "f32[128, 6, 49, 49]" = torch.ops.aten.neg.default(div_4);  div_4 = None
        fma_default: "f32[128, 6, 49, 49]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        mul_tensor_1: "f32[128, 6, 49, 49]" = torch.ops.aten.mul.Tensor(fma_default, 0.08838834764831845);  fma_default = None
        reshape_default_2: "f32[768, 49, 49]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_3);  mul_tensor_1 = _shape_param_3 = None
        return (permute_default, reshape_default_2)



def make_inputs():
    return [
    torch.randn([128, 6, 49, 49], dtype=torch.float32, device='cuda'),
    torch.randn([768, 49, 49], dtype=torch.float32, device='cuda'),
    [128, 6, 49, 49],  # _shape_param_0
    [768, 49, 49],  # _shape_param_1
    [128, 6, 49, 49],  # _shape_param_2
    [768, 49, 49],  # _shape_param_3
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
