"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train
Pattern hash: 3d797e9a12e2
Shape hash: 313636bb
"""
_shapes_config = "(T([2048, 8, 49, 49], f32, stride=(19456, 2432, 49, 1)), T([16384, 49, 49], f32), S([2048, 8, 49, 49]), S([16384, 49, 49]), S([2048, 8, 49, 49]), S([16384, 49, 49]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, div_4: "f32[2048, 8, 49, 49]", bmm_133: "f32[16384, 49, 49]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_default: "f32[2048, 8, 49, 49]" = torch.ops.aten.expand.default(div_4, _shape_param_0);  _shape_param_0 = None
        reshape_default: "f32[16384, 49, 49]" = torch.ops.aten.reshape.default(expand_default, _shape_param_1);  expand_default = _shape_param_1 = None
        permute_default: "f32[16384, 49, 49]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1]);  reshape_default = None
        reshape_default_1: "f32[2048, 8, 49, 49]" = torch.ops.aten.reshape.default(bmm_133, _shape_param_2);  bmm_133 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        mul_tensor: "f32[2048, 8, 49, 49]" = torch.ops.aten.mul.Tensor(reshape_default_1, div_4);  reshape_default_1 = None
        sum_dim_int_list: "f32[2048, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True)
        neg_default: "f32[2048, 8, 49, 49]" = torch.ops.aten.neg.default(div_4);  div_4 = None
        fma_default: "f32[2048, 8, 49, 49]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        reshape_default_2: "f32[16384, 49, 49]" = torch.ops.aten.reshape.default(fma_default, _shape_param_3);  fma_default = _shape_param_3 = None
        return (permute_default, reshape_default_2)



def make_inputs():
    return [
    torch.randn(39845857, dtype=torch.float32, device='cuda').as_strided([2048, 8, 49, 49], [19456, 2432, 49, 1]),  # div_4
    torch.randn([16384, 49, 49], dtype=torch.float32, device='cuda'),
    [2048, 8, 49, 49],  # _shape_param_0
    [16384, 49, 49],  # _shape_param_1
    [2048, 8, 49, 49],  # _shape_param_2
    [16384, 49, 49],  # _shape_param_3
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
