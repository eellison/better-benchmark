"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: c2ee9c72bf0f
Shape hash: c4929701
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, div_4: "f32[64, 6, 49, 49]", _shape_param_0, _shape_param_1, bmm_61: "f32[384, 49, 49]", _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:214 in shifted_window_attention, code: x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
        expand_default: "f32[64, 6, 49, 49]" = torch.ops.aten.expand.default(div_4, _shape_param_0);  _shape_param_0 = None
        reshape_default: "f32[384, 49, 49]" = torch.ops.aten.reshape.default(expand_default, _shape_param_1);  expand_default = _shape_param_1 = None
        permute_default: "f32[384, 49, 49]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1]);  reshape_default = None
        reshape_default_1: "f32[64, 6, 49, 49]" = torch.ops.aten.reshape.default(bmm_61, _shape_param_2);  bmm_61 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:211 in shifted_window_attention, code: attn = F.softmax(attn, dim=-1)
        mul_tensor: "f32[64, 6, 49, 49]" = torch.ops.aten.mul.Tensor(reshape_default_1, div_4);  reshape_default_1 = None
        sum_dim_int_list: "f32[64, 6, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True)
        neg_default: "f32[64, 6, 49, 49]" = torch.ops.aten.neg.default(div_4);  div_4 = None
        fma_default: "f32[64, 6, 49, 49]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:189 in shifted_window_attention, code: attn = q.matmul(k.transpose(-2, -1))
        reshape_default_2: "f32[384, 49, 49]" = torch.ops.aten.reshape.default(fma_default, _shape_param_3);  fma_default = _shape_param_3 = None
        return (permute_default, reshape_default_2)



def make_inputs():
    return [
    torch.randn(933857, dtype=torch.float32, device='cuda').as_strided([64, 6, 49, 49], [14592, 2432, 49, 1]),  # div_4
    [64, 6, 49, 49],  # _shape_param_0
    [384, 49, 49],  # _shape_param_1
    torch.randn([384, 49, 49], dtype=torch.float32, device='cuda'),
    [64, 6, 49, 49],  # _shape_param_2
    [384, 49, 49],  # _shape_param_3
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
