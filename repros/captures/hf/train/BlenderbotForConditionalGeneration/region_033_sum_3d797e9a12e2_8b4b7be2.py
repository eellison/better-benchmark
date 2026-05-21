"""
Standalone repro captured via capture_hook.
Label: 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_train
Pattern hash: 3d797e9a12e2
Shape hash: 8b4b7be2
"""
_shapes_config = "(T([16, 32, 128, 128], f32), T([512, 128, 128], f32), S([16, 32, 128, 128]), S([512, 128, 128]), S([16, 32, 128, 128]), S([512, 128, 128]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, where_1: "f32[16, 32, 128, 128]", bmm_3: "f32[512, 128, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_default: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_1, _shape_param_0);  _shape_param_0 = None
        reshape_default: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_default, _shape_param_1);  expand_default = _shape_param_1 = None
        permute_default: "f32[512, 128, 128]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1]);  reshape_default = None
        reshape_default_1: "f32[16, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_3, _shape_param_2);  bmm_3 = _shape_param_2 = None
        mul_tensor: "f32[16, 32, 128, 128]" = torch.ops.aten.mul.Tensor(reshape_default_1, where_1);  reshape_default_1 = None
        sum_dim_int_list: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True)
        neg_default: "f32[16, 32, 128, 128]" = torch.ops.aten.neg.default(where_1);  where_1 = None
        fma_default: "f32[16, 32, 128, 128]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None
        reshape_default_2: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(fma_default, _shape_param_3);  fma_default = _shape_param_3 = None
        return (permute_default, reshape_default_2)



def make_inputs():
    return [
    torch.randn([16, 32, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([512, 128, 128], dtype=torch.float32, device='cuda'),
    [16, 32, 128, 128],  # _shape_param_0
    [512, 128, 128],  # _shape_param_1
    [16, 32, 128, 128],  # _shape_param_2
    [512, 128, 128],  # _shape_param_3
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
