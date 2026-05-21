"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train
Pattern hash: 5327bafd7855
Shape hash: 8675112c
"""
_shapes_config = "(T([1024, 128, 128], f32), T([256, 4, 128, 128], b8), T([256, 4, 128, 128], f32), S([256, 4, 128, 128]), S([1024, 128, 128]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, bmm_137: "f32[1024, 128, 128]", gt_1: "b8[256, 4, 128, 128]", where_3: "f32[256, 4, 128, 128]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_137, _shape_param_0);  bmm_137 = _shape_param_0 = None
        convert_element_type_default: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_tensor: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  reshape_default = mul_tensor = None
        mul_tensor_2: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_1, where_3);  mul_tensor_1 = None
        sum_dim_int_list: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
        neg_default: "f32[256, 4, 128, 128]" = torch.ops.aten.neg.default(where_3);  where_3 = None
        fma_default: "f32[256, 4, 128, 128]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_2);  neg_default = sum_dim_int_list = mul_tensor_2 = None
        reshape_default_1: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(fma_default, _shape_param_1);  fma_default = _shape_param_1 = None
        return reshape_default_1



def make_inputs():
    return [
    torch.randn([1024, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [256, 4, 128, 128], dtype=torch.bool, device='cuda'),
    torch.randn([256, 4, 128, 128], dtype=torch.float32, device='cuda'),
    [256, 4, 128, 128],  # _shape_param_0
    [1024, 128, 128],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
