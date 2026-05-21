"""
Standalone repro captured via capture_hook.
Label: hf_DistilBertForMaskedLM_train
Pattern hash: fd206342afb8
Shape hash: 26c075b5
"""
_shapes_config = "(T([3072, 128, 128], f32), T([256, 12, 128, 128], b8), T([1, 1, 128, 1], b8), T([], f32), T([3072, 128, 128], f32), T([256, 12, 128, 1], f32), T([256, 12, 128, 1], f32), T([256, 12, 128, 1], b8), S([256, 12, 128, 128]), S([256, -1, 128, 128]), S([256, 12, 128, 128]), S([3072, 128, 128]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, bmm_33: "f32[3072, 128, 128]", gt_1: "b8[256, 12, 128, 128]", ge: "b8[1, 1, 128, 1]", full_default_1: "f32[]", bmm: "f32[3072, 128, 128]", amax: "f32[256, 12, 128, 1]", sum_1: "f32[256, 12, 128, 1]", logical_not_1: "b8[256, 12, 128, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default: "f32[256, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm_33, _shape_param_0);  bmm_33 = _shape_param_0 = None
        convert_element_type_default: "f32[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_tensor: "f32[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  reshape_default = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_default: "b8[256, 1, 128, 128]" = torch.ops.aten.expand.default(ge, _shape_param_1);  ge = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[256, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default_1, full_default);  expand_default = full_default_1 = full_default = None
        reshape_default_1: "f32[256, 12, 128, 128]" = torch.ops.aten.reshape.default(bmm, _shape_param_2);  bmm = _shape_param_2 = None
        add_tensor: "f32[256, 12, 128, 128]" = torch.ops.aten.add.Tensor(reshape_default_1, where_self);  reshape_default_1 = where_self = None
        sub_tensor: "f32[256, 12, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor, amax);  add_tensor = amax = None
        exp_default: "f32[256, 12, 128, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        div_tensor: "f32[256, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_1);  exp_default = sum_1 = None
        full_default_2: "f32[256, 12, 128, 128]" = torch.ops.aten.full.default([256, 12, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[256, 12, 128, 128]" = torch.ops.aten.where.self(logical_not_1, full_default_2, div_tensor);  logical_not_1 = full_default_2 = div_tensor = None
        mul_tensor_2: "f32[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_1, where_self_1);  mul_tensor_1 = None
        sum_dim_int_list: "f32[256, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
        neg_default: "f32[256, 12, 128, 128]" = torch.ops.aten.neg.default(where_self_1);  where_self_1 = None
        fma_default: "f32[256, 12, 128, 128]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_2);  neg_default = sum_dim_int_list = mul_tensor_2 = None
        reshape_default_2: "f32[3072, 128, 128]" = torch.ops.aten.reshape.default(fma_default, _shape_param_3);  fma_default = _shape_param_3 = None
        return reshape_default_2



def make_inputs():
    return [
    torch.randn([3072, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [256, 12, 128, 128], dtype=torch.bool, device='cuda'),
    torch.randint(0, 2, [1, 1, 128, 1], dtype=torch.bool, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([256, 12, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256, 12, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [256, 12, 128, 1], dtype=torch.bool, device='cuda'),
    [256, 12, 128, 128],  # _shape_param_0
    [256, -1, 128, 128],  # _shape_param_1
    [256, 12, 128, 128],  # _shape_param_2
    [3072, 128, 128],  # _shape_param_3
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
