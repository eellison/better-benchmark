"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_infer
Pattern hash: 77e3c0c677fe
Shape hash: bb1b4a93
"""
_shapes_config = "(T([1024, 128, 128], f32), T([256, 1, 128, 128], b8, stride=(0, 128, 1, 0)), T([32768, 128], f32), S([256, 4, 128, 128]), S([256, 4, 128, 128]), S([1024, 128, 128]), S([256, 128, 128]), S([256, 128, -1, 32]), S([256, 4, 128, 32]), S([1024, 128, 32]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, bmm_46: "f32[1024, 128, 128]", expand: "b8[256, 1, 128, 128]", addmm_350: "f32[32768, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_46, _shape_param_0);  bmm_46 = _shape_param_0 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[256, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default, full_default_1);  expand = full_default = full_default_1 = None
        add_tensor: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(reshape_default, where_self);  reshape_default = where_self = None
        eq_scalar: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_tensor, -inf)
        logical_not_default: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
        any_dim: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
        logical_not_default_1: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
        full_default_2: "f32[256, 4, 128, 128]" = torch.ops.aten.full.default([256, 4, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_default: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_tensor, [-1], True)
        sub_tensor: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  add_tensor = amax_default = None
        exp_default: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        where_self_1: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_default_1, full_default_2, div_tensor);  logical_not_default_1 = full_default_2 = div_tensor = None
        expand_default: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_self_1, _shape_param_1);  where_self_1 = _shape_param_1 = None
        reshape_default_1: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_default, _shape_param_2);  expand_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        reshape_default_2: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_350, _shape_param_3);  addmm_350 = _shape_param_3 = None
        reshape_default_3: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_4);  reshape_default_2 = _shape_param_4 = None
        permute_default: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_default_1: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_default, _shape_param_5);  permute_default = _shape_param_5 = None
        clone_default: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_4: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_default, _shape_param_6);  clone_default = _shape_param_6 = None
        return (reshape_default_1, reshape_default_4)



def make_inputs():
    return [
    torch.randn([1024, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn(128, dtype=torch.bool, device='cuda').as_strided([256, 1, 128, 128], [0, 128, 1, 0]),  # expand
    torch.randn([32768, 128], dtype=torch.float32, device='cuda'),
    [256, 4, 128, 128],  # _shape_param_0
    [256, 4, 128, 128],  # _shape_param_1
    [1024, 128, 128],  # _shape_param_2
    [256, 128, 128],  # _shape_param_3
    [256, 128, -1, 32],  # _shape_param_4
    [256, 4, 128, 32],  # _shape_param_5
    [1024, 128, 32],  # _shape_param_6
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
