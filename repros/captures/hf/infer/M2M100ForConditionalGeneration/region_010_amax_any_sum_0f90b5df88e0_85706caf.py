"""
Standalone repro captured via capture_hook.
Label: hf_M2M100ForConditionalGeneration_infer
Pattern hash: 0f90b5df88e0
Shape hash: 85706caf
"""
_shapes_config = "(T([1024, 128, 128], f32), T([8192, 1024], f32), S([64, 16, 128, 128]), S([64, -1, 128, 128]), S([64, 16, 128, 128]), S([1024, 128, 128]), S([64, 128, 1024]), S([64, 128, -1, 64]), S([64, 16, 128, 64]), S([1024, 128, 64]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, bmm_24: "f32[1024, 128, 128]", addmm_78: "f32[8192, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default: "f32[64, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_24, _shape_param_0);  bmm_24 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_default: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[128]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_default: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None
        unsqueeze_default_1: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 1);  unsqueeze_default = None
        unsqueeze_default_2: "i64[1, 1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:87 in bidirectional_mask_function, code: return q_idx >= 0
        ge_scalar: "b8[1, 1, 128, 1]" = torch.ops.aten.ge.Scalar(unsqueeze_default_2, 0);  unsqueeze_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_default: "b8[64, 1, 128, 128]" = torch.ops.aten.expand.default(ge_scalar, _shape_param_1);  ge_scalar = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[64, 1, 128, 128]" = torch.ops.aten.where.self(expand_default, full_default, full_default_1);  expand_default = full_default = full_default_1 = None
        add_tensor_1: "f32[64, 16, 128, 128]" = torch.ops.aten.add.Tensor(reshape_default, where_self);  reshape_default = where_self = None
        eq_scalar: "b8[64, 16, 128, 128]" = torch.ops.aten.eq.Scalar(add_tensor_1, -inf)
        logical_not_default: "b8[64, 16, 128, 128]" = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
        any_dim: "b8[64, 16, 128, 1]" = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
        logical_not_default_1: "b8[64, 16, 128, 1]" = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
        full_default_2: "f32[64, 16, 128, 128]" = torch.ops.aten.full.default([64, 16, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_default: "f32[64, 16, 128, 1]" = torch.ops.aten.amax.default(add_tensor_1, [-1], True)
        sub_tensor: "f32[64, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor_1, amax_default);  add_tensor_1 = amax_default = None
        exp_default: "f32[64, 16, 128, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[64, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[64, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        where_self_1: "f32[64, 16, 128, 128]" = torch.ops.aten.where.self(logical_not_default_1, full_default_2, div_tensor);  logical_not_default_1 = full_default_2 = div_tensor = None
        expand_default_1: "f32[64, 16, 128, 128]" = torch.ops.aten.expand.default(where_self_1, _shape_param_2);  where_self_1 = _shape_param_2 = None
        reshape_default_1: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_default_1, _shape_param_3);  expand_default_1 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_2: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(addmm_78, _shape_param_4);  addmm_78 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        reshape_default_3: "f32[64, 128, 16, 64]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_5);  reshape_default_2 = _shape_param_5 = None
        permute_default: "f32[64, 16, 128, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_default_2: "f32[64, 16, 128, 64]" = torch.ops.aten.expand.default(permute_default, _shape_param_6);  permute_default = _shape_param_6 = None
        clone_default: "f32[64, 16, 128, 64]" = torch.ops.aten.clone.default(expand_default_2, memory_format = torch.contiguous_format);  expand_default_2 = None
        reshape_default_4: "f32[1024, 128, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_7);  clone_default = _shape_param_7 = None
        return (reshape_default_1, reshape_default_4)



def make_inputs():
    return [
    torch.randn([1024, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    [64, 16, 128, 128],  # _shape_param_0
    [64, -1, 128, 128],  # _shape_param_1
    [64, 16, 128, 128],  # _shape_param_2
    [1024, 128, 128],  # _shape_param_3
    [64, 128, 1024],  # _shape_param_4
    [64, 128, -1, 64],  # _shape_param_5
    [64, 16, 128, 64],  # _shape_param_6
    [1024, 128, 64],  # _shape_param_7
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
