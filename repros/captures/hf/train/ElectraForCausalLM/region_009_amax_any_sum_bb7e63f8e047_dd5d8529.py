"""
Standalone repro captured via capture_hook.
Label: hf_ElectraForCausalLM_train
Pattern hash: bb7e63f8e047
Shape hash: dd5d8529
"""
_shapes_config = "(T([32768, 256], f32), T([256, 512, 512], f32), T([37], i64), S([64, -1, 512, 512]), S([64, 512, 256]), S([64, 512, -1, 64]), S([64, 4, 512, 512]), S([64, 4, 512, 512]), S([256, 512, 512]), S([64, 4, 512, 64]), S([256, 512, 64]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_3: "f32[32768, 256]", bmm: "f32[256, 512, 512]", inductor_seeds_default: "i64[37]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[512]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_default: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None
        unsqueeze_default_1: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 1);  unsqueeze_default = None
        unsqueeze_default_2: "i64[1, 1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:87 in bidirectional_mask_function, code: return q_idx >= 0
        ge_scalar: "b8[1, 1, 512, 1]" = torch.ops.aten.ge.Scalar(unsqueeze_default_2, 0);  unsqueeze_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_default: "b8[64, 1, 512, 512]" = torch.ops.aten.expand.default(ge_scalar, _shape_param_0);  ge_scalar = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_3, _shape_param_1);  addmm_3 = _shape_param_1 = None
        reshape_default_1: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_2);  reshape_default = _shape_param_2 = None
        permute_default: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[64, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_1, full_default);  expand_default = full_default_1 = full_default = None
        reshape_default_2: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm, _shape_param_3);  bmm = _shape_param_3 = None
        add_tensor_1: "f32[64, 4, 512, 512]" = torch.ops.aten.add.Tensor(reshape_default_2, where_self);  reshape_default_2 = where_self = None
        amax_default: "f32[64, 4, 512, 1]" = torch.ops.aten.amax.default(add_tensor_1, [-1], True)
        sub_tensor: "f32[64, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_tensor_1, amax_default);  amax_default = None
        exp_default: "f32[64, 4, 512, 512]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[64, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        eq_scalar: "b8[64, 4, 512, 512]" = torch.ops.aten.eq.Scalar(add_tensor_1, -inf);  add_tensor_1 = None
        logical_not_default: "b8[64, 4, 512, 512]" = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
        any_dim: "b8[64, 4, 512, 1]" = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
        logical_not_default_1: "b8[64, 4, 512, 1]" = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
        full_default_2: "f32[64, 4, 512, 512]" = torch.ops.aten.full.default([64, 4, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[64, 4, 512, 512]" = torch.ops.aten.where.self(logical_not_default_1, full_default_2, div_tensor);  logical_not_default_1 = full_default_2 = div_tensor = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1);  inductor_seeds_default = None
        inductor_random_default: "f32[64, 4, 512, 512]" = torch.ops.prims.inductor_random.default([64, 4, 512, 512], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[64, 4, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(gt_scalar, where_self_1);  gt_scalar = where_self_1 = None
        mul_tensor_1: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        expand_default_1: "f32[64, 4, 512, 512]" = torch.ops.aten.expand.default(mul_tensor_1, _shape_param_4);  mul_tensor_1 = _shape_param_4 = None
        reshape_default_3: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(expand_default_1, _shape_param_5);  expand_default_1 = _shape_param_5 = None
        expand_default_2: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(permute_default, _shape_param_6);  permute_default = _shape_param_6 = None
        clone_default: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_default_2, memory_format = torch.contiguous_format);  expand_default_2 = None
        reshape_default_4: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_7);  clone_default = _shape_param_7 = None
        return (reshape_default_3, reshape_default_4)



def make_inputs():
    return [
    torch.randn([32768, 256], dtype=torch.float32, device='cuda'),
    torch.randn([256, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 37, [37], dtype=torch.int64, device='cuda'),
    [64, -1, 512, 512],  # _shape_param_0
    [64, 512, 256],  # _shape_param_1
    [64, 512, -1, 64],  # _shape_param_2
    [64, 4, 512, 512],  # _shape_param_3
    [64, 4, 512, 512],  # _shape_param_4
    [256, 512, 512],  # _shape_param_5
    [64, 4, 512, 64],  # _shape_param_6
    [256, 512, 64],  # _shape_param_7
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
