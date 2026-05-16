"""
Standalone repro captured via capture_hook.
Label: falcon_rw_1b
Pattern hash: 2522ec736c14
Shape hash: 81b49207
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm: "f16[2048, 6144]", _shape_param_0, arg7_1: "f16[6144]", _shape_param_1, _shape_param_2, arg2_1: "i64[4, 512]", _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, cumsum: "i64[4, 512]", _shape_param_7, _shape_param_8):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:62 in forward, code: hidden_states = input @ self.weight.T
        reshape_default: "f16[4, 512, 6144]" = torch.ops.aten.reshape.default(mm, _shape_param_0);  mm = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:65 in forward, code: return hidden_states + self.bias
        add_tensor: "f16[4, 512, 6144]" = torch.ops.aten.add.Tensor(reshape_default, arg7_1);  reshape_default = arg7_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:283 in _split_heads, code: fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
        reshape_default_1: "f16[4, 512, 32, 3, 64]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:284 in _split_heads, code: return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]
        select_int: "f16[4, 512, 32, 64]" = torch.ops.aten.select.int(reshape_default_1, 3, 0)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:335 in forward, code: query_layer = query_layer.transpose(1, 2).reshape(batch_size, self.num_heads, query_length, self.head_dim)
        permute_default: "f16[4, 32, 512, 64]" = torch.ops.aten.permute.default(select_int, [0, 2, 1, 3]);  select_int = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:284 in _split_heads, code: return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]
        select_int_1: "f16[4, 512, 32, 64]" = torch.ops.aten.select.int(reshape_default_1, 3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:336 in forward, code: key_layer = key_layer.transpose(1, 2).reshape(batch_size, num_kv_heads, query_length, self.head_dim)
        permute_default_1: "f16[4, 32, 512, 64]" = torch.ops.aten.permute.default(select_int_1, [0, 2, 1, 3]);  select_int_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:143 in update, code: self.keys = torch.cat([self.keys, key_states], dim=-2)
        clone_default: "f16[4, 32, 512, 64]" = torch.ops.aten.clone.default(permute_default_1);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:284 in _split_heads, code: return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]
        select_int_2: "f16[4, 512, 32, 64]" = torch.ops.aten.select.int(reshape_default_1, 3, 2);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:337 in forward, code: value_layer = value_layer.transpose(1, 2).reshape(batch_size, num_kv_heads, query_length, self.head_dim)
        permute_default_2: "f16[4, 32, 512, 64]" = torch.ops.aten.permute.default(select_int_2, [0, 2, 1, 3]);  select_int_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/cache_utils.py:144 in update, code: self.values = torch.cat([self.values, value_states], dim=-2)
        clone_default_1: "f16[4, 32, 512, 64]" = torch.ops.aten.clone.default(permute_default_2);  permute_default_2 = None

        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:158 in __torch_function__, code: return func(*args, **(kwargs or {}))
        full_default: "b8[512, 1]" = torch.ops.aten.full.default([512, 1], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "b8[512, 1]" = torch.ops.aten.full.default([512, 1], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_1: "i64[512]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_default_1: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_2: "i64[512]" = torch.ops.aten.add.Tensor(iota_default_1, 0);  iota_default_1 = None

        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:158 in __torch_function__, code: return func(*args, **(kwargs or {}))
        reshape_default_2: "i64[512, 1]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_2);  add_tensor_2 = _shape_param_2 = None
        le_tensor: "b8[512, 512]" = torch.ops.aten.le.Tensor(add_tensor_1, reshape_default_2);  reshape_default_2 = None
        bitwise_and_tensor: "b8[512, 512]" = torch.ops.aten.bitwise_and.Tensor(full_default_1, le_tensor);  full_default_1 = le_tensor = None
        full_default_2: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        bitwise_and_tensor_1: "b8[512, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_tensor, full_default_2);  bitwise_and_tensor = full_default_2 = None
        bitwise_and_tensor_2: "b8[512, 512]" = torch.ops.aten.bitwise_and.Tensor(full_default, bitwise_and_tensor_1);  full_default = bitwise_and_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:841 in _preprocess_mask_arguments, code: attention_mask = attention_mask.to(device=inputs_embeds.device, dtype=torch.bool)
        convert_element_type_default: "b8[4, 512]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bool)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:509 in sdpa_mask, code: batch_arange = torch.arange(batch_size, device=device)
        iota_default_2: "i64[4]" = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:100 in forward, code: return torch.ops.aten.index(x, indices)
        reshape_default_3: "i64[4, 1]" = torch.ops.aten.reshape.default(iota_default_2, _shape_param_3);  iota_default_2 = _shape_param_3 = None
        index_tensor: "b8[4, 512]" = torch.ops.aten.index.Tensor(convert_element_type_default, [reshape_default_3, add_tensor_1]);  convert_element_type_default = reshape_default_3 = add_tensor_1 = None

        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:158 in __torch_function__, code: return func(*args, **(kwargs or {}))
        reshape_default_4: "b8[4, 1, 512]" = torch.ops.aten.reshape.default(index_tensor, _shape_param_4);  index_tensor = _shape_param_4 = None
        bitwise_and_tensor_3: "b8[4, 512, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_tensor_2, reshape_default_4);  bitwise_and_tensor_2 = reshape_default_4 = None

        # File: /tmp/pytorch-work/torch/_functorch/vmap.py:204 in _maybe_remove_batch_dim, code: return _remove_batch_dim(batched_output, vmap_level, batch_size, out_dim)
        reshape_default_5: "b8[4, 1, 512, 512]" = torch.ops.aten.reshape.default(bitwise_and_tensor_3, _shape_param_5);  bitwise_and_tensor_3 = _shape_param_5 = None
        expand_default: "b8[4, 1, 512, 512]" = torch.ops.aten.expand.default(reshape_default_5, _shape_param_6);  reshape_default_5 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:782 in forward, code: causal_mask, torch.tensor(0.0, device=causal_mask.device, dtype=inputs_embeds.dtype), min_dtype
        full_default_3: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:781 in forward, code: causal_mask = torch.where(
        full_default_4: "f16[]" = torch.ops.aten.full.default([], -65504.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_3, full_default_4);  expand_default = full_default_3 = full_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:789 in forward, code: causal_mask < -1,
        lt_scalar: "b8[4, 1, 512, 512]" = torch.ops.aten.lt.Scalar(where_self, -1);  where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:787 in forward, code: causal_mask = torch.masked_fill(
        full_default_5: "f16[]" = torch.ops.aten.full.default([], -65504.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:171 in build_alibi_tensor, code: base = torch.tensor(
        full_default_6: "f32[]" = torch.ops.aten.full.default([], 0.8408964276313782, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:174 in build_alibi_tensor, code: powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
        iota_default_3: "i32[32]" = torch.ops.prims.iota.default(32, start = 1, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:175 in build_alibi_tensor, code: slopes = torch.pow(base, powers)
        pow_tensor_tensor: "f32[32]" = torch.ops.aten.pow.Tensor_Tensor(full_default_6, iota_default_3);  full_default_6 = iota_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:192 in build_alibi_tensor, code: alibi = slopes[..., None].bfloat16() * arange_tensor
        unsqueeze_default: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(pow_tensor_tensor, 1);  pow_tensor_tensor = None
        convert_element_type_default_1: "bf16[32, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_default, torch.bfloat16);  unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:191 in build_alibi_tensor, code: arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
        sub_tensor: "i64[4, 512]" = torch.ops.aten.sub.Tensor(cumsum, 1);  cumsum = None
        mul_tensor: "i64[4, 512]" = torch.ops.aten.mul.Tensor(sub_tensor, arg2_1);  sub_tensor = arg2_1 = None
        unsqueeze_default_1: "i64[4, 1, 512]" = torch.ops.aten.unsqueeze.default(mul_tensor, 1);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:192 in build_alibi_tensor, code: alibi = slopes[..., None].bfloat16() * arange_tensor
        mul_tensor_1: "bf16[4, 32, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, unsqueeze_default_1);  convert_element_type_default_1 = unsqueeze_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:193 in build_alibi_tensor, code: return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)
        reshape_default_6: "bf16[128, 1, 512]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_7);  mul_tensor_1 = _shape_param_7 = None
        convert_element_type_default_2: "f16[128, 1, 512]" = torch.ops.prims.convert_element_type.default(reshape_default_6, torch.float16);  reshape_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:786 in forward, code: alibi = alibi.reshape(batch_size, -1, *alibi.shape[1:])
        reshape_default_7: "f16[4, 32, 1, 512]" = torch.ops.aten.reshape.default(convert_element_type_default_2, _shape_param_8);  convert_element_type_default_2 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:788 in forward, code: alibi / math.sqrt(self.config.hidden_size // self.num_heads),
        div_tensor: "f16[4, 32, 1, 512]" = torch.ops.aten.div.Tensor(reshape_default_7, 8.0);  reshape_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py:787 in forward, code: causal_mask = torch.masked_fill(
        where_self_1: "f16[4, 32, 512, 512]" = torch.ops.aten.where.self(lt_scalar, full_default_5, div_tensor);  lt_scalar = full_default_5 = div_tensor = None
        return (permute_default, clone_default, clone_default_1, where_self_1)



def make_inputs():
    return [
    torch.randn([2048, 6144], dtype=torch.float16, device='cuda'),
    [4, 512, 6144],  # _shape_param_0
    torch.randn([6144], dtype=torch.float16, device='cuda'),
    [4, 512, 32, 3, 64],  # _shape_param_1
    [512, 1],  # _shape_param_2
    torch.randint(0, 2, [4, 512], dtype=torch.int64, device='cuda'),
    [4, 1],  # _shape_param_3
    [4, 1, 512],  # _shape_param_4
    [4, 1, 512, 512],  # _shape_param_5
    [4, 1, 512, 512],  # _shape_param_6
    torch.randint(0, 2, [4, 512], dtype=torch.int64, device='cuda'),
    [128, 1, 512],  # _shape_param_7
    [4, -1, 1, 512],  # _shape_param_8
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
