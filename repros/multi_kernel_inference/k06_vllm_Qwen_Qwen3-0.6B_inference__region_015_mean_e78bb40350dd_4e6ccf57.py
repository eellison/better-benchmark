"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=mean, ranges=['4', '512', '16', '1'], reduction_ranges=[]
#   origins: ['aten.mean.dim']
#   type=mean, ranges=['4', '512', '8', '1'], reduction_ranges=[]
#   origins: ['aten.mean.dim']
"""
import torch
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, mm: "bf16[2048, 2048]", arg5_1: "bf16[128]", arg2_1: "f32[64]", mm_1: "bf16[2048, 1024]", arg7_1: "bf16[128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:200 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        reshape_default: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm, [4, 512, 2048]);  mm = None
        reshape_default_1: "bf16[4, 512, 16, 128]" = torch.ops.aten.reshape.default(reshape_default, [4, 512, -1, 128]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(reshape_default_1, torch.float32);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2)
        mean_dim: "f32[4, 512, 16, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor: "f32[4, 512, 16, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[4, 512, 16, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default, rsqrt_default);  convert_element_type_default = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_1: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        mul_tensor_1: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(arg5_1, convert_element_type_default_1);  arg5_1 = convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:200 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default: "bf16[4, 16, 512, 128]" = torch.ops.aten.permute.default(mul_tensor_1, [0, 2, 1, 3]);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:322 in forward, code: inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        unsqueeze_default: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(arg2_1, 0);  arg2_1 = None
        unsqueeze_default_1: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        expand_default: "f32[1, 64, 1]" = torch.ops.aten.expand.default(unsqueeze_default_1, [1, -1, 1]);  unsqueeze_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:327 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_default_1: "f32[1, 64, 1]" = torch.ops.aten.expand.default(expand_default, [1, 64, 1]);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:378 in forward, code: cache_position = torch.arange(
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:383 in forward, code: position_ids = cache_position.unsqueeze(0)
        unsqueeze_default_2: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_default, 0);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:323 in forward, code: position_ids_expanded = position_ids[:, None, :].float()
        unsqueeze_default_3: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 1)
        convert_element_type_default_2: "f32[1, 1, 512]" = torch.ops.prims.convert_element_type.default(unsqueeze_default_3, torch.float32);  unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:327 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_default_2: "f32[1, 1, 512]" = torch.ops.aten.expand.default(convert_element_type_default_2, [1, 1, 512]);  convert_element_type_default_2 = None
        mul_tensor_2: "f32[1, 64, 512]" = torch.ops.aten.mul.Tensor(expand_default_1, expand_default_2);  expand_default_1 = expand_default_2 = None
        permute_default_1: "f32[1, 512, 64]" = torch.ops.aten.permute.default(mul_tensor_2, [0, 2, 1]);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:328 in forward, code: emb = torch.cat((freqs, freqs), dim=-1)
        unsqueeze_default_4: "f32[1, 512, 1, 64]" = torch.ops.aten.unsqueeze.default(permute_default_1, 2);  permute_default_1 = None
        expand_default_3: "f32[1, 512, 2, 64]" = torch.ops.aten.expand.default(unsqueeze_default_4, [1, 512, 2, 64]);  unsqueeze_default_4 = None
        clone_default: "f32[1, 512, 2, 64]" = torch.ops.aten.clone.default(expand_default_3, memory_format = torch.contiguous_format);  expand_default_3 = None
        reshape_default_2: "f32[1, 512, 128]" = torch.ops.aten.reshape.default(clone_default, [1, 512, 128]);  clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:329 in forward, code: cos = emb.cos() * self.attention_scaling
        cos_default: "f32[1, 512, 128]" = torch.ops.aten.cos.default(reshape_default_2)
        mul_tensor_3: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(cos_default, 1.0);  cos_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:332 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_default_3: "bf16[1, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.bfloat16);  mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:113 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_default_5: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_3, 1);  convert_element_type_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:115 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_tensor_4: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(permute_default, unsqueeze_default_5)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:89 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_tensor: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:90 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_default: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_tensor);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:88 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_tensor_1: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default, 3, 0, 64);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:90 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_default: "bf16[4, 16, 512, 128]" = torch.ops.aten.cat.default([neg_default, slice_tensor_1], -1);  neg_default = slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:330 in forward, code: sin = emb.sin() * self.attention_scaling
        sin_default: "f32[1, 512, 128]" = torch.ops.aten.sin.default(reshape_default_2);  reshape_default_2 = None
        mul_tensor_5: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sin_default, 1.0);  sin_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:332 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_default_4: "bf16[1, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.bfloat16);  mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:114 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_default_6: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_4, 1);  convert_element_type_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:115 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_tensor_6: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(cat_default, unsqueeze_default_6);  cat_default = None
        add_tensor_1: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_6);  mul_tensor_4 = mul_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:201 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        reshape_default_3: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_1, [4, 512, 1024]);  mm_1 = None
        reshape_default_4: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(reshape_default_3, [4, 512, -1, 128]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default_5: "f32[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(reshape_default_4, torch.float32);  reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar_1: "f32[4, 512, 8, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_5, 2)
        mean_dim_1: "f32[4, 512, 8, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar_1, [-1], True);  pow_tensor_scalar_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor_2: "f32[4, 512, 8, 1]" = torch.ops.aten.add.Tensor(mean_dim_1, 1e-06);  mean_dim_1 = None
        rsqrt_default_1: "f32[4, 512, 8, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor_7: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default_5, rsqrt_default_1);  convert_element_type_default_5 = rsqrt_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_6: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_7, torch.bfloat16);  mul_tensor_7 = None
        mul_tensor_8: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(arg7_1, convert_element_type_default_6);  arg7_1 = convert_element_type_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:201 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_2: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(mul_tensor_8, [0, 2, 1, 3]);  mul_tensor_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:116 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor_9: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_default_2, unsqueeze_default_5);  unsqueeze_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:89 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_tensor_2: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default_2, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:90 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_default_1: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_tensor_2);  slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:88 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_tensor_3: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default_2, 3, 0, 64);  permute_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:90 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_default_1: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_default_1, slice_tensor_3], -1);  neg_default_1 = slice_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:116 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor_10: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_default_1, unsqueeze_default_6);  cat_default_1 = unsqueeze_default_6 = None
        add_tensor_3: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_tensor_9, mul_tensor_10);  mul_tensor_9 = mul_tensor_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:26 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_default_7: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_tensor_3, 2);  add_tensor_3 = None
        expand_default_4: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_default_7, [4, 8, 2, 512, 128]);  unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:27 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_default_1: "bf16[4, 8, 2, 512, 128]" = torch.ops.aten.clone.default(expand_default_4, memory_format = torch.contiguous_format);  expand_default_4 = None
        reshape_default_5: "bf16[4, 16, 512, 128]" = torch.ops.aten.reshape.default(clone_default_1, [4, 16, 512, 128]);  clone_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:739 in _preprocess_mask_arguments, code: position_ids = position_ids.expand(batch_size, -1)
        expand_default_5: "i64[4, 512]" = torch.ops.aten.expand.default(unsqueeze_default_2, [4, -1]);  unsqueeze_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:655 in find_packed_sequence_indices, code: first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
        slice_tensor_4: "i64[4, 1]" = torch.ops.aten.slice.Tensor(expand_default_5, 1, 0, 1)
        sub_tensor: "i64[4, 1]" = torch.ops.aten.sub.Tensor(slice_tensor_4, 1);  slice_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:656 in find_packed_sequence_indices, code: position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
        cat_default_2: "i64[4, 513]" = torch.ops.aten.cat.default([sub_tensor, expand_default_5], -1);  sub_tensor = expand_default_5 = None
        slice_tensor_5: "i64[4, 512]" = torch.ops.aten.slice.Tensor(cat_default_2, -1, 1, 513)
        slice_tensor_6: "i64[4, 512]" = torch.ops.aten.slice.Tensor(cat_default_2, -1, 0, 512);  cat_default_2 = None
        sub_tensor_1: "i64[4, 512]" = torch.ops.aten.sub.Tensor(slice_tensor_5, slice_tensor_6);  slice_tensor_5 = slice_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:657 in find_packed_sequence_indices, code: packed_sequence_mask = (position_diff != 1).cumsum(-1)
        ne_scalar: "b8[4, 512]" = torch.ops.aten.ne.Scalar(sub_tensor_1, 1);  sub_tensor_1 = None
        return (add_tensor_1, reshape_default_5, ne_scalar)



def make_inputs():
    return [
    torch.randn([2048, 2048], dtype=torch.bfloat16, device='cuda'),
    torch.randn([128], dtype=torch.bfloat16, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([128], dtype=torch.bfloat16, device='cuda'),
    ]


def _count_bytes(inputs, outputs):
    """Count total read + write bytes for SOL calculation."""
    total = 0
    for t in inputs:
        if isinstance(t, torch.Tensor):
            total += t.nelement() * t.element_size()
    if isinstance(outputs, torch.Tensor):
        total += outputs.nelement() * outputs.element_size()
    elif isinstance(outputs, (tuple, list)):
        for o in outputs:
            if isinstance(o, torch.Tensor):
                total += o.nelement() * o.element_size()
    return total


def _measure_memcopy_sol(total_bytes, n_warmup=25, n_iter=200):
    """Measure memcopy speed-of-light at multiple transfer sizes.

    Returns dict mapping size label to time in microseconds.
    We measure at the exact kernel size plus a few reference sizes
    so we can see where on the bandwidth curve this kernel sits.
    """
    results = {}
    sizes = set()
    sizes.add(total_bytes)
    # Add reference sizes: 1KB, 64KB, 1MB, 16MB, 64MB, 256MB
    for ref in [1024, 64*1024, 1024*1024, 16*1024*1024, 64*1024*1024, 256*1024*1024]:
        sizes.add(ref)

    for nbytes in sorted(sizes):
        n_elems = max(nbytes // 4, 1)
        src = torch.empty(n_elems, dtype=torch.float32, device="cuda")
        dst = torch.empty_like(src)
        for _ in range(n_warmup):
            dst.copy_(src)
        torch.cuda.synchronize()

        import time
        start = time.perf_counter()
        for _ in range(n_iter):
            dst.copy_(src)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / n_iter

        bw_gbps = (nbytes * 2) / elapsed / 1e9  # read + write
        label = f"{nbytes / 1024:.0f}KB" if nbytes < 1024*1024 else f"{nbytes / 1024 / 1024:.1f}MB"
        is_kernel_size = (nbytes == total_bytes)
        results[nbytes] = {
            "label": label,
            "time_us": elapsed * 1e6,
            "bw_gbps": bw_gbps,
            "is_kernel_size": is_kernel_size,
        }
    return results


def benchmark(n_warmup=25, n_iter=200):
    mod = Repro()
    inputs = make_inputs()

    # Eager baseline
    with torch.no_grad():
        eager_out = mod(*inputs)

    total_bytes = _count_bytes(inputs, eager_out)

    # Compiled (default heuristics)
    compiled = torch.compile(mod)
    with torch.no_grad():
        for _ in range(n_warmup):
            compiled(*inputs)
        torch.cuda.synchronize()

        import time
        start = time.perf_counter()
        for _ in range(n_iter):
            compiled(*inputs)
        torch.cuda.synchronize()
        compiled_time = (time.perf_counter() - start) / n_iter * 1e6

    # Compiled with coordinate descent tuning
    inductor_config.coordinate_descent_tuning = True
    torch._dynamo.reset()
    compiled_cd = torch.compile(mod)
    with torch.no_grad():
        for _ in range(n_warmup):
            compiled_cd(*inputs)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_iter):
            compiled_cd(*inputs)
        torch.cuda.synchronize()
        cd_time = (time.perf_counter() - start) / n_iter * 1e6

    # Memcopy SOL at multiple sizes
    sol_results = _measure_memcopy_sol(total_bytes)
    kernel_sol = sol_results[total_bytes]

    print(f"\nKernel data: {total_bytes / 1024:.1f} KB (read+write)")
    print(f"Compiled (default):      {compiled_time:8.1f} us")
    print(f"Compiled (coord descent):{cd_time:8.1f} us")
    print(f"Memcopy SOL (same size): {kernel_sol['time_us']:8.1f} us  ({kernel_sol['bw_gbps']:.1f} GB/s)")
    print(f"Gap (default / SOL):     {compiled_time / kernel_sol['time_us']:8.2f}x")
    print(f"Gap (CD / SOL):          {cd_time / kernel_sol['time_us']:8.2f}x")
    print(f"\nMemcopy bandwidth curve:")
    for nbytes in sorted(sol_results):
        r = sol_results[nbytes]
        marker = " <-- kernel size" if r["is_kernel_size"] else ""
        print(f"  {r['label']:>10s}: {r['time_us']:8.1f} us  {r['bw_gbps']:8.1f} GB/s{marker}")

    return {
        "compiled_us": compiled_time,
        "coord_descent_us": cd_time,
        "memcopy_sol_us": kernel_sol["time_us"],
        "total_bytes": total_bytes,
        "sol_curve": sol_results,
    }


if __name__ == "__main__":
    benchmark()
