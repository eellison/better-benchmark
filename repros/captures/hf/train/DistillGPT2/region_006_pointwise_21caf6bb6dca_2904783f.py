"""
Standalone repro captured via capture_hook.
Label: hf_DistillGPT2_train
Pattern hash: 21caf6bb6dca
Shape hash: 2904783f
"""
_shapes_config = "(T([1, 512], i64), T([32, 512], i64), T([16384, 2304], f32), S([32, -1, 512, 512]), S([32, 512, 2304]), S([32, 512, -1, 64]), S([32, 512, -1, 64]), S([32, 512, -1, 64]), S([32, 12, 512, 512]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, unsqueeze: "i64[1, 512]", cumsum: "i64[32, 512]", addmm: "f32[16384, 2304]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:509 in sdpa_mask, code: batch_arange = torch.arange(batch_size, device=device)
        iota_default: "i64[32]" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:360 in _non_vmap_expansion_sdpa, code: batch_indices = batch_indices[:, None, None, None]
        unsqueeze_default: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(iota_default, 1);  iota_default = None
        unsqueeze_default_1: "i64[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "i64[32, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_default_3: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_default_4: "i64[1, 1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 3)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_default_5: "i64[1, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:52 in and_mask, code: result = q_idx.new_ones((), dtype=torch.bool)
        full_default: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le_tensor: "b8[1, 1, 512, 512]" = torch.ops.aten.le.Tensor(unsqueeze_default_5, unsqueeze_default_4)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and_tensor: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_and.Tensor(full_default, le_tensor);  full_default = le_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:168 in inner_mask, code: return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]
        index_tensor: "i64[32, 1, 512, 1]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_default_2, unsqueeze_default_4]);  unsqueeze_default_4 = None
        index_tensor_1: "i64[32, 1, 1, 512]" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_default_2, unsqueeze_default_5]);  cumsum = unsqueeze_default_2 = unsqueeze_default_5 = None
        eq_tensor: "b8[32, 1, 512, 512]" = torch.ops.aten.eq.Tensor(index_tensor, index_tensor_1);  index_tensor = index_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and_tensor_1: "b8[32, 1, 512, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_tensor, eq_tensor);  bitwise_and_tensor = eq_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_default: "b8[32, 1, 512, 512]" = torch.ops.aten.expand.default(bitwise_and_tensor_1, _shape_param_0);  bitwise_and_tensor_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        reshape_default: "f32[32, 512, 2304]" = torch.ops.aten.reshape.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_tensor = torch.ops.aten.split.Tensor(reshape_default, 768, 2);  reshape_default = None
        getitem: "f32[32, 512, 768]" = split_tensor[0]
        getitem_1: "f32[32, 512, 768]" = split_tensor[1]
        getitem_2: "f32[32, 512, 768]" = split_tensor[2];  split_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        reshape_default_1: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_1, _shape_param_2);  getitem_1 = _shape_param_2 = None
        permute_default: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        reshape_default_2: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_2, _shape_param_3);  getitem_2 = _shape_param_3 = None
        permute_default_1: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        reshape_default_3: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem, _shape_param_4);  getitem = _shape_param_4 = None
        permute_default_2: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[32, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_2, full_default_1);  expand_default = full_default_2 = full_default_1 = None
        expand_default_1: "f32[32, 12, 512, 512]" = torch.ops.aten.expand.default(where_self, _shape_param_5);  where_self = _shape_param_5 = None
        return (permute_default, permute_default_1, permute_default_2, expand_default_1)



def make_inputs():
    return [
    torch.randint(0, 32, [1, 512], dtype=torch.int64, device='cuda'),
    torch.randint(0, 32, [32, 512], dtype=torch.int64, device='cuda'),
    torch.randn([16384, 2304], dtype=torch.float32, device='cuda'),
    [32, -1, 512, 512],  # _shape_param_0
    [32, 512, 2304],  # _shape_param_1
    [32, 512, -1, 64],  # _shape_param_2
    [32, 512, -1, 64],  # _shape_param_3
    [32, 512, -1, 64],  # _shape_param_4
    [32, 12, 512, 512],  # _shape_param_5
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
