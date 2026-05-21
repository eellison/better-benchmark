"""
Standalone repro captured via capture_hook.
Label: hf_GPTNeoForSequenceClassification_train
Pattern hash: 70e6cdffba4d
Shape hash: 954b6941
"""
_shapes_config = "(T([4096, 2048], f32), T([512, 128, 128], f32), T([1, 1, 2048, 2048], b8), T([], f32), T([32, 1, 128, 128], f32), S([32, 128, 2048]), S([32, 128, 16, 128]), S([32, 16, 128, 128]), S([32, 16, 128, 128]), S([512, 128, 128]), S([32, 16, 128, 128]), S([512, 128, 128]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_71: "f32[4096, 2048]", bmm_46: "f32[512, 128, 128]", primals_331: "b8[1, 1, 2048, 2048]", full_default_2: "f32[]", where: "f32[32, 1, 128, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        reshape_default: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_71, _shape_param_0);  mm_71 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        reshape_default_1: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_default: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        reshape_default_2: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_46, _shape_param_2);  bmm_46 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_tensor: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_331, 2, 0, 128);  primals_331 = None
        slice_tensor_1: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_tensor, 3, 0, 128);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_self: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_tensor_1, reshape_default_2, full_default_2);  slice_tensor_1 = reshape_default_2 = full_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_tensor: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_self, where);  where_self = where = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_default: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_tensor, [-1], True)
        sub_tensor: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  add_tensor = amax_default = None
        exp_default: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_default: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_tensor, _shape_param_3);  div_tensor = _shape_param_3 = None
        reshape_default_3: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_default, _shape_param_4);  expand_default = _shape_param_4 = None
        expand_default_1: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_default, _shape_param_5);  permute_default = _shape_param_5 = None
        clone_default: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_4: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_default, _shape_param_6);  clone_default = _shape_param_6 = None
        return (reshape_default_3, reshape_default_4)



def make_inputs():
    return [
    torch.randn([4096, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [1, 1, 2048, 2048], dtype=torch.bool, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1, 128, 128], dtype=torch.float32, device='cuda'),
    [32, 128, 2048],  # _shape_param_0
    [32, 128, 16, 128],  # _shape_param_1
    [32, 16, 128, 128],  # _shape_param_2
    [32, 16, 128, 128],  # _shape_param_3
    [512, 128, 128],  # _shape_param_4
    [32, 16, 128, 128],  # _shape_param_5
    [512, 128, 128],  # _shape_param_6
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
