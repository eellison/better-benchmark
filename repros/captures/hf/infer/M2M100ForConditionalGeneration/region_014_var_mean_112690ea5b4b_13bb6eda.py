"""
Standalone repro captured via capture_hook.
Label: hf_M2M100ForConditionalGeneration_infer
Pattern hash: 112690ea5b4b
Shape hash: 13bb6eda
"""
_shapes_config = "(T([1024, 128, 64], f32), T([1024, 1024], f32), T([128112, 1024], f32), T([64, 128], i64), T([64, 128], i64), T([64, 128], i32), T([1026, 1024], f32), T([1024], f32), T([1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), S([64, 16, 128, 64]), S([64, 128, -1]), S([8192, 1024]), S([64, 128, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, bmm_23: "f32[1024, 128, 64]", arg188_1: "f32[1024, 1024]", arg2_1: "f32[128112, 1024]", arg1_1: "i64[64, 128]", cumsum_1: "i64[64, 128]", convert_element_type_3: "i32[64, 128]", arg198_1: "f32[1026, 1024]", arg199_1: "f32[1024]", arg200_1: "f32[1024]", arg201_1: "f32[1024, 1024]", arg203_1: "f32[1024, 1024]", arg205_1: "f32[1024, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default: "f32[64, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_23, _shape_param_0);  bmm_23 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default: "f32[64, 128, 16, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[64, 128, 16, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_1: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default_2: "f32[8192, 1024]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:77 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding_default: "f32[64, 128, 1024]" = torch.ops.aten.embedding.default(arg2_1, arg1_1, 1);  arg2_1 = arg1_1 = None
        mul_tensor: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(embedding_default, 32.0);  embedding_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:178 in create_position_ids_from_input_ids, code: incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        convert_element_type_default: "i32[64, 128]" = torch.ops.prims.convert_element_type.default(cumsum_1, torch.int32);  cumsum_1 = None
        add_tensor: "i32[64, 128]" = torch.ops.aten.add.Tensor(convert_element_type_default, 0);  convert_element_type_default = None
        mul_tensor_1: "i32[64, 128]" = torch.ops.aten.mul.Tensor(add_tensor, convert_element_type_3);  add_tensor = convert_element_type_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:179 in create_position_ids_from_input_ids, code: return incremental_indices.long() + padding_idx
        convert_element_type_default_1: "i64[64, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.int64);  mul_tensor_1 = None
        add_tensor_1: "i64[64, 128]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1);  convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:144 in forward, code: return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()
        reshape_default_3: "i64[8192]" = torch.ops.aten.reshape.default(add_tensor_1, [-1]);  add_tensor_1 = None
        index_tensor: "f32[8192, 1024]" = torch.ops.aten.index.Tensor(arg198_1, [reshape_default_3]);  arg198_1 = reshape_default_3 = None
        reshape_default_4: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(index_tensor, _shape_param_3);  index_tensor = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:687 in forward, code: hidden_states = inputs_embeds + positions
        add_tensor_2: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(mul_tensor, reshape_default_4);  mul_tensor = reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:441 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_2, [2], correction = 0, keepdim = True)
        getitem: "f32[64, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[64, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(add_tensor_2, getitem_1);  add_tensor_2 = getitem_1 = None
        add_tensor_3: "f32[64, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[64, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_3);  add_tensor_3 = None
        mul_tensor_2: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_3: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_2, arg199_1);  mul_tensor_2 = arg199_1 = None
        add_tensor_4: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_3, arg200_1);  mul_tensor_3 = arg200_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_5: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_4, _shape_param_4);  _shape_param_4 = None
        permute_default_2: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_6: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_4, _shape_param_5);  _shape_param_5 = None
        permute_default_3: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_7: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_4, _shape_param_6);  add_tensor_4 = _shape_param_6 = None
        permute_default_4: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
        return (reshape_default_2, permute_default_1, reshape_default_5, permute_default_2, reshape_default_6, permute_default_3, reshape_default_7, permute_default_4)



def make_inputs():
    return [
    torch.randn([1024, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([128112, 1024], dtype=torch.float32, device='cuda'),
    torch.randint(0, 128112, [64, 128], dtype=torch.int64, device='cuda'),
    torch.randint(0, 128, [64, 128], dtype=torch.int64, device='cuda'),
    torch.randint(0, 128, [64, 128], dtype=torch.int32, device='cuda'),
    torch.randn([1026, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    [64, 16, 128, 64],  # _shape_param_0
    [64, 128, -1],  # _shape_param_1
    [8192, 1024],  # _shape_param_2
    [64, 128, 1024],  # _shape_param_3
    [8192, 1024],  # _shape_param_4
    [8192, 1024],  # _shape_param_5
    [8192, 1024],  # _shape_param_6
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
