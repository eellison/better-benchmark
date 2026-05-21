"""
Standalone repro captured via capture_hook.
Label: hf_MBartForCausalLM_infer
Pattern hash: 7b0960c6db08
Shape hash: 0e7063e3
"""
_shapes_config = "(T([50265, 1024], f32), T([8, 1024], i64), T([1026, 1024], f32), T([1024], f32), T([1024], f32), T([1024], f32), T([1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[50265, 1024]", arg0_1: "i64[8, 1024]", arg2_1: "f32[1026, 1024]", arg3_1: "f32[1024]", arg4_1: "f32[1024]", arg5_1: "f32[1024]", arg6_1: "f32[1024]", arg7_1: "f32[1024, 1024]", arg9_1: "f32[1024, 1024]", arg11_1: "f32[1024, 1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:123 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding_default: "f32[8, 1024, 1024]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 1);  arg1_1 = arg0_1 = None
        mul_tensor: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(embedding_default, 1.0);  embedding_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:703 in forward, code: position_ids = torch.arange(seq_length, device=inputs_embeds.device) + past_key_values_length
        iota_default: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[1024]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:107 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze_default: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:109 in forward, code: return super().forward(position_ids + self.offset)
        add_tensor_1: "i64[1, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_default, 2);  unsqueeze_default = None
        embedding_default_1: "f32[1, 1024, 1024]" = torch.ops.aten.embedding.default(arg2_1, add_tensor_1);  arg2_1 = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:732 in forward, code: hidden_states = inputs_embeds + position_ids.to(inputs_embeds.device)
        add_tensor_2: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_tensor, embedding_default_1);  mul_tensor = embedding_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:733 in forward, code: hidden_states = self.layernorm_embedding(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_2, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1024, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 1024, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_tensor_2, getitem_1);  add_tensor_2 = getitem_1 = None
        add_tensor_3: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor_3);  add_tensor_3 = None
        mul_tensor_1: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_2: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg3_1);  mul_tensor_1 = arg3_1 = None
        add_tensor_4: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_2, arg4_1);  mul_tensor_2 = arg4_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:383 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(add_tensor_4, [2], correction = 0, keepdim = True)
        getitem_2: "f32[8, 1024, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[8, 1024, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        sub_tensor_1: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_tensor_4, getitem_3);  add_tensor_4 = getitem_3 = None
        add_tensor_5: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_default_1: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor_5);  add_tensor_5 = None
        mul_tensor_3: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        mul_tensor_4: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_3, arg5_1);  mul_tensor_3 = arg5_1 = None
        add_tensor_6: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_4, arg6_1);  mul_tensor_4 = arg6_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:220 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_6, _shape_param_0);  _shape_param_0 = None
        permute_default: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:240 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_1: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_6, _shape_param_1);  _shape_param_1 = None
        permute_default_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:241 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_2: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_6, _shape_param_2);  add_tensor_6 = _shape_param_2 = None
        permute_default_2: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        return (reshape_default, permute_default, reshape_default_1, permute_default_1, reshape_default_2, permute_default_2)



def make_inputs():
    return [
    torch.randn([50265, 1024], dtype=torch.float32, device='cuda'),
    torch.randint(0, 50265, [8, 1024], dtype=torch.int64, device='cuda'),
    torch.randn([1026, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    [8192, 1024],  # _shape_param_0
    [8192, 1024],  # _shape_param_1
    [8192, 1024],  # _shape_param_2
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
