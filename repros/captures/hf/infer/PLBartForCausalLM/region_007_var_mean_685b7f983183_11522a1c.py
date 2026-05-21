"""
Standalone repro captured via capture_hook.
Label: hf_PLBartForCausalLM_infer
Pattern hash: 685b7f983183
Shape hash: 11522a1c
"""
_shapes_config = "(T([50005, 768], f32), T([16, 1024], i64), T([1026, 768], f32), T([768], f32), T([768], f32), T([768, 768], f32), T([768, 768], f32), T([768, 768], f32), S([16384, 768]), S([16384, 768]), S([16384, 768]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[50005, 768]", arg0_1: "i64[16, 1024]", arg2_1: "f32[1026, 768]", arg3_1: "f32[768]", arg4_1: "f32[768]", arg5_1: "f32[768, 768]", arg7_1: "f32[768, 768]", arg9_1: "f32[768, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:71 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding_default: "f32[16, 1024, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 1);  arg1_1 = arg0_1 = None
        mul_tensor: "f32[16, 1024, 768]" = torch.ops.aten.mul.Tensor(embedding_default, 27.712812921102035);  embedding_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:553 in forward, code: position_ids = torch.arange(seq_length, device=inputs_embeds.device) + past_key_values_length
        iota_default: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[1024]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:112 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze_default: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:114 in forward, code: return super().forward(position_ids + self.offset)
        add_tensor_1: "i64[1, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_default, 2);  unsqueeze_default = None
        embedding_default_1: "f32[1, 1024, 768]" = torch.ops.aten.embedding.default(arg2_1, add_tensor_1);  arg2_1 = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:583 in forward, code: hidden_states = inputs_embeds + positions
        add_tensor_2: "f32[16, 1024, 768]" = torch.ops.aten.add.Tensor(mul_tensor, embedding_default_1);  mul_tensor = embedding_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:584 in forward, code: hidden_states = self.layernorm_embedding(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_2, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 1024, 1]" = var_mean_correction[0]
        getitem_1: "f32[16, 1024, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[16, 1024, 768]" = torch.ops.aten.sub.Tensor(add_tensor_2, getitem_1);  add_tensor_2 = getitem_1 = None
        add_tensor_3: "f32[16, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[16, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor_3);  add_tensor_3 = None
        mul_tensor_1: "f32[16, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_2: "f32[16, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg3_1);  mul_tensor_1 = arg3_1 = None
        add_tensor_4: "f32[16, 1024, 768]" = torch.ops.aten.add.Tensor(mul_tensor_2, arg4_1);  mul_tensor_2 = arg4_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:209 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_tensor_4, _shape_param_0);  _shape_param_0 = None
        permute_default: "f32[768, 768]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:229 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_1: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_tensor_4, _shape_param_1);  _shape_param_1 = None
        permute_default_1: "f32[768, 768]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:230 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_2: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_tensor_4, _shape_param_2);  add_tensor_4 = _shape_param_2 = None
        permute_default_2: "f32[768, 768]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        return (reshape_default, permute_default, reshape_default_1, permute_default_1, reshape_default_2, permute_default_2)



def make_inputs():
    return [
    torch.randn([50005, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 50005, [16, 1024], dtype=torch.int64, device='cuda'),
    torch.randn([1026, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [16384, 768],  # _shape_param_0
    [16384, 768],  # _shape_param_1
    [16384, 768],  # _shape_param_2
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
