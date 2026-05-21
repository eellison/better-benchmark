"""
Standalone repro captured via capture_hook.
Label: hf_MBartForCausalLM_infer
Pattern hash: 9cd6a91db116
Shape hash: 7fa8ae7d
"""
_shapes_config = "(T([8192, 1024], f32), T([8, 1024, 1024], f32), T([1024], f32), T([1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), S([8, 1024, 1024]), S([8192, 1024]), S([8192, 1024]), S([8192, 1024]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_65: "f32[8192, 1024]", add_81: "f32[8, 1024, 1024]", arg181_1: "f32[1024]", arg182_1: "f32[1024]", arg183_1: "f32[1024, 1024]", arg185_1: "f32[1024, 1024]", arg187_1: "f32[1024, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:415 in forward, code: hidden_states = self.fc2(hidden_states)
        reshape_default: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_65, _shape_param_0);  addmm_65 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:417 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_81, reshape_default);  add_81 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:383 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1024, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 1024, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        add_tensor_1: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, arg181_1);  mul_tensor = arg181_1 = None
        add_tensor_2: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg182_1);  mul_tensor_1 = arg182_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:220 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_1: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  _shape_param_1 = None
        permute_default: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:240 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_2: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_2);  _shape_param_2 = None
        permute_default_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py:241 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_3: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_3);  add_tensor_2 = _shape_param_3 = None
        permute_default_2: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        return (reshape_default_1, permute_default, reshape_default_2, permute_default_1, reshape_default_3, permute_default_2)



def make_inputs():
    return [
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    [8, 1024, 1024],  # _shape_param_0
    [8192, 1024],  # _shape_param_1
    [8192, 1024],  # _shape_param_2
    [8192, 1024],  # _shape_param_3
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
