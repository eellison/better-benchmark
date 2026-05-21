"""
Standalone repro captured via capture_hook.
Label: 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_infer
Pattern hash: 9cd6a91db116
Shape hash: a71da292
"""
_shapes_config = "(T([2048, 2560], f32), T([16, 128, 2560], f32), T([2560], f32), T([2560], f32), T([2560, 2560], f32), T([2560, 2560], f32), T([2560, 2560], f32), S([16, 128, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_241: "f32[2048, 2560]", add_275: "f32[16, 128, 2560]", arg637_1: "f32[2560]", arg638_1: "f32[2560]", arg639_1: "f32[2560, 2560]", arg641_1: "f32[2560, 2560]", arg643_1: "f32[2560, 2560]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        reshape_default: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_241, _shape_param_0);  addmm_241 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_275, reshape_default);  add_275 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[16, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        add_tensor_1: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_tensor, arg637_1);  mul_tensor = arg637_1 = None
        add_tensor_2: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg638_1);  mul_tensor_1 = arg638_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_1: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  _shape_param_1 = None
        permute_default: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg639_1, [1, 0]);  arg639_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_2: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_2);  _shape_param_2 = None
        permute_default_1: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg641_1, [1, 0]);  arg641_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_3: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_3);  add_tensor_2 = _shape_param_3 = None
        permute_default_2: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg643_1, [1, 0]);  arg643_1 = None
        return (reshape_default_1, permute_default, reshape_default_2, permute_default_1, reshape_default_3, permute_default_2)



def make_inputs():
    return [
    torch.randn([2048, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([16, 128, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([2560], dtype=torch.float32, device='cuda'),
    torch.randn([2560], dtype=torch.float32, device='cuda'),
    torch.randn([2560, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([2560, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([2560, 2560], dtype=torch.float32, device='cuda'),
    [16, 128, 2560],  # _shape_param_0
    [2048, 2560],  # _shape_param_1
    [2048, 2560],  # _shape_param_2
    [2048, 2560],  # _shape_param_3
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
