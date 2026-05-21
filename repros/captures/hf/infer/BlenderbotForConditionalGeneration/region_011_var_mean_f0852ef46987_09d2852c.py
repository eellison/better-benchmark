"""
Standalone repro captured via capture_hook.
Label: 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_infer
Pattern hash: f0852ef46987
Shape hash: 09d2852c
"""
_shapes_config = "(T([2048, 2560], f32), T([16, 128, 2560], f32), T([2048, 2560], f32), T([16, 128, 2560], f32), T([2560], f32), T([2560], f32), T([2560, 2560], f32), T([2560], f32), T([2560], f32), T([2560, 2560], f32), S([16, 128, 2560]), S([16, 128, 2560]), S([2048, 2560]), S([2048, 2560]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_11: "f32[2048, 2560]", add_14: "f32[16, 128, 2560]", addmm_15: "f32[2048, 2560]", add_26: "f32[16, 128, 2560]", arg49_1: "f32[2560]", arg50_1: "f32[2560]", arg51_1: "f32[2560, 2560]", arg36_1: "f32[2560]", arg37_1: "f32[2560]", arg53_1: "f32[2560, 2560]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:292 in forward, code: hidden_states = self.fc2(hidden_states)
        reshape_default: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_11, _shape_param_0);  addmm_11 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:294 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_14, reshape_default);  add_14 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:508 in forward, code: hidden_states = self.layer_norm(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[16, 128, 1]" = var_mean_correction[1];  var_mean_correction = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default_1: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_15, _shape_param_1);  addmm_15 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_tensor_1: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_26, reshape_default_1);  add_26 = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True)
        getitem_2: "f32[16, 128, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[16, 128, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        sub_tensor: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_tensor_1, getitem_3);  add_tensor_1 = getitem_3 = None
        add_tensor_2: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_default: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_tensor, arg49_1);  mul_tensor = arg49_1 = None
        add_tensor_3: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg50_1);  mul_tensor_1 = arg50_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_2: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_3, _shape_param_2);  add_tensor_3 = _shape_param_2 = None
        permute_default: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:508 in forward, code: hidden_states = self.layer_norm(hidden_states)
        sub_tensor_1: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        add_tensor_4: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default_1: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        mul_tensor_2: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        mul_tensor_3: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_tensor_2, arg36_1);  mul_tensor_2 = arg36_1 = None
        add_tensor_5: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_tensor_3, arg37_1);  mul_tensor_3 = arg37_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_3: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_5, _shape_param_3);  add_tensor_5 = _shape_param_3 = None
        permute_default_1: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        return (reshape_default_2, permute_default, reshape_default_3, permute_default_1)



def make_inputs():
    return [
    torch.randn([2048, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([16, 128, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([16, 128, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([2560], dtype=torch.float32, device='cuda'),
    torch.randn([2560], dtype=torch.float32, device='cuda'),
    torch.randn([2560, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([2560], dtype=torch.float32, device='cuda'),
    torch.randn([2560], dtype=torch.float32, device='cuda'),
    torch.randn([2560, 2560], dtype=torch.float32, device='cuda'),
    [16, 128, 2560],  # _shape_param_0
    [16, 128, 2560],  # _shape_param_1
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
