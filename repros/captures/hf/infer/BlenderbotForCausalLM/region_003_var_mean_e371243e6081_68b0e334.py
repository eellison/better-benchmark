"""
Standalone repro captured via capture_hook.
Label: hf_BlenderbotForCausalLM_infer
Pattern hash: e371243e6081
Shape hash: 68b0e334
"""
_shapes_config = "(T([4096, 2560], f32), T([32, 128, 2560], f32), T([2560], f32), T([2560], f32), T([10240, 2560], f32), S([32, 128, 2560]), S([4096, 2560]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_141: "f32[4096, 2560]", add_166: "f32[32, 128, 2560]", arg381_1: "f32[2560]", arg382_1: "f32[2560]", arg383_1: "f32[10240, 2560]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default: "f32[32, 128, 2560]" = torch.ops.aten.reshape.default(addmm_141, _shape_param_0);  addmm_141 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "f32[32, 128, 2560]" = torch.ops.aten.add.Tensor(add_166, reshape_default);  add_166 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[32, 128, 2560]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        add_tensor_1: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_tensor, arg381_1);  mul_tensor = arg381_1 = None
        add_tensor_2: "f32[32, 128, 2560]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg382_1);  mul_tensor_1 = arg382_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        reshape_default_1: "f32[4096, 2560]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        permute_default: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg383_1, [1, 0]);  arg383_1 = None
        return (reshape_default_1, permute_default)



def make_inputs():
    return [
    torch.randn([4096, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([2560], dtype=torch.float32, device='cuda'),
    torch.randn([2560], dtype=torch.float32, device='cuda'),
    torch.randn([10240, 2560], dtype=torch.float32, device='cuda'),
    [32, 128, 2560],  # _shape_param_0
    [4096, 2560],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
