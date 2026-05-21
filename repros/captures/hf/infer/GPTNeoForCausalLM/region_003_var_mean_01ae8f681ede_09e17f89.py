"""
Standalone repro captured via capture_hook.
Label: hf_GPTNeoForCausalLM_infer
Pattern hash: 01ae8f681ede
Shape hash: 09e17f89
"""
_shapes_config = "(T([4096, 2048], f32), T([32, 128, 2048], f32), T([2048], f32), T([2048], f32), T([8192, 2048], f32), S([32, 128, 2048]), S([4096, 2048]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_69: "f32[4096, 2048]", add_210: "f32[32, 128, 2048]", arg333_1: "f32[2048]", arg334_1: "f32[2048]", arg335_1: "f32[8192, 2048]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(addmm_69, _shape_param_0);  addmm_69 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        add_tensor: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(reshape_default, add_210);  reshape_default = add_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        add_tensor_1: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_tensor, arg333_1);  mul_tensor = arg333_1 = None
        add_tensor_2: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg334_1);  mul_tensor_1 = arg334_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        reshape_default_1: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        permute_default: "f32[2048, 8192]" = torch.ops.aten.permute.default(arg335_1, [1, 0]);  arg335_1 = None
        return (reshape_default_1, permute_default)



def make_inputs():
    return [
    torch.randn([4096, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 2048], dtype=torch.float32, device='cuda'),
    [32, 128, 2048],  # _shape_param_0
    [4096, 2048],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
