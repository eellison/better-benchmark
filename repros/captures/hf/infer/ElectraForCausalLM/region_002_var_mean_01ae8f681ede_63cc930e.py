"""
Standalone repro captured via capture_hook.
Label: hf_ElectraForCausalLM_infer
Pattern hash: 01ae8f681ede
Shape hash: 63cc930e
"""
_shapes_config = "(T([32768, 256], f32), T([64, 512, 256], f32), T([256], f32), T([256], f32), T([128, 256], f32), S([64, 512, 256]), S([32768, 256]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_72: "f32[32768, 256]", add_97: "f32[64, 512, 256]", arg201_1: "f32[256]", arg202_1: "f32[256]", arg203_1: "f32[128, 256]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_72, _shape_param_0);  addmm_72 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_tensor: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(reshape_default, add_97);  reshape_default = add_97 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[64, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[64, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        add_tensor_1: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_tensor, arg201_1);  mul_tensor = arg201_1 = None
        add_tensor_2: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg202_1);  mul_tensor_1 = arg202_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:499 in forward, code: hidden_states = self.dense(generator_hidden_states)
        reshape_default_1: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        permute_default: "f32[256, 128]" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
        return (reshape_default_1, permute_default)



def make_inputs():
    return [
    torch.randn([32768, 256], dtype=torch.float32, device='cuda'),
    torch.randn([64, 512, 256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([128, 256], dtype=torch.float32, device='cuda'),
    [64, 512, 256],  # _shape_param_0
    [32768, 256],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
