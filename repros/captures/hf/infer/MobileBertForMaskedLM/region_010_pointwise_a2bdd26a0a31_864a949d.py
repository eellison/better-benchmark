"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_infer
Pattern hash: a2bdd26a0a31
Shape hash: 864a949d
"""
_shapes_config = "(T([32768, 512], f32), T([256, 128, 512], f32), T([512], f32), T([512], f32), T([128, 512], f32), S([256, 128, 512]), S([32768, 512]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_345: "f32[32768, 512]", add_334: "f32[256, 128, 512]", arg1065_1: "f32[512]", arg1066_1: "f32[512]", arg1071_1: "f32[128, 512]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        reshape_default: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_345, _shape_param_0);  addmm_345 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_tensor: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(reshape_default, add_334);  reshape_default = add_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_tensor: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor, arg1065_1);  add_tensor = arg1065_1 = None
        add_tensor_1: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_tensor, arg1066_1);  mul_tensor = arg1066_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        reshape_default_1: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_1);  add_tensor_1 = _shape_param_1 = None
        permute_default: "f32[512, 128]" = torch.ops.aten.permute.default(arg1071_1, [1, 0]);  arg1071_1 = None
        return (reshape_default_1, permute_default)



def make_inputs():
    return [
    torch.randn([32768, 512], dtype=torch.float32, device='cuda'),
    torch.randn([256, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512], dtype=torch.float32, device='cuda'),
    [256, 128, 512],  # _shape_param_0
    [32768, 512],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
