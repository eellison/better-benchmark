"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train
Pattern hash: a2bdd26a0a31
Shape hash: 78eab4c7
"""
_shapes_config = "(T([32768, 128], f32), T([256, 128, 128], f32), T([128], f32), T([128], f32), T([512, 128], f32), S([256, 128, 128]), S([32768, 128]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_359: "f32[32768, 128]", add_360: "f32[256, 128, 128]", primals_1108: "f32[128]", primals_1109: "f32[128]", primals_1110: "f32[512, 128]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        reshape_default: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_359, _shape_param_0);  addmm_359 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_tensor: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(reshape_default, add_360);  reshape_default = add_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_tensor: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_tensor, primals_1108);  add_tensor = primals_1108 = None
        add_tensor_1: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_tensor, primals_1109);  mul_tensor = primals_1109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        reshape_default_1: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_1);  add_tensor_1 = _shape_param_1 = None
        permute_default: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1110, [1, 0]);  primals_1110 = None
        return (reshape_default_1, permute_default)



def make_inputs():
    return [
    torch.randn([32768, 128], dtype=torch.float32, device='cuda'),
    torch.randn([256, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([512, 128], dtype=torch.float32, device='cuda'),
    [256, 128, 128],  # _shape_param_0
    [32768, 128],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
