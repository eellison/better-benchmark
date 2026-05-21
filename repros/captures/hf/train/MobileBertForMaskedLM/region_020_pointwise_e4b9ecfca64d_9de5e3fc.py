"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train
Pattern hash: e4b9ecfca64d
Shape hash: 9de5e3fc
"""
_shapes_config = "(T([32768, 128], f32), T([256, 128, 128], f32), T([128], f32), T([128, 128], f32), S([256, 128, 128]), S([32768, 128]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_711: "f32[32768, 128]", mul_792: "f32[256, 128, 128]", primals_26: "f32[128]", primals_24: "f32[128, 128]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(mm_711, _shape_param_0);  mm_711 = _shape_param_0 = None
        add_tensor: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_792, reshape_default);  mul_792 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_tensor: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_tensor, primals_26);  add_tensor = primals_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        reshape_default_1: "f32[32768, 128]" = torch.ops.aten.reshape.default(mul_tensor, _shape_param_1);  mul_tensor = _shape_param_1 = None
        permute_default: "f32[128, 128]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
        permute_default_1: "f32[128, 128]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_1, permute_default_1)



def make_inputs():
    return [
    torch.randn([32768, 128], dtype=torch.float32, device='cuda'),
    torch.randn([256, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128, 128], dtype=torch.float32, device='cuda'),
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
