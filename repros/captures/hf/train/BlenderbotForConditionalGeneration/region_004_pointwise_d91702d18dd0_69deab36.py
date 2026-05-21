"""
Standalone repro captured via capture_hook.
Label: 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_train
Pattern hash: d91702d18dd0
Shape hash: 69deab36
"""
_shapes_config = "(T([2048, 10240], f32), T([2560, 10240], f32), S([16, 128, 10240]), S([2048, 10240]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_4: "f32[2048, 10240]", primals_17: "f32[2560, 10240]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:290 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        reshape_default: "f32[16, 128, 10240]" = torch.ops.aten.reshape.default(addmm_4, _shape_param_0);  addmm_4 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_tensor: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(reshape_default, 0.5)
        mul_tensor_1: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(reshape_default, 0.7071067811865476);  reshape_default = None
        erf_default: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:292 in forward, code: hidden_states = self.fc2(hidden_states)
        reshape_default_1: "f32[2048, 10240]" = torch.ops.aten.reshape.default(mul_tensor_2, _shape_param_1);  mul_tensor_2 = _shape_param_1 = None
        permute_default: "f32[10240, 2560]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
        return (reshape_default_1, permute_default)



def make_inputs():
    return [
    torch.randn([2048, 10240], dtype=torch.float32, device='cuda'),
    torch.randn([2560, 10240], dtype=torch.float32, device='cuda'),
    [16, 128, 10240],  # _shape_param_0
    [2048, 10240],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
