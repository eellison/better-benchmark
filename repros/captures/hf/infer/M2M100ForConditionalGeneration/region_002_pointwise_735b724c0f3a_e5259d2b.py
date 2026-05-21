"""
Standalone repro captured via capture_hook.
Label: hf_M2M100ForConditionalGeneration_infer
Pattern hash: 735b724c0f3a
Shape hash: e5259d2b
"""
_shapes_config = "(T([8192, 4096], f32), T([1024, 4096], f32), S([64, 128, 4096]), S([8192, 4096]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_190: "f32[8192, 4096]", arg509_1: "f32[1024, 4096]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:471 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        reshape_default: "f32[64, 128, 4096]" = torch.ops.aten.reshape.default(addmm_190, _shape_param_0);  addmm_190 = _shape_param_0 = None
        relu_default: "f32[64, 128, 4096]" = torch.ops.aten.relu.default(reshape_default);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:473 in forward, code: hidden_states = self.fc2(hidden_states)
        reshape_default_1: "f32[8192, 4096]" = torch.ops.aten.reshape.default(relu_default, _shape_param_1);  relu_default = _shape_param_1 = None
        permute_default: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg509_1, [1, 0]);  arg509_1 = None
        return (reshape_default_1, permute_default)



def make_inputs():
    return [
    torch.randn([8192, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 4096], dtype=torch.float32, device='cuda'),
    [64, 128, 4096],  # _shape_param_0
    [8192, 4096],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
