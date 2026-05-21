"""
Standalone repro captured via capture_hook.
Label: hf_T5ForConditionalGeneration_infer
Pattern hash: 735b724c0f3a
Shape hash: b10d072c
"""
_shapes_config = "(T([8192, 2048], f32), T([512, 2048], f32), S([8, 1024, 2048]), S([8192, 2048]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_94: "f32[8192, 2048]", arg131_1: "f32[512, 2048]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        reshape_default: "f32[8, 1024, 2048]" = torch.ops.aten.reshape.default(mm_94, _shape_param_0);  mm_94 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_default: "f32[8, 1024, 2048]" = torch.ops.aten.relu.default(reshape_default);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        reshape_default_1: "f32[8192, 2048]" = torch.ops.aten.reshape.default(relu_default, _shape_param_1);  relu_default = _shape_param_1 = None
        permute_default: "f32[2048, 512]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        return (reshape_default_1, permute_default)



def make_inputs():
    return [
    torch.randn([8192, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    [8, 1024, 2048],  # _shape_param_0
    [8192, 2048],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
