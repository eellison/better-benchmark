"""
Standalone repro captured via capture_hook.
Label: t5_base
Pattern hash: a8e741d0b164
Shape hash: 4bc0f59c
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_64: "f16[2048, 3072]", _shape_param_0, _shape_param_1, arg91_1: "f32[768, 3072]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        reshape_default: "f16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_64, _shape_param_0);  mm_64 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_default: "f16[4, 512, 3072]" = torch.ops.aten.relu.default(reshape_default);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:88 in forward, code: hidden_states = hidden_states.to(self.wo.weight.dtype)
        convert_element_type_default: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(relu_default, torch.float32);  relu_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        reshape_default_1: "f32[2048, 3072]" = torch.ops.aten.reshape.default(convert_element_type_default, _shape_param_1);  convert_element_type_default = _shape_param_1 = None
        permute_default: "f32[3072, 768]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        return (reshape_default_1, permute_default)



def make_inputs():
    return [
    torch.randn([2048, 3072], dtype=torch.float16, device='cuda'),
    [4, 512, 3072],  # _shape_param_0
    [2048, 3072],  # _shape_param_1
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
