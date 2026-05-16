"""
Standalone repro captured via capture_hook.
Label: t5_base
Pattern hash: a8e741d0b164
Shape hash: 0195c3bb
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_190: "f16[512, 3072]", _shape_param_0, _shape_param_1, arg258_1: "f32[768, 3072]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        reshape_default: "f16[4, 128, 3072]" = torch.ops.aten.reshape.default(mm_190, _shape_param_0);  mm_190 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_default: "f16[4, 128, 3072]" = torch.ops.aten.relu.default(reshape_default);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:88 in forward, code: hidden_states = hidden_states.to(self.wo.weight.dtype)
        convert_element_type_default: "f32[4, 128, 3072]" = torch.ops.prims.convert_element_type.default(relu_default, torch.float32);  relu_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        reshape_default_1: "f32[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_default, _shape_param_1);  convert_element_type_default = _shape_param_1 = None
        permute_default: "f32[3072, 768]" = torch.ops.aten.permute.default(arg258_1, [1, 0]);  arg258_1 = None
        return (reshape_default_1, permute_default)



def make_inputs():
    return [
    torch.randn([512, 3072], dtype=torch.float16, device='cuda'),
    [4, 128, 3072],  # _shape_param_0
    [512, 3072],  # _shape_param_1
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
