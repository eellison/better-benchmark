"""
Standalone repro captured via capture_hook.
Label: t5_base
Pattern hash: c9af7485968d
Shape hash: 7bad0d4d
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_66: "f16[2048, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, mm_67: "f16[2048, 768]", _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, mul_47: "f16[4, 512, 768]", _shape_param_8, arg95_1: "f16[768, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f16[4, 512, 768]" = torch.ops.aten.reshape.default(mm_66, _shape_param_0);  mm_66 = _shape_param_0 = None
        reshape_default_1: "f16[4, 512, 12, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f16[4, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        expand_default: "f16[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_default, _shape_param_2);  permute_default = _shape_param_2 = None
        clone_default: "f16[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_2: "f16[48, 512, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_3: "f16[4, 512, 768]" = torch.ops.aten.reshape.default(mm_67, _shape_param_4);  mm_67 = _shape_param_4 = None
        reshape_default_4: "f16[4, 512, 12, 64]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_5);  reshape_default_3 = _shape_param_5 = None
        permute_default_1: "f16[4, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_4, [0, 2, 1, 3]);  reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_2: "f16[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_default_1, [0, 1, 3, 2]);  permute_default_1 = None
        expand_default_1: "f16[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_default_2, _shape_param_6);  permute_default_2 = _shape_param_6 = None
        clone_default_1: "f16[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_5: "f16[48, 64, 512]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_7);  clone_default_1 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_6: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_47, _shape_param_8);  mul_47 = _shape_param_8 = None
        permute_default_3: "f16[768, 768]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        return (reshape_default_2, reshape_default_5, reshape_default_6, permute_default_3)



def make_inputs():
    return [
    torch.randn([2048, 768], dtype=torch.float16, device='cuda'),
    [4, 512, 768],  # _shape_param_0
    [4, 512, -1, 64],  # _shape_param_1
    [4, 12, 512, 64],  # _shape_param_2
    [48, 512, 64],  # _shape_param_3
    torch.randn([2048, 768], dtype=torch.float16, device='cuda'),
    [4, 512, 768],  # _shape_param_4
    [4, 512, -1, 64],  # _shape_param_5
    [4, 12, 64, 512],  # _shape_param_6
    [48, 64, 512],  # _shape_param_7
    torch.randn([4, 512, 768], dtype=torch.float16, device='cuda'),
    [2048, 768],  # _shape_param_8
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
