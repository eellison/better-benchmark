"""
Standalone repro captured via capture_hook.
Label: hf_T5ForConditionalGeneration_train
Pattern hash: a06452a815ae
Shape hash: db44a8fe
"""
_shapes_config = "(T([8192, 512], f32), T([8192, 512], f32), T([512, 512], f32), T([8, 1024, 512], f32), S([8, 1024, 512]), S([8, 1024, -1, 64]), S([8, 1024, 512]), S([8, 1024, -1, 64]), S([8192, 512]), S([8, 8, 1024, 64]), S([64, 1024, 64]), S([8, 8, 64, 1024]), S([64, 64, 1024]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_90: "f32[8192, 512]", mm_91: "f32[8192, 512]", primals_128: "f32[512, 512]", mul_79: "f32[8, 1024, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_90, _shape_param_0);  mm_90 = _shape_param_0 = None
        reshape_default_1: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_2: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_91, _shape_param_2);  mm_91 = _shape_param_2 = None
        reshape_default_3: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_1: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_2: "f32[512, 512]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
        reshape_default_4: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, _shape_param_4);  mul_79 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_3: "f32[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_default_1, [0, 1, 3, 2]);  permute_default_1 = None
        expand_default: "f32[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_default, _shape_param_5);  permute_default = _shape_param_5 = None
        clone_default: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_5: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_6);  clone_default = _shape_param_6 = None
        expand_default_1: "f32[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_default_3, _shape_param_7);  permute_default_3 = _shape_param_7 = None
        clone_default_1: "f32[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_6: "f32[64, 64, 1024]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_8);  clone_default_1 = _shape_param_8 = None
        return (permute_default_2, reshape_default_4, reshape_default_5, reshape_default_6)



def make_inputs():
    return [
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    [8, 1024, 512],  # _shape_param_0
    [8, 1024, -1, 64],  # _shape_param_1
    [8, 1024, 512],  # _shape_param_2
    [8, 1024, -1, 64],  # _shape_param_3
    [8192, 512],  # _shape_param_4
    [8, 8, 1024, 64],  # _shape_param_5
    [64, 1024, 64],  # _shape_param_6
    [8, 8, 64, 1024],  # _shape_param_7
    [64, 64, 1024],  # _shape_param_8
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
