"""
Standalone repro captured via capture_hook.
Label: hf_T5Small_infer
Pattern hash: 763aa63c6b58
Shape hash: 28022a6d
"""
_shapes_config = "(T([64, 1024, 64], f32), T([512, 512], f32), S([8, 8, 1024, 64]), S([8, 1024, -1]), S([8192, 512]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, bmm_default_1: "f32[64, 1024, 64]", arg123_1: "f32[512, 512]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_default_1, _shape_param_0);  bmm_default_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        reshape_default_1: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        reshape_default_2: "f32[8192, 512]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_1: "f32[512, 512]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        return (reshape_default_2, permute_default_1)



def make_inputs():
    return [
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    [8, 8, 1024, 64],  # _shape_param_0
    [8, 1024, -1],  # _shape_param_1
    [8192, 512],  # _shape_param_2
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
