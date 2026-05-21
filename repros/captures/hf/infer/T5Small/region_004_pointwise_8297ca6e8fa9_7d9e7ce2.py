"""
Standalone repro captured via capture_hook.
Label: hf_T5Small_infer
Pattern hash: 8297ca6e8fa9
Shape hash: 7d9e7ce2
"""
_shapes_config = "(T([8, 8, 1024, 64], f32, stride=(524288, 64, 512, 1)), T([512, 512], f32), S([8, 1024, -1]), S([8192, 512]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem: "f32[8, 8, 1024, 64]", arg128_1: "f32[512, 512]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None
        clone_default: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        reshape_default: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        reshape_default_1: "f32[8192, 512]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default_1: "f32[512, 512]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        return (reshape_default_1, permute_default_1)



def make_inputs():
    return [
    torch.randn(4194304, dtype=torch.float32, device='cuda').as_strided([8, 8, 1024, 64], [524288, 64, 512, 1]),  # getitem
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    [8, 1024, -1],  # _shape_param_0
    [8192, 512],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
