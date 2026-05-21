"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_train
Pattern hash: 72f44d0fb7a1
Shape hash: 9714e178
"""
_shapes_config = "(T([192, 128, 64], f32), T([384, 512], f32), S([32, 6, 128, 64]), S([32, 128, 384]), S([4096, 384]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, bmm_99: "f32[192, 128, 64]", primals_99: "f32[384, 512]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_99, _shape_param_0);  bmm_99 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        reshape_default_2: "f32[4096, 384]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_1: "f32[512, 384]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
        permute_default_2: "f32[384, 512]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None
        return (reshape_default_2, permute_default_2)



def make_inputs():
    return [
    torch.randn([192, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512], dtype=torch.float32, device='cuda'),
    [32, 6, 128, 64],  # _shape_param_0
    [32, 128, 384],  # _shape_param_1
    [4096, 384],  # _shape_param_2
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
