"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_train
Pattern hash: 7a593b2ec8a4
Shape hash: e169a598
"""
_shapes_config = "(T([4096, 384], f32), T([4096, 384], f32), T([384, 512], f32), S([32, 128, 384]), S([32, 128, -1, 64]), S([32, 128, 384]), S([32, 128, -1, 64]), S([32, 6, 128, 64]), S([192, 128, 64]), S([32, 6, 64, 128]), S([192, 64, 128]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_133: "f32[4096, 384]", mm_134: "f32[4096, 384]", primals_180: "f32[384, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_133, _shape_param_0);  mm_133 = _shape_param_0 = None
        reshape_default_1: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_2: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_134, _shape_param_2);  mm_134 = _shape_param_2 = None
        reshape_default_3: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_1: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_2: "f32[512, 384]" = torch.ops.aten.permute.default(primals_180, [1, 0]);  primals_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_3: "f32[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_default_1, [0, 1, 3, 2]);  permute_default_1 = None
        expand_default: "f32[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_default, _shape_param_4);  permute_default = _shape_param_4 = None
        clone_default: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_4: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_5);  clone_default = _shape_param_5 = None
        expand_default_1: "f32[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_default_3, _shape_param_6);  permute_default_3 = _shape_param_6 = None
        clone_default_1: "f32[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_5: "f32[192, 64, 128]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_7);  clone_default_1 = _shape_param_7 = None
        return (permute_default_2, reshape_default_4, reshape_default_5)



def make_inputs():
    return [
    torch.randn([4096, 384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512], dtype=torch.float32, device='cuda'),
    [32, 128, 384],  # _shape_param_0
    [32, 128, -1, 64],  # _shape_param_1
    [32, 128, 384],  # _shape_param_2
    [32, 128, -1, 64],  # _shape_param_3
    [32, 6, 128, 64],  # _shape_param_4
    [192, 128, 64],  # _shape_param_5
    [32, 6, 64, 128],  # _shape_param_6
    [192, 64, 128],  # _shape_param_7
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
