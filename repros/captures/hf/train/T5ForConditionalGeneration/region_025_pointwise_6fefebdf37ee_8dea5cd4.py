"""
Standalone repro captured via capture_hook.
Label: hf_T5ForConditionalGeneration_train
Pattern hash: 6fefebdf37ee
Shape hash: 8dea5cd4
"""
_shapes_config = "(T([64, 1024, 64], f32), T([64, 64, 1024], f32), T([64, 1024, 64], f32), T([512, 512], f32), T([512, 512], f32), T([512, 512], f32), S([8, 8, 1024, 64]), S([8, 8, 64, 1024]), S([8, 8, 1024, 64]), S([8, 1024, 512]), S([8192, 512]), S([8, 1024, 512]), S([8192, 512]), S([8, 1024, 512]), S([8192, 512]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, bmm_100: "f32[64, 1024, 64]", bmm_102: "f32[64, 64, 1024]", bmm_103: "f32[64, 1024, 64]", primals_15: "f32[512, 512]", primals_14: "f32[512, 512]", primals_13: "f32[512, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_100, _shape_param_0);  bmm_100 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_1: "f32[8, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_102, _shape_param_1);  bmm_102 = _shape_param_1 = None
        reshape_default_2: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_103, _shape_param_2);  bmm_103 = _shape_param_2 = None
        permute_default: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_1: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_3: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        reshape_default_4: "f32[8192, 512]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_4);  reshape_default_3 = _shape_param_4 = None
        permute_default_2: "f32[512, 512]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        permute_default_3: "f32[512, 512]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_4: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        reshape_default_5: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(permute_default_4, _shape_param_5);  permute_default_4 = _shape_param_5 = None
        clone_default_1: "f32[8, 1024, 512]" = torch.ops.aten.clone.default(reshape_default_5, memory_format = torch.contiguous_format);  reshape_default_5 = None
        reshape_default_6: "f32[8192, 512]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_6);  clone_default_1 = _shape_param_6 = None
        permute_default_5: "f32[512, 512]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
        permute_default_6: "f32[512, 512]" = torch.ops.aten.permute.default(permute_default_5, [1, 0]);  permute_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_7: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None
        clone_default_2: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_7, memory_format = torch.contiguous_format);  permute_default_7 = None
        reshape_default_7: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_7);  clone_default_2 = _shape_param_7 = None
        reshape_default_8: "f32[8192, 512]" = torch.ops.aten.reshape.default(reshape_default_7, _shape_param_8);  reshape_default_7 = _shape_param_8 = None
        permute_default_8: "f32[512, 512]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        permute_default_9: "f32[512, 512]" = torch.ops.aten.permute.default(permute_default_8, [1, 0]);  permute_default_8 = None
        return (reshape_default_4, permute_default_3, reshape_default_6, permute_default_6, reshape_default_8, permute_default_9)



def make_inputs():
    return [
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    [8, 8, 1024, 64],  # _shape_param_0
    [8, 8, 64, 1024],  # _shape_param_1
    [8, 8, 1024, 64],  # _shape_param_2
    [8, 1024, 512],  # _shape_param_3
    [8192, 512],  # _shape_param_4
    [8, 1024, 512],  # _shape_param_5
    [8192, 512],  # _shape_param_6
    [8, 1024, 512],  # _shape_param_7
    [8192, 512],  # _shape_param_8
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
