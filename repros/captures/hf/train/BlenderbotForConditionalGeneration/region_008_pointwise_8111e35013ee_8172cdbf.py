"""
Standalone repro captured via capture_hook.
Label: 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_train
Pattern hash: 8111e35013ee
Shape hash: 8172cdbf
"""
_shapes_config = "(T([2048, 2560], f32), T([2048, 2560], f32), T([2560, 2560], f32), S([16, 128, 2560]), S([16, 128, -1, 80]), S([16, 128, 2560]), S([16, 128, -1, 80]), S([16, 32, 128, 80]), S([512, 128, 80]), S([16, 32, 80, 128]), S([512, 80, 128]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm: "f32[2048, 2560]", addmm_1: "f32[2048, 2560]", primals_8: "f32[2560, 2560]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        reshape_default_1: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_2: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_1, _shape_param_2);  addmm_1 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        permute_default_1: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        reshape_default_3: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_2: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_scalar: "f32[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_default, 0.334370152488211);  permute_default = None
        permute_default_3: "f32[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_default_2, [0, 1, 3, 2]);  permute_default_2 = None
        mul_scalar_1: "f32[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_default_3, 0.334370152488211);  permute_default_3 = None
        expand_default: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_scalar, _shape_param_4);  mul_scalar = _shape_param_4 = None
        clone_default: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_4: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_default, _shape_param_5);  clone_default = _shape_param_5 = None
        expand_default_1: "f32[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_scalar_1, _shape_param_6);  mul_scalar_1 = _shape_param_6 = None
        clone_default_1: "f32[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_5: "f32[512, 80, 128]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_7);  clone_default_1 = _shape_param_7 = None
        return (permute_default_1, reshape_default_4, reshape_default_5)



def make_inputs():
    return [
    torch.randn([2048, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([2560, 2560], dtype=torch.float32, device='cuda'),
    [16, 128, 2560],  # _shape_param_0
    [16, 128, -1, 80],  # _shape_param_1
    [16, 128, 2560],  # _shape_param_2
    [16, 128, -1, 80],  # _shape_param_3
    [16, 32, 128, 80],  # _shape_param_4
    [512, 128, 80],  # _shape_param_5
    [16, 32, 80, 128],  # _shape_param_6
    [512, 80, 128],  # _shape_param_7
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
