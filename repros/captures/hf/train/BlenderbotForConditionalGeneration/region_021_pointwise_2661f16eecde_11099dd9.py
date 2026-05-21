"""
Standalone repro captured via capture_hook.
Label: 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_train
Pattern hash: 2661f16eecde
Shape hash: 11099dd9
"""
_shapes_config = "(T([512, 128, 80], f32), T([512, 80, 128], f32), T([2560, 2560], f32), T([2560, 2560], f32), T([16, 32, 128, 80], f32, stride=(327680, 80, 2560, 1)), T([16, 32, 128, 80], f32, stride=(327680, 80, 2560, 1)), T([2560, 2560], f32), T([2560, 2560], f32), T([16, 32, 128, 80], f32, stride=(327680, 80, 2560, 1)), T([2560, 2560], f32), S([16, 32, 128, 80]), S([16, 32, 80, 128]), S([16, 128, 2560]), S([16, 128, 2560]), S([2048, 2560]), S([2048, 2560]), S([16, 128, 2560]), S([16, 128, 2560]), S([2048, 2560]), S([2048, 2560]), S([16, 128, 2560]), S([2048, 2560]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, bmm_2: "f32[512, 128, 80]", bmm_4: "f32[512, 80, 128]", primals_20: "f32[2560, 2560]", primals_18: "f32[2560, 2560]", getitem_12: "f32[16, 32, 128, 80]", getitem_11: "f32[16, 32, 128, 80]", primals_8: "f32[2560, 2560]", primals_6: "f32[2560, 2560]", getitem_10: "f32[16, 32, 128, 80]", primals_4: "f32[2560, 2560]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_2, _shape_param_0);  bmm_2 = _shape_param_0 = None
        reshape_default_1: "f32[16, 32, 80, 128]" = torch.ops.aten.reshape.default(bmm_4, _shape_param_1);  bmm_4 = _shape_param_1 = None
        mul_scalar: "f32[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(reshape_default_1, 0.334370152488211);  reshape_default_1 = None
        permute_default: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(mul_scalar, [0, 1, 3, 2]);  mul_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        permute_default_1: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_2: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        permute_default_2: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        reshape_default_3: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_default_2, _shape_param_3);  permute_default_2 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_4: "f32[2048, 2560]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_4);  reshape_default_2 = _shape_param_4 = None
        permute_default_3: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
        permute_default_4: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_default_3, [1, 0]);  permute_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        clone_default_1: "f32[16, 128, 2560]" = torch.ops.aten.clone.default(reshape_default_3, memory_format = torch.contiguous_format);  reshape_default_3 = None
        reshape_default_5: "f32[2048, 2560]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_5);  clone_default_1 = _shape_param_5 = None
        permute_default_5: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
        permute_default_6: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_default_5, [1, 0]);  permute_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        permute_default_7: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_12, [0, 2, 1, 3]);  getitem_12 = None
        reshape_default_6: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_default_7, _shape_param_6);  permute_default_7 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        permute_default_8: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_11, [0, 2, 1, 3]);  getitem_11 = None
        reshape_default_7: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_default_8, _shape_param_7);  permute_default_8 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_8: "f32[2048, 2560]" = torch.ops.aten.reshape.default(reshape_default_6, _shape_param_8);  reshape_default_6 = _shape_param_8 = None
        permute_default_9: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_default_10: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_default_9, [1, 0]);  permute_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_9: "f32[2048, 2560]" = torch.ops.aten.reshape.default(reshape_default_7, _shape_param_9);  reshape_default_7 = _shape_param_9 = None
        permute_default_11: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_default_12: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_default_11, [1, 0]);  permute_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_13: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_10, [0, 2, 1, 3]);  getitem_10 = None
        reshape_default_10: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_default_13, _shape_param_10);  permute_default_13 = _shape_param_10 = None
        reshape_default_11: "f32[2048, 2560]" = torch.ops.aten.reshape.default(reshape_default_10, _shape_param_11);  reshape_default_10 = _shape_param_11 = None
        permute_default_14: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_default_15: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_default_14, [1, 0]);  permute_default_14 = None
        return (reshape_default_4, permute_default_4, reshape_default_5, permute_default_6, reshape_default_8, permute_default_10, reshape_default_9, permute_default_12, reshape_default_11, permute_default_15)



def make_inputs():
    return [
    torch.randn([512, 128, 80], dtype=torch.float32, device='cuda'),
    torch.randn([512, 80, 128], dtype=torch.float32, device='cuda'),
    torch.randn([2560, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([2560, 2560], dtype=torch.float32, device='cuda'),
    torch.randn(5242880, dtype=torch.float32, device='cuda').as_strided([16, 32, 128, 80], [327680, 80, 2560, 1]),  # getitem_12
    torch.randn(5242880, dtype=torch.float32, device='cuda').as_strided([16, 32, 128, 80], [327680, 80, 2560, 1]),  # getitem_11
    torch.randn([2560, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([2560, 2560], dtype=torch.float32, device='cuda'),
    torch.randn(5242880, dtype=torch.float32, device='cuda').as_strided([16, 32, 128, 80], [327680, 80, 2560, 1]),  # getitem_10
    torch.randn([2560, 2560], dtype=torch.float32, device='cuda'),
    [16, 32, 128, 80],  # _shape_param_0
    [16, 32, 80, 128],  # _shape_param_1
    [16, 128, 2560],  # _shape_param_2
    [16, 128, 2560],  # _shape_param_3
    [2048, 2560],  # _shape_param_4
    [2048, 2560],  # _shape_param_5
    [16, 128, 2560],  # _shape_param_6
    [16, 128, 2560],  # _shape_param_7
    [2048, 2560],  # _shape_param_8
    [2048, 2560],  # _shape_param_9
    [16, 128, 2560],  # _shape_param_10
    [2048, 2560],  # _shape_param_11
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
