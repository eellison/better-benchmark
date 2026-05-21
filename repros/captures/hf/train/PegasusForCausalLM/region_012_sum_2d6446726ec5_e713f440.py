"""
Standalone repro captured via capture_hook.
Label: hf_PegasusForCausalLM_train
Pattern hash: 2d6446726ec5
Shape hash: e713f440
"""
_shapes_config = "(T([16384, 1024], f32), T([1024], f32), T([16384, 1024], f32), T([128, 128, 1024], b8), T([128, 128, 1024], f32), T([128, 128, 1], f32), T([128, 128, 1], f32), T([128, 128, 1024], f32), T([1024, 1024], f32), S([128, 128, 1024]), S([128, 128, 1024]), S([16384, 1024]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_2: "f32[16384, 1024]", primals_13: "f32[1024]", addmm_3: "f32[16384, 1024]", gt: "b8[128, 128, 1024]", primals_3: "f32[128, 128, 1024]", getitem_7: "f32[128, 128, 1]", rsqrt_1: "f32[128, 128, 1]", tangents_1: "f32[128, 128, 1024]", primals_11: "f32[1024, 1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:397 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        reshape_default: "f32[128, 128, 1024]" = torch.ops.aten.reshape.default(mm_2, _shape_param_0);  mm_2 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:396 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        mul_tensor: "f32[128, 128, 1024]" = torch.ops.aten.mul.Tensor(reshape_default, primals_13);  reshape_default = primals_13 = None
        mul_tensor_1: "f32[128, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, 1024)
        sum_dim_int_list: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:250 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default_1: "f32[128, 128, 1024]" = torch.ops.aten.reshape.default(addmm_3, _shape_param_1);  addmm_3 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:376 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        mul_tensor_2: "f32[128, 128, 1024]" = torch.ops.aten.mul.Tensor(gt, reshape_default_1);  reshape_default_1 = None
        mul_tensor_3: "f32[128, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.1111111111111112);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:377 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "f32[128, 128, 1024]" = torch.ops.aten.add.Tensor(primals_3, mul_tensor_3);  primals_3 = mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:396 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        sub_tensor: "f32[128, 128, 1024]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_7);  add_tensor = getitem_7 = None
        mul_tensor_4: "f32[128, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_1);  sub_tensor = None
        mul_tensor_5: "f32[128, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_4);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [2], True);  mul_tensor_5 = None
        mul_tensor_6: "f32[128, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_4, sum_dim_int_list_1);  mul_tensor_4 = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[128, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[128, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_6);  sub_tensor_1 = mul_tensor_6 = None
        div_tensor: "f32[128, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 1024);  rsqrt_1 = None
        mul_tensor_7: "f32[128, 128, 1024]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        add_tensor_1: "f32[128, 128, 1024]" = torch.ops.aten.add.Tensor(tangents_1, mul_tensor_7);  tangents_1 = mul_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:376 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_default: "f32[128, 128, 1024]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor_8: "f32[128, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_9: "f32[128, 128, 1024]" = torch.ops.aten.mul.Tensor(add_tensor_1, mul_tensor_8);  add_tensor_1 = mul_tensor_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:250 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default_2: "f32[16384, 1024]" = torch.ops.aten.reshape.default(mul_tensor_9, _shape_param_2);  mul_tensor_9 = _shape_param_2 = None
        permute_default: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_default_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_2, permute_default_1)



def make_inputs():
    return [
    torch.randn([16384, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 1024], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [128, 128, 1024], dtype=torch.bool, device='cuda'),
    torch.randn([128, 128, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([128, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 128, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    [128, 128, 1024],  # _shape_param_0
    [128, 128, 1024],  # _shape_param_1
    [16384, 1024],  # _shape_param_2
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
