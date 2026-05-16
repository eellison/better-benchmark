"""
Standalone repro captured via capture_hook.
Label: distilbert
Pattern hash: 38d5f8e025b9
Shape hash: bae9172e
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convert_element_type_121: "f32[4, 512, 768]", getitem_66: "f32[4, 512, 1]", getitem_65: "f32[4, 512, 1]", arg85_1: "f16[768]", arg86_1: "f16[768]", _shape_param_0, arg87_1: "f16[768, 768]", _shape_param_1, arg89_1: "f16[768, 768]", _shape_param_2, arg91_1: "f16[768, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        sub_tensor: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_121, getitem_66);  convert_element_type_121 = getitem_66 = None
        add_tensor: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_65, 1e-12);  getitem_65 = None
        rsqrt_default: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg85_1);  mul_tensor = arg85_1 = None
        add_tensor_1: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg86_1);  mul_tensor_1 = arg86_1 = None
        convert_element_type_default: "f16[4, 512, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:187 in forward, code: query_layer = self.q_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default: "f16[2048, 768]" = torch.ops.aten.reshape.default(convert_element_type_default, _shape_param_0);  _shape_param_0 = None
        permute_default: "f16[768, 768]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:188 in forward, code: key_layer = self.k_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_1: "f16[2048, 768]" = torch.ops.aten.reshape.default(convert_element_type_default, _shape_param_1);  _shape_param_1 = None
        permute_default_1: "f16[768, 768]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:189 in forward, code: value_layer = self.v_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_2: "f16[2048, 768]" = torch.ops.aten.reshape.default(convert_element_type_default, _shape_param_2);  convert_element_type_default = _shape_param_2 = None
        permute_default_2: "f16[768, 768]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        return (reshape_default, permute_default, reshape_default_1, permute_default_1, reshape_default_2, permute_default_2)



def make_inputs():
    return [
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float16, device='cuda'),
    torch.randn([768], dtype=torch.float16, device='cuda'),
    [2048, 768],  # _shape_param_0
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    [2048, 768],  # _shape_param_1
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    [2048, 768],  # _shape_param_2
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
