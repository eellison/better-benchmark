"""
Standalone repro captured via capture_hook.
Label: distilbert
Pattern hash: c0f0885a79c6
Shape hash: f660bb6d
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg107_1: "f16[30522]", convert_element_type_default: "f32[4, 512, 768]", getitem_81: "f32[4, 512, 1]", getitem_80: "f32[4, 512, 1]", arg105_1: "f16[768]", arg106_1: "f16[768]", _shape_param_0, arg1_1: "f16[30522, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:517 in forward, code: prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
        full_default: "f16[6]" = torch.ops.aten.full.default([6], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default: "f16[30528]" = torch.ops.aten.cat.default([arg107_1, full_default]);  arg107_1 = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:516 in forward, code: prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        sub_tensor: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_81);  convert_element_type_default = getitem_81 = None
        add_tensor: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-12);  getitem_80 = None
        rsqrt_default: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg105_1);  mul_tensor = arg105_1 = None
        add_tensor_1: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg106_1);  mul_tensor_1 = arg106_1 = None
        convert_element_type_default_1: "f16[4, 512, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:517 in forward, code: prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
        reshape_default: "f16[2048, 768]" = torch.ops.aten.reshape.default(convert_element_type_default_1, _shape_param_0);  convert_element_type_default_1 = _shape_param_0 = None
        permute_default: "f16[768, 30522]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        constant_pad_nd_default: "f16[768, 30528]" = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 6, 0, 0]);  permute_default = None
        return (cat_default, reshape_default, constant_pad_nd_default)



def make_inputs():
    return [
    torch.randn([30522], dtype=torch.float16, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float16, device='cuda'),
    torch.randn([768], dtype=torch.float16, device='cuda'),
    [2048, 768],  # _shape_param_0
    torch.randn([30522, 768], dtype=torch.float16, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
