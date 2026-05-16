"""
Standalone repro captured via capture_hook.
Label: pythia_410m
Pattern hash: 287090d98877
Shape hash: 123d99b9
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convert_element_type_436: "f32[4, 512, 1024]", getitem_385: "f32[4, 512, 1]", getitem_384: "f32[4, 512, 1]", arg292_1: "f16[1024]", arg293_1: "f16[1024]", _shape_param_0, arg294_1: "f16[50304, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:376 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        sub_tensor: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_436, getitem_385);  convert_element_type_436 = getitem_385 = None
        add_tensor: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_384, 1e-05);  getitem_384 = None
        rsqrt_default: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, arg292_1);  mul_tensor = arg292_1 = None
        add_tensor_1: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg293_1);  mul_tensor_1 = arg293_1 = None
        convert_element_type_default: "f16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:464 in forward, code: logits = self.embed_out(hidden_states[:, slice_indices, :])
        reshape_default: "f16[2048, 1024]" = torch.ops.aten.reshape.default(convert_element_type_default, _shape_param_0);  convert_element_type_default = _shape_param_0 = None
        permute_default: "f16[1024, 50304]" = torch.ops.aten.permute.default(arg294_1, [1, 0]);  arg294_1 = None
        return (reshape_default, permute_default)



def make_inputs():
    return [
    torch.randn([4, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float16, device='cuda'),
    torch.randn([1024], dtype=torch.float16, device='cuda'),
    [2048, 1024],  # _shape_param_0
    torch.randn([50304, 1024], dtype=torch.float16, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
