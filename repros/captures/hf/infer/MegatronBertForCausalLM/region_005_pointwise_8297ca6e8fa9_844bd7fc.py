"""
Standalone repro captured via capture_hook.
Label: hf_MegatronBertForCausalLM_infer
Pattern hash: 8297ca6e8fa9
Shape hash: 844bd7fc
"""
_shapes_config = "(T([16, 16, 512, 64], f32, stride=(524288, 64, 1024, 1)), T([1024, 1024], f32), S([16, 512, 1024]), S([8192, 1024]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_100: "f32[16, 16, 512, 64]", arg382_1: "f32[1024, 1024]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_default: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_100, [0, 2, 1, 3]);  getitem_100 = None
        clone_default: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        reshape_default: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_1: "f32[8192, 1024]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg382_1, [1, 0]);  arg382_1 = None
        return (reshape_default_1, permute_default_1)



def make_inputs():
    return [
    torch.randn(8388608, dtype=torch.float32, device='cuda').as_strided([16, 16, 512, 64], [524288, 64, 1024, 1]),  # getitem_100
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    [16, 512, 1024],  # _shape_param_0
    [8192, 1024],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
