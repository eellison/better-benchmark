"""
Standalone repro captured via capture_hook.
Label: hf_DistillGPT2_train
Pattern hash: 3478a4eb1ba3
Shape hash: 39e37d3d
"""
_shapes_config = "(T([16384, 2304], f32), S([32, 512, 2304]), S([32, 512, -1, 64]), S([32, 512, -1, 64]), S([32, 512, -1, 64]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_20: "f32[16384, 2304]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        reshape_default: "f32[32, 512, 2304]" = torch.ops.aten.reshape.default(addmm_20, _shape_param_0);  addmm_20 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_tensor = torch.ops.aten.split.Tensor(reshape_default, 768, 2);  reshape_default = None
        getitem: "f32[32, 512, 768]" = split_tensor[0]
        getitem_1: "f32[32, 512, 768]" = split_tensor[1]
        getitem_2: "f32[32, 512, 768]" = split_tensor[2];  split_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        reshape_default_1: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_1, _shape_param_1);  getitem_1 = _shape_param_1 = None
        permute_default: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        reshape_default_2: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_2, _shape_param_2);  getitem_2 = _shape_param_2 = None
        permute_default_1: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        reshape_default_3: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem, _shape_param_3);  getitem = _shape_param_3 = None
        permute_default_2: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None
        return (permute_default, permute_default_1, permute_default_2)



def make_inputs():
    return [
    torch.randn([16384, 2304], dtype=torch.float32, device='cuda'),
    [32, 512, 2304],  # _shape_param_0
    [32, 512, -1, 64],  # _shape_param_1
    [32, 512, -1, 64],  # _shape_param_2
    [32, 512, -1, 64],  # _shape_param_3
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
