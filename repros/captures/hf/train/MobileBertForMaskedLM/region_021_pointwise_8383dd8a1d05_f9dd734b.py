"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train
Pattern hash: 8383dd8a1d05
Shape hash: f9dd734b
"""
_shapes_config = "(T([32768, 512], f32), T([256, 128, 512], b8), T([], f32), T([512, 128], f32), S([256, 128, 512]), S([32768, 512]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_709: "f32[32768, 512]", le_96: "b8[256, 128, 512]", full_default_2: "f32[]", primals_28: "f32[512, 128]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        reshape_default: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(mm_709, _shape_param_0);  mm_709 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        where_self: "f32[256, 128, 512]" = torch.ops.aten.where.self(le_96, full_default_2, reshape_default);  le_96 = full_default_2 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_1: "f32[32768, 512]" = torch.ops.aten.reshape.default(where_self, _shape_param_1);  where_self = _shape_param_1 = None
        permute_default: "f32[128, 512]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
        permute_default_1: "f32[512, 128]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_1, permute_default_1)



def make_inputs():
    return [
    torch.randn([32768, 512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [256, 128, 512], dtype=torch.bool, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([512, 128], dtype=torch.float32, device='cuda'),
    [256, 128, 512],  # _shape_param_0
    [32768, 512],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
