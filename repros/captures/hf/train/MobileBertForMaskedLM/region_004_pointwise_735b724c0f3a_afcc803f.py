"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train
Pattern hash: 735b724c0f3a
Shape hash: afcc803f
"""
_shapes_config = "(T([32768, 512], f32), T([128, 512], f32), S([256, 128, 512]), S([32768, 512]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_358: "f32[32768, 512]", primals_1106: "f32[128, 512]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_358, _shape_param_0);  addmm_358 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_default: "f32[256, 128, 512]" = torch.ops.aten.relu.default(reshape_default);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        reshape_default_1: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_default, _shape_param_1);  relu_default = _shape_param_1 = None
        permute_default: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1106, [1, 0]);  primals_1106 = None
        return (reshape_default_1, permute_default)



def make_inputs():
    return [
    torch.randn([32768, 512], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512], dtype=torch.float32, device='cuda'),
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
