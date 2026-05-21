"""
Standalone repro captured via capture_hook.
Label: hf_T5ForConditionalGeneration_train
Pattern hash: e4c7457c2f54
Shape hash: b2d80d1c
"""
_shapes_config = "(T([8192, 2048], f32), T([64], i64), T([512, 2048], f32), S([8, 1024, 2048]), S([8192, 2048]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_94: "f32[8192, 2048]", inductor_seeds_default: "i64[64]", primals_132: "f32[512, 2048]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        reshape_default: "f32[8, 1024, 2048]" = torch.ops.aten.reshape.default(mm_94, _shape_param_0);  mm_94 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        relu_default: "f32[8, 1024, 2048]" = torch.ops.aten.relu.default(reshape_default);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 61);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 1024, 2048]" = torch.ops.prims.inductor_random.default([8, 1024, 2048], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[8, 1024, 2048]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(gt_scalar, relu_default);  gt_scalar = relu_default = None
        mul_tensor_1: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        permute_default: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
        reshape_default_1: "f32[8192, 2048]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_1);  mul_tensor_1 = _shape_param_1 = None
        return (permute_default, reshape_default_1)



def make_inputs():
    return [
    torch.randn([8192, 2048], dtype=torch.float32, device='cuda'),
    torch.randint(0, 64, [64], dtype=torch.int64, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    [8, 1024, 2048],  # _shape_param_0
    [8192, 2048],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
