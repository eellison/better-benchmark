"""
Standalone repro captured via capture_hook.
Label: hf_M2M100ForConditionalGeneration_train
Pattern hash: 75795e2c97dd
Shape hash: 5d9a9b51
"""
_shapes_config = "(T([64, 128], i64))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[64, 128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:177 in create_position_ids_from_input_ids, code: mask = input_ids.ne(padding_idx).int()
        ne_scalar: "b8[64, 128]" = torch.ops.aten.ne.Scalar(arg0_1, 1);  arg0_1 = None
        convert_element_type_default: "i32[64, 128]" = torch.ops.prims.convert_element_type.default(ne_scalar, torch.int32);  ne_scalar = None
        return convert_element_type_default



def make_inputs():
    return [
    torch.randint(0, 128, [64, 128], dtype=torch.int64, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
