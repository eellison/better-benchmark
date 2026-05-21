"""
Standalone repro captured via capture_hook.
Label: hf_RobertaForCausalLM_train
Pattern hash: 75795e2c97dd
Shape hash: 9c27f029
"""
_shapes_config = "(T([32, 512], i64))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, primals_2: "i64[32, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:157 in create_position_ids_from_input_ids, code: mask = input_ids.ne(padding_idx).int()
        ne_scalar: "b8[32, 512]" = torch.ops.aten.ne.Scalar(primals_2, 0);  primals_2 = None
        convert_element_type_default: "i32[32, 512]" = torch.ops.prims.convert_element_type.default(ne_scalar, torch.int32);  ne_scalar = None
        return convert_element_type_default



def make_inputs():
    return [
    torch.randint(0, 512, [32, 512], dtype=torch.int64, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
