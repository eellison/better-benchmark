class GraphModule(torch.nn.Module):
    def forward(self, arg1_1: "i64[64, 128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:177 in create_position_ids_from_input_ids, code: mask = input_ids.ne(padding_idx).int()
        ne_scalar: "b8[64, 128]" = torch.ops.aten.ne.Scalar(arg1_1, 1);  arg1_1 = None
        convert_element_type_default: "i32[64, 128]" = torch.ops.prims.convert_element_type.default(ne_scalar, torch.int32);  ne_scalar = None
        return convert_element_type_default
