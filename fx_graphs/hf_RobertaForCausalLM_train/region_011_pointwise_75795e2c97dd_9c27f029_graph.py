class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "i64[32, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:157 in create_position_ids_from_input_ids, code: mask = input_ids.ne(padding_idx).int()
        ne_scalar: "b8[32, 512]" = torch.ops.aten.ne.Scalar(primals_2, 0);  primals_2 = None
        convert_element_type_default: "i32[32, 512]" = torch.ops.prims.convert_element_type.default(ne_scalar, torch.int32);  ne_scalar = None
        return convert_element_type_default
