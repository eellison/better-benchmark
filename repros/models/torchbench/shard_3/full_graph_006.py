class GraphModule(torch.nn.Module):
    def forward(self, view: "f16[1896, 768][768, 1]cuda:0", permute_1: "f16[2, 768][768, 1]cuda:0", tangents_1: "f16[1896, 2][2, 1]cuda:0"):
        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/models/bert.py:266 in torch_dynamo_resume_in_forward_at_265, code: logits = self.qa_outputs(sequence_output)  # [batch_size, seq_len, num_labels]
        mm: "f16[1896, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
        permute_2: "f16[2, 1896][1, 2]cuda:0" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "f16[2, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
        sum_1: "f32[1, 2][2, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True, dtype = torch.float32);  tangents_1 = None
        view_2: "f32[2][1]cuda:0" = torch.ops.aten.reshape.default(sum_1, [2]);  sum_1 = None
        view_3: "f16[4, 474, 768][364032, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [4, 474, 768]);  mm = None
        convert_element_type_11: "f32[4, 474, 768][364032, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        convert_element_type_12: "f32[2, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        return (convert_element_type_12, view_2, convert_element_type_11)
