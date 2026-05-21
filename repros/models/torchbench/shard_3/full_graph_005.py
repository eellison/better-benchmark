class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[2, 768][768, 1]cuda:0", primals_2: "f32[2][1]cuda:0", primals_3: "f32[4, 474, 768][364032, 768, 1]cuda:0"):
        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/models/bert.py:266 in torch_dynamo_resume_in_forward_at_265, code: logits = self.qa_outputs(sequence_output)  # [batch_size, seq_len, num_labels]
        convert_element_type: "f16[2][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_2, torch.float16);  primals_2 = None
        convert_element_type_1: "f16[2, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_1, torch.float16);  primals_1 = None
        convert_element_type_2: "f16[4, 474, 768][364032, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_3, torch.float16);  primals_3 = None
        view: "f16[1896, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2, [1896, 768]);  convert_element_type_2 = None
        permute: "f16[768, 2][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "f16[1896, 2][2, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type, view, permute);  convert_element_type = None
        view_1: "f16[4, 474, 2][948, 2, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [4, 474, 2])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/models/bert.py:268 in torch_dynamo_resume_in_forward_at_265, code: return {'pred_start': logits[:, :, 0], 'pred_end': logits[:, :, 1]}
        select: "f16[4, 474][948, 2]cuda:0" = torch.ops.aten.select.int(view_1, 2, 0)
        select_1: "f16[4, 474][948, 2]cuda:0" = torch.ops.aten.select.int(view_1, 2, 1);  view_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/models/bert.py:266 in torch_dynamo_resume_in_forward_at_265, code: logits = self.qa_outputs(sequence_output)  # [batch_size, seq_len, num_labels]
        permute_1: "f16[2, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (select, select_1, addmm, view, permute_1)
