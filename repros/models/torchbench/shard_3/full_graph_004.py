class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[1, 4, 474, 768][1456128, 364032, 768, 1]cuda:0"):
        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/embeddings/bert_embedding.py:138 in torch_dynamo_resume_in_forward_at_137, code: outputs = torch.cat([*outputs], dim=-1)
        select: "f32[4, 474, 768][364032, 768, 1]cuda:0" = torch.ops.aten.select.int(primals_1, 0, 0);  primals_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/embeddings/embedding.py:160 in dropout, code: return self.dropout_layer(words)
        clone_1: "f32[4, 474, 768][364032, 768, 1]cuda:0" = torch.ops.aten.clone.default(select);  select = None
        return (clone_1,)
