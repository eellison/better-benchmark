class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[4, 474, 768][364032, 768, 1]cuda:0"):
        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/embeddings/bert_embedding.py:138 in torch_dynamo_resume_in_forward_at_137, code: outputs = torch.cat([*outputs], dim=-1)
        full_default: "f32[1, 4, 474, 768][1456128, 364032, 768, 1]cuda:0" = torch.ops.aten.full.default([1, 4, 474, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # No stacktrace found for following nodes
        select_scatter_default: "f32[1, 4, 474, 768][1456128, 364032, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default, tangents_1, 0, 0);  full_default = tangents_1 = None
        return (select_scatter_default,)
