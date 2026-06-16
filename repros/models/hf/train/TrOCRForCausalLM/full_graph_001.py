class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[514, 1024][1024, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:55 in forward, code: position_ids = torch.arange(
        iota: "i64[256][1]cuda:0" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:57 in forward, code: ).expand(bsz, -1)
        expand: "i64[64, 256][0, 1]cuda:0" = torch.ops.aten.expand.default(iota, [64, -1]);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:61 in forward, code: return super().forward(position_ids + self.offset)
        add: "i64[64, 256][256, 1]cuda:0" = torch.ops.aten.add.Tensor(expand, 2);  expand = None
        embedding: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.embedding.default(primals_1, add);  primals_1 = None
        return (embedding, add)
