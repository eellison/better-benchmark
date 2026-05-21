class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[2, 200000][200000, 1]cuda:0"):
        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch_geometric/utils/loop.py:340 in add_remaining_self_loops, code: mask = edge_index[0] != edge_index[1]
        select: "i64[200000][1]cuda:0" = torch.ops.aten.select.int(arg0_1, 0, 0)
        select_1: "i64[200000][1]cuda:0" = torch.ops.aten.select.int(arg0_1, 0, 1);  arg0_1 = None
        ne: "b8[200000][1]cuda:0" = torch.ops.aten.ne.Tensor(select, select_1);  select = select_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch_geometric/utils/loop.py:342 in add_remaining_self_loops, code: loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
        iota: "i64[10000][1]cuda:0" = torch.ops.prims.iota.default(10000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch_geometric/utils/loop.py:343 in add_remaining_self_loops, code: loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        unsqueeze: "i64[1, 10000][10000, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
        repeat: "i64[2, 10000][10000, 1]cuda:0" = torch.ops.aten.repeat.default(unsqueeze, [2, 1]);  unsqueeze = None
        return (ne, repeat)
