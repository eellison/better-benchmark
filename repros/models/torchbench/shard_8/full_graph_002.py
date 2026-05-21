class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "Sym(s77)", primals_2: "Sym(s7)", primals_3: "f16[s77, 1][s7, 1]cuda:0"):
        # File: /torchbench/torchbenchmark/util/distribution.py:31 in _call, code: return x.tanh()
        tanh: "f16[s77, 1][1, 1]cuda:0" = torch.ops.aten.tanh.default(primals_3);  primals_3 = None
        return (tanh, tanh, primals_1)
