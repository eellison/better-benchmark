class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 64][64, 1]cuda:0", primals_2: "f32[10000, 64][64, 1]cuda:0", primals_3: "i64[2, 209981][209981, 1]cuda:0", primals_4: "f32[209981][1]cuda:0", primals_5: "f32[64][1]cuda:0"):
        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch_geometric/nn/dense/linear.py:132 in forward, code: return F.linear(x, self.weight, self.bias)
        convert_element_type: "f16[64, 64][64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_1, torch.float16);  primals_1 = None
        convert_element_type_1: "f16[10000, 64][64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_2, torch.float16);  primals_2 = None
        permute: "f16[64, 64][1, 64]cuda:0" = torch.ops.aten.permute.default(convert_element_type, [1, 0]);  convert_element_type = None
        mm: "f16[10000, 64][64, 1]cuda:0" = torch.ops.aten.mm.default(convert_element_type_1, permute)

        # File: /tmp/jenkins_pyg/tmpipi09vqz.py:88 in _lift, code: index = edge_index[dim]
        select: "i64[209981][1]cuda:0" = torch.ops.aten.select.int(primals_3, 0, 0)

        # File: /tmp/jenkins_pyg/tmpipi09vqz.py:89 in _lift, code: return src.index_select(self.node_dim, index)
        index: "f16[209981, 64][64, 1]cuda:0" = torch.ops.aten.index.Tensor(mm, [select]);  mm = select = None

        # File: /tmp/jenkins_pyg/tmpipi09vqz.py:139 in _collect, code: edge_index_i = edge_def[i]
        select_1: "i64[209981][1]cuda:0" = torch.ops.aten.select.int(primals_3, 0, 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch_geometric/nn/conv/gcn_conv.py:241 in message, code: return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
        view: "f32[209981, 1][1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_4, [-1, 1])
        mul: "f32[209981, 64][64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view, index);  view = index = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:169 in broadcast, code: return src.view(size).expand_as(ref)
        view_1: "i64[209981, 1][1, 1]cuda:0" = torch.ops.aten.reshape.default(select_1, [-1, 1]);  select_1 = None
        expand: "i64[209981, 64][1, 0]cuda:0" = torch.ops.aten.expand.default(view_1, [209981, 64]);  view_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch_geometric/utils/scatter.py:74 in scatter, code: return src.new_zeros(size).scatter_add_(dim, index, src)
        full_default: "f32[10000, 64][64, 1]cuda:0" = torch.ops.aten.full.default([10000, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scatter_add: "f32[10000, 64][64, 1]cuda:0" = torch.ops.aten.scatter_add.default(full_default, 0, expand, mul);  full_default = expand = mul = None

        # File: /tmp/jenkins_pyg/tmpipi09vqz.py:247 in torch_dynamo_resume_in_forward_at_221, code: out = out + self.bias
        add: "f32[10000, 64][64, 1]cuda:0" = torch.ops.aten.add.Tensor(scatter_add, primals_5);  scatter_add = primals_5 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch_geometric/nn/dense/linear.py:132 in forward, code: return F.linear(x, self.weight, self.bias)
        permute_3: "f16[64, 64][64, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (add, primals_3, primals_4, convert_element_type_1, permute_3)
