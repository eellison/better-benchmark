class GraphModule(torch.nn.Module):
    def forward(self, addmm: "f32[16384, 768]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:137 in forward, code: embeddings = self.projection(embeddings)
        reshape_default: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:176 in forward, code: outputs = self.fourier_transform(hidden_states).real
        convert_element_type_default: "c64[32, 512, 768]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.complex64);  reshape_default = None
        return convert_element_type_default
