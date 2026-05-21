class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[32, 128]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/moco/moco/builder.py:180 in concat_all_gather, code: torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
        full_default: "f32[32, 128]" = torch.ops.aten.full.default([32, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:209 in all_gather_tensor, code: tensor = torch.ops._c10d_functional.all_gather_into_tensor(
        all_gather_into_tensor: "f32[32, 128]" = torch.ops._c10d_functional.all_gather_into_tensor.default(arg0_1, 1, '0');  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:142 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor: "f32[32, 128]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None
        return (wait_tensor,)
