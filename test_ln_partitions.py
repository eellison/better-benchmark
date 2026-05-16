"""
Test: does the ATen-level extraction capture LayerNorm backward as one partition?

LayerNorm bwd has:
- Inner reductions: sum(tangents * weight, dim=-1), sum(tangents * weight * normalized, dim=-1)
- Outer reductions: sum(tangents * normalized, dim=[0,1]) for weight grad
- db = sum(tangents, dim=[0,1]) for bias grad

Inductor fuses all of these into one kernel (mix-order reduction).
The capability partitioner should too, if shared intermediates connect them.
"""
import copy
import torch
import torch.fx as fx
import torch._inductor.config as inductor_config

inductor_config.force_disable_caches = True
inductor_config.split_reductions = False


def capture_and_partition():
    captured_gms = []

    def capture_post_grad(graph_or_gm):
        if isinstance(graph_or_gm, fx.GraphModule):
            captured_gms.append(copy.deepcopy(graph_or_gm))
        elif isinstance(graph_or_gm, fx.Graph):
            if hasattr(graph_or_gm, 'owning_module') and graph_or_gm.owning_module is not None:
                captured_gms.append(copy.deepcopy(graph_or_gm.owning_module))
        return graph_or_gm

    # Simple LayerNorm model
    class LNModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = torch.nn.LayerNorm(768)

        def forward(self, x):
            return self.ln(x)

    model = LNModel().cuda()
    x = torch.randn(8, 512, 768, device="cuda", requires_grad=True)

    torch._dynamo.reset()
    inductor_config.post_grad_custom_pre_pass = capture_post_grad

    compiled = torch.compile(model)
    out = compiled(x)
    loss = out.sum()
    loss.backward()

    inductor_config.post_grad_custom_pre_pass = None

    print(f"Captured {len(captured_gms)} post-grad graphs")

    from torch._inductor.fx_passes.fusion_regions import is_fusible_node
    from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
    from torch.fx.passes.operator_support import create_op_support

    def _is_supported(_submodules, node):
        return is_fusible_node(node)

    support = create_op_support(_is_supported)

    for i, gm in enumerate(captured_gms):
        n_nodes = sum(1 for n in gm.graph.nodes if n.op == "call_function")
        print(f"\n=== Graph {i}: {n_nodes} call_function nodes ===")

        # Show all ops
        ops = {}
        for n in gm.graph.nodes:
            if n.op == "call_function":
                name = str(n.target)
                ops[name] = ops.get(name, 0) + 1
                # Check fusibility
                fusible = is_fusible_node(n)
                is_red = False
                if isinstance(n.target, torch._ops.OpOverload):
                    is_red = torch.Tag.reduction in n.target.tags
                if is_red:
                    val = n.meta.get("val", None)
                    shape = list(val.shape) if isinstance(val, torch.Tensor) else "?"
                    print(f"  REDUCTION: {name} -> {shape} (fusible={fusible})")

        print(f"  Op counts: {dict(sorted(ops.items()))}")

        partitioner = CapabilityBasedPartitioner(
            gm, support, allows_single_node_partition=True,
            skip_horizontal_fusion=True,
        )
        partitions = partitioner.propose_partitions()

        print(f"  {len(partitions)} partitions (skip_horizontal_fusion=True)")

        for j, p in enumerate(partitions):
            nodes = list(p.nodes.keys())
            n_ops = sum(1 for n in nodes if n.op == "call_function")
            reductions = []
            for n in nodes:
                if n.op == "call_function" and isinstance(n.target, torch._ops.OpOverload):
                    if torch.Tag.reduction in n.target.tags:
                        val = n.meta.get("val", None)
                        shape = list(val.shape) if isinstance(val, torch.Tensor) else "?"
                        reductions.append(f"{n.target.overloadpacket.__name__}->{shape}")
            has_red = " [HAS REDUCTION]" if reductions else ""
            print(f"    Partition {j}: {n_ops} ops{has_red}")
            if reductions:
                for r in reductions:
                    print(f"      {r}")

        # Also try WITHOUT skip_horizontal_fusion for comparison
        partitioner2 = CapabilityBasedPartitioner(
            gm, support, allows_single_node_partition=True,
            skip_horizontal_fusion=False,
        )
        partitions2 = partitioner2.propose_partitions()
        print(f"  {len(partitions2)} partitions (skip_horizontal_fusion=False)")
        for j, p in enumerate(partitions2):
            nodes = list(p.nodes.keys())
            n_ops = sum(1 for n in nodes if n.op == "call_function")
            reductions = []
            for n in nodes:
                if n.op == "call_function" and isinstance(n.target, torch._ops.OpOverload):
                    if torch.Tag.reduction in n.target.tags:
                        val = n.meta.get("val", None)
                        shape = list(val.shape) if isinstance(val, torch.Tensor) else "?"
                        reductions.append(f"{n.target.overloadpacket.__name__}->{shape}")
            has_red = " [HAS REDUCTION]" if reductions else ""
            print(f"    Partition {j}: {n_ops} ops{has_red}")
            if reductions:
                for r in reductions:
                    print(f"      {r}")


if __name__ == "__main__":
    capture_and_partition()
