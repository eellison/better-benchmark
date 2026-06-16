# Follow-up (FILE ONLY — do not fix yet): trim boundary views from partitions

Status: LOGGED 2026-06-16. Not actioned. Capture/partitioning side.

## The issue
Partitions can include a leading view on an INPUT (or a trailing view on an
OUTPUT). A view is pure metadata (no compute), so the partition is semantically
identical to one that starts *after* the input view / ends *before* the output
view. Example captured partition:

    def forward(self, arg35_1: "f32[64, 4, 2, 371372]", _shape_param_0):
        view_default: "f32[64, 8, 371372]" = torch.ops.aten.view.default(arg35_1, _shape_param_0)
        sum_dim_int_list: "f32[8]" = torch.ops.aten.sum.dim_IntList(view_default, [0, 2])
        return sum_dim_int_list

The leading `view` adds nothing — this is equivalent to a partition whose input
is already `f32[64, 8, 371372]` and which just does the `sum`.

## Why it matters
- Bloats the subgraph with no-op metadata ops.
- Distorts pattern IDENTITY: the same reduction hashes differently depending on
  whether a boundary view rides along — the same hash-fork failure mode already
  noted in CORPUS_MIGRATION_PLAN.md line 68 ("reshape/view hash fork: same
  partition counted as two patterns; duplicate oracle effort"). Boundary views
  on inputs/outputs are a specific, cleanly-trimmable instance of that.
- Oracles end up reproducing a no-op view to match scope, which is wasted.

## Proposed direction (NOT implemented)
At partition formation (capture_hook.get_fusion_partitions / pattern hashing),
peel transparent view/reshape ops that sit at the partition boundary on a
graph input or output: start the partition after a leading input-view, end it
before a trailing output-view. Interior views that feed real compute stay.
Keep it boundary-only so it doesn't change real dataflow. Re-hash after trim so
the view-vs-no-view variants collapse to one pattern.

## Why not now
Capture-side change; the migration is at 1726/1727 and shouldn't be perturbed.
Revisit post-flip alongside the existing reshape/view canonicalization work
(plan line 82, retrace∘serialize fixed-point).
