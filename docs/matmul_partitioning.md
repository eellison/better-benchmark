# Core MM fusion corpus

## Generate the committed corpus

From the repository root, with a compatible CUDA-enabled PyTorch nightly:

```bash
python scripts/extract_matmul_corpus.py
```

The command reads the saved `repros/models/**/full_graph_*.py` and metadata
sidecars, runs CPU/FakeTensor capture, applies core-only selection and semantic
normalization, and writes **only `mm_repro/`**. Existing generated entries are
replaced atomically, so old hashes cannot remain mixed into the new corpus.
Source captures are held in an automatically cleaned temporary staging tree.
The corpus in this commit is **276 repros / 501 shape configurations**. This is
the outcome of the committed source graphs and heuristics, not a quota or a
count limit. Different source graphs or PyTorch decompositions can change it.

Optional arguments:

```bash
python scripts/extract_matmul_corpus.py --output /tmp/mm_repro_check --workers 2
python scripts/extract_matmul_corpus.py --output /tmp/mm_one_graph \
  --source repros/models/torchbench/infer/maml/full_graph_001.py
```

Each `mm_repro/canonical/<mm_names>_<hash>/` contains the actual Torch FX code
(`repro.py`), all input shape/dtype/stride/offset/generator configurations
(`shapes.json`), and identity/normalization metadata (`meta.json`). The report,
inventory, model manifests, and source input/output mappings are generated too.
No handwritten example corpus or pre-existing extracted corpus is required.

## Partition policy

The detailed comment block in `matmul_partitioner.py` documents the purpose,
search, rejection rules and boundaries. The original model graph is unchanged;
selected benchmark windows can overlap.

* MM/BMM/ADDMM/MV/etc. and schema-tagged ordinary reductions share one maximum
  depth of five reductions on every value dependency path.
* Ordinary reductions can precede MM. A final region contains at least one MM.
  Gather, real pointwise operations and views may precede or follow reductions.
* Every ordinary reduction consumes at most128 elements per result, using the
  product of its actual reduced axes. First-MM K is unrestricted; subsequent
  internal MM K is at most128.
* Independent horizontal MMs have identical full output shapes, including
  batch/M/N. K may differ; vertical MM output shapes may change.
* Unbind and oversized/unknown ordinary reductions are boundaries. Tuple
  projections outside a region become separate tensor inputs.
* Outputs need MM plus actual arithmetic/indexing. Cast/view/allocation-only
  additions do not make a bare MM meaningful. Pure horizontal MMs are retained.
* Single-MM alignment padding/crop, intrinsic bias packing, whole input copies,
  and sole BMM infinity masks are rejected by the partitioner. They are not
  captured or exported as another corpus. Real PW/reductions/gather and
  horizontal regions remain eligible.
* Escaping outputs, graph reentry, device transfers and unsafe aliases are
  bounded or rejected. Search snapshots retain a latest valid smaller window
  when a maximal one violates these constraints.

The growth envelope is maxMM4/nodes96/compute48/inputs24/outputs4. Completed
regions then pass the final maxMM3/nodes42/compute32/inputs8/outputs2 envelope.
These are local size limits, without ranking, pattern exceptions or sampling.
`max_contraction_depth` is a constructor alias for older saved policies; new
captures serialize `max_reductions_chain`.

## Canonicalization and input correctness

MM hashes encode scalar literals, dtype/modes, structural axes and nested
argument wiring; dead tuple-projection tracing noise is ignored. Other capture
hash defaults remain compatible with the existing non-MM corpus.

Normalization only removes or changes operations whose value/layout contracts
are proven for the recorded inputs: input permutations (including checked float
casts), identity metadata views, contiguous reshape chains, reduction axes and
keepdim interfaces, internal pure CSE and identity adaptive pooling. Equivalent
DAGs can merge across input order and commutative operands. Every input shape,
dtype, actual stride, storage offset, alias relation and source mapping remains.
Different real layouts, casts/rounding, and scalar values are preserved.

External floating sqrt/rsqrt statistics receive an explicit positive generator
(`abs(randn)+0.1`) through a conservative value-dependency rule. Computed
variance and arbitrary floating inputs are not changed. Floating input replay
preserves requested storage offsets, including strided and alias inputs.

## Validation

```bash
python scripts/validate_matmul_repros.py mm_repro
pytest tests/test_matmul_partitioner.py tests/test_mm_corpus_generation.py \
  tests/test_mm_corpus_dedup.py tests/test_mm_scalar_hash.py \
  tests/test_mm_input_constraints.py tests/test_mm_corpus_quality.py
```

FakeTensor gates check execution and metadata; targeted CPU tests check scalar
sensitivity, BF16 rounding and alias/layout equivalence. The corpus covers the
eight requested fusion categories, including one ordinary reduction→MM→
reduction case. The exact gather/LayerNorm/MM/LayerNorm combination is absent
from the saved input graphs. Selection does not certify one GPU kernel or a
performance improvement.
