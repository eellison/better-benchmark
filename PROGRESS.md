# Repro Quality Progress

## Priority Tasks

### 1. Fix partitioner: no horizontal fusion of independent chains
- [ ] `CapabilityBasedPartitioner` groups all reachable supported nodes, not just data-dependent ones
- [ ] After partitioning, split each partition into connected components (by data flow)
- [ ] Handle `operator.getitem` correctly (it IS a data dependency even though it's "transparent")
- [ ] Add test: two independent pointwise chains should NOT be one repro
- [ ] Add test: var_mean → getitem → pointwise SHOULD be one repro (data dependent)

### 2. Recapture all models with fixed partitioner
- [ ] Delete old canonical repros
- [ ] Re-run timm (18 models, infer+train)
- [ ] Re-run HF (32 models, infer+train)
- [ ] Re-run torchbench (63 models, infer+train)
- [ ] Re-run genai (8 kernels)
- [ ] Re-run vLLM (5 models)
- [ ] Validate all repros pass eager

### 3. Format versioning + upgrade path
- [ ] All repros have `_repro_version = N`
- [ ] `upgrade_repros.py` re-compiles through hook to get current format
- [ ] `test_repro_integrity.py` asserts all repros have current version
- [ ] After any format change: run upgrade, validate, commit

### 4. Fix vLLM MoE routing index bounds
- [ ] The 11 failing repros use blanket `gen=Index(100)` instead of proper bounds
- [ ] After capture, if a repro fails eager validation, try increasing bounds automatically
- [ ] Iterative: try Index(1000), Index(10000), until it passes or we give up

### 5. Clean up old/stale artifacts
- [ ] Remove old results files that are now stale (results_baseline_v2.json, results_v1_sampled.json)
- [ ] Remove stale shapes.txt entries after recapture
- [ ] Ensure no `__pycache__` committed
- [ ] Remove any repros without `_repro_version`

## Status
- Partitioner fix: NOT STARTED
- Recapture: BLOCKED on partitioner fix
- Format: v2 defined, 306/1453 upgraded, rest blocked on partitioner
- vLLM bounds: NOT STARTED
- Cleanup: NOT STARTED
