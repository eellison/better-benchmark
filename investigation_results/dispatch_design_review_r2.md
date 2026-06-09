# Adversarial Review R2: oracle_impl dispatch (run inline, Bedrock killed the agent)

Round 1 covered: parser vs corpus, empty/S()-only sigs, downstream parsers,
worker isolation, callable audit, S/T alignment, f8. Round 2 fresh surfaces:

## Findings

1. **MAJOR (FIXED) — check/bench callable skew.** `check_oracle` did not
   resolve dispatch: `--check` verified the raw `oracle_forward` while
   `--bench` timed the dispatched variant (e.g. a B200 kwargs variant).
   Correctness was being attested for a different function than the one
   measured. Fix: check_oracle now calls resolve_oracle first and reports
   NO_ORACLE_FOR_SHAPE -> False on dispatch failure.

2. **MINOR (FIXED) — duplicate exact registrations shadow silently.** Two
   registrations at identical (hardware, signature) dispatch to the first;
   the second is dead code with no signal. Registration now warns to stderr
   on an exact duplicate point so a copy-paste during mass migration is
   visible instead of silent.

3. **MINOR (documented) — variants-only modules hard-fail off-point.** If a
   file registers only e.g. a B200 variant and never registers
   oracle_forward, inputs at any other shape raise instead of falling back
   to the plain oracle_forward. Migration rule: ALWAYS register
   oracle_forward (the mechanical migration does); variants are additive.

4. **NIT — eval namespace is not a sandbox.** `T.__globals__` inside a
   signature string reaches oracle_harness module globals (os etc.).
   Signature strings are repo-internal and trusted; do not feed untrusted
   strings to parse_shapes_signature. Not worth an AST parser today.

5. **OK — CPU bench path.** resolve_oracle runs before the cuda/cpu fork in
   bench_oracle, so _bench_oracle_cpu times the dispatched callable too
   (it just omits the dispatch field in its JSON — cosmetic only).

6. **OK — status-set exhaustiveness.** All `status` comparisons in scripts/
   are equality checks against specific values (== "BAD_ORACLE") or .get()
   aggregations; NO_ORACLE_FOR_SHAPE flows through as "not measured" rather
   than crashing or being misclassified.

7. **OK — re-import behavior.** Loading the same oracle file under two
   module names yields independent registries (1 entry each); no duplicate
   accumulation. Stale registries persist in _module_registries for the
   process lifetime — negligible (subprocess-per-oracle execution model).

8. **OK — __main__ keying.** Running `python oracle_x.py` keys the registry
   as "__main__" and resolve_oracle looks up oracle_forward.__module__ ==
   "__main__" in the same process — verified by both pilots on GPU.
