# Investigation: multi_kernel=1 broken on pr-184905 branch

## Summary

`triton.multi_kernel = 1` fails with `TypeError: 'NoneType' object does not support the context manager protocol` on **any repro that generates multi_kernel code AND hits the FxGraphCache**. This is a cache deserialization bug, not a multi_kernel logic bug.

## Root Cause

The error occurs because `triton.JITFunction._hash_lock` (a `threading.RLock()`) is set to `None` during cache serialization and never restored on the `StaticAutotunerFuture` deserialization path.

### Detailed Code Path

1. **First compile (succeeds):** `async_compile.triton()` creates `CachingAutotuner` with a valid `JITFunction` (`_hash_lock = threading.RLock()`). The graph is compiled and saved to `FxGraphCache`. During save, `TritonBundler.put_static_autotuner()` calls:
   - `kernel.prepare_for_pickle()` -- sets `kernel.fn._hash_lock = None`
   - `copy.deepcopy(kernel)` -- copies the kernel with `_hash_lock = None`
   - `kernel.restore_after_unpickle(old_values)` -- restores the **original** kernel, but the **copy** retains `_hash_lock = None`

2. **Second compile (fails on cache hit):** `FxGraphCache._lookup_graph()` loads the cached graph. `TritonBundler.load_autotuners()` creates `StaticAutotunerFuture(result.kernel)` with the deep-copied kernel that has `_hash_lock = None`. When `async_compile.triton()` gets a cache hit, it calls `future.result()` which calls `precompile()` -- but never calls `restore_after_unpickle()` to restore `_hash_lock`.

3. **Crash:** `MultiKernelCall.__init__()` -> `load_cache()` -> `cache_file_path()` accesses `k.fn.cache_key` which calls triton's `JITFunction.cache_key` property using `with self._hash_lock:` -- but `_hash_lock` is `None`.

### Key Files

- `/tmp/pytorch-work/torch/_inductor/codecache.py` line 5113-5127: `StaticAutotunerFuture.result()` -- missing `restore_after_unpickle()` call
- `/tmp/pytorch-work/torch/_inductor/triton_bundler.py` line 181-202: `put_static_autotuner()` -- where `_hash_lock` gets set to None on the copy
- `/tmp/pytorch-work/torch/_inductor/runtime/triton_heuristics.py` line 900-939: `prepare_for_pickle()` / `restore_after_unpickle()`
- `/tmp/pytorch-work/torch/_inductor/codegen/multi_kernel.py` line 314-324: `cache_file_path()` -- where the crash occurs

## History

- Triton PR #7974 added `_hash_lock = threading.RLock()` to `JITCallable`
- PyTorch PR #161768 added `_hash_lock` to `prepare_for_pickle()` to handle subprocess serialization
- PyTorch PR #162244 added `restore_after_unpickle()` specifically for the multi_kernel case, noting: "It turns out that we do need to restore the `fn._hash_lock` field, even in the non-triton_bundler case - the MultiKernel case uses the hash lock."
- **But the TritonBundler/StaticAutotunerFuture path was missed** -- it strips `_hash_lock` but never restores it.

## Fix

Add `self.static_autotuner.restore_after_unpickle(None)` in `StaticAutotunerFuture.result()`:

```python
# In torch/_inductor/codecache.py, StaticAutotunerFuture.result():
def result(self, timeout: float | None = None) -> CachingAutotuner:
    assert self.reload_kernel_from_src is not None
    with dynamo_timed("StaticAutotunerFuture.warm_precompile"):
        self.static_autotuner.restore_after_unpickle(None)  # <-- FIX: restore _hash_lock
        self.static_autotuner.recheck_autotune_cache(
            reload_kernel_from_src=self.reload_kernel_from_src
        )
        self.static_autotuner.precompile(
            warm_cache_only=False,
            reload_kernel=self.reload_kernel_from_src,
            static_triton_bundle_key=None,
        )
        return self.static_autotuner
```

`restore_after_unpickle(None)` does two things:
1. Sets `self._cached_launcher = None` (safe -- `precompile()` will recreate it)
2. Sets `self.fn._hash_lock = threading.RLock()` (the actual fix)

## Verification

Tested with the fix applied via monkey-patch:
- 30/30 repros that previously failed now pass
- First-compile + cache-hit-second-compile pattern works correctly
- Clearing FxGraphCache also works as a workaround

## Scope of Impact

- **Affected:** Any repro that generates multi_kernel code (reductions: sum, mean, var, amax, argmax, max) AND has been previously compiled (FxGraphCache hit)
- **Not affected:** Pointwise-only kernels, first-time compilations (no cache hit)
- **87.6% failure rate explained:** Most repros in the corpus are reductions (1083/1482), and the benchmark sweep populates the FxGraphCache on the first pass, causing all subsequent passes to fail. The 184 passing repros are likely pointwise-only kernels that don't generate multi_kernel code.

## Workaround

Clear the inductor cache before running with `multi_kernel=1`:
```python
from torch._inductor.codecache import FxGraphCache
FxGraphCache.clear()
```
Or set `TORCHINDUCTOR_CACHE_DIR` to a fresh directory.

## Is This Specific to This Branch?

No. This bug exists on any PyTorch version that has:
1. `use_static_triton_launcher = True` (default in OSS since ~v2.6)
2. Triton with `_hash_lock` on `JITCallable` (triton >= the version with PR #7974)
3. `triton.multi_kernel = 1`

It's a latent bug that only manifests when multi_kernel is enabled AND the FxGraphCache is populated.
