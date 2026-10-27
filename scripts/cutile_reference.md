# cuTile porting reference (Triton -> cuda.tile)

This is the reference for porting the Triton oracles in `repros/canonical/` to
cuTile oracles that live in the parallel mirror `repros_cutile/canonical/`.

## Setup

* Every dir in `repros_cutile/canonical/<name>/` already has symlinks for
  `repro.py`, `shapes.json`, and `meta.json`. You only need to write a new
  `oracle.py` in each such dir.
* Import cuTile as `import cuda.tile as ct`.
* Use `from oracle_harness import oracle_impl` for registration — identical
  usage to Triton oracles.

## Skeleton

```python
"""cuTile port of <name>: <one-line description>."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _my_kernel(x_ptr, out_ptr, N: ct.Constant[int], BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    ct.store(out_ptr, index=(pid,), tile=x + 1)


@oracle_impl(hardware="B200", point="<shape_hash>", BLOCK=<value>)
def oracle_forward(inputs, *, BLOCK):
    x = inputs[0]
    out = torch.empty_like(x)
    stream = torch.cuda.current_stream()
    ct.launch(stream, (ct.cdiv(x.numel(), BLOCK), 1, 1), _my_kernel,
              (x, out, x.numel(), BLOCK))
    return out
```

## Triton -> cuTile cheat sheet

| Triton                                              | cuTile                                                |
|-----------------------------------------------------|-------------------------------------------------------|
| `@triton.jit`                                       | `@ct.kernel`                                          |
| `tl.constexpr`                                      | `ct.Constant[int]`                                    |
| `tl.program_id(k)`                                  | `ct.bid(k)`                                           |
| `tl.arange(0, N)`                                   | `ct.arange(N, dtype=ct.int32)`                        |
| `tl.load(ptr + off, mask=m, other=v)`               | See load section below                                |
| `tl.store(ptr + off, x, mask=m)`                    | See store section below                               |
| `tl.sum(x, axis=k)`                                 | `ct.sum(x, axis=k)`                                   |
| `tl.max(x, axis=k)`                                 | `ct.max(x, axis=k)`                                   |
| `tl.where(c, a, b)`                                 | `ct.where(c, a, b)`                                   |
| `x.to(tl.float32)`                                  | `ct.astype(x, ct.float32)`                            |
| `tl.rsqrt(x)`                                       | `ct.rsqrt(x)`                                         |
| `libdevice.rsqrt(x)`                                | `ct.rsqrt(x)`                                         |
| `tl.exp(x), tl.log(x), tl.sqrt(x)`                  | `ct.exp(x), ct.log(x), ct.sqrt(x)`                    |
| `tl.full((N,), v, dtype)`                           | `ct.full((N,), v, dtype=dtype)`                       |
| `tl.zeros((N,), dtype)`                             | `ct.zeros((N,), dtype=dtype)`                         |
| `tl.trans(x)` (2d)                                  | `ct.transpose(x)`                                     |
| `tl.reshape(x, shape)` / `x.reshape(shape)`         | `ct.reshape(x, shape)`                                |
| `triton.cdiv(a, b)`                                 | `ct.cdiv(a, b)`                                       |

## Data types

`ct.bool_, ct.int8..int64, ct.uint8..uint64, ct.float16, ct.bfloat16,
ct.float32, ct.float64, ct.tfloat32, ct.float8_e4m3fn, ct.float8_e5m2`.

## Loads and stores

cuTile does not use raw pointer arithmetic. Instead, it partitions each
`Array` into a **tile space** and you index with tile-space coordinates.

Simplest pattern — 1D array, tile of length BLOCK:

```python
x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
ct.store(out_ptr, index=(pid,), tile=result)
```

The array shape is inferred from the torch tensor passed at launch. Given a
tensor of shape `(N,)`, the tile space partitioning by `(BLOCK,)` has
`cdiv(N, BLOCK)` tiles indexed by `pid`.

2D pattern — one row at a time:

```python
# For torch tensor of shape (rows, HIDDEN):
row = ct.bid(0)
x = ct.load(x_ptr, index=(row, 0), shape=(1, HIDDEN))
```

Multi-block tile — tile of shape `(BM, BH)`:

```python
row_block = ct.bid(0)  # walks rows in units of BM
col_block = ct.bid(1)  # walks cols in units of BH
x = ct.load(x_ptr, index=(row_block, col_block), shape=(BM, BH))
```

**Out-of-bounds:** cuTile pads OOB elements with `padding_mode` (default is
UNDETERMINED — undefined for OOB). Use `padding_mode=ct.PaddingMode.ZERO` to
pad with zero when the tile may extend past the array.

```python
x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,),
            padding_mode=ct.PaddingMode.ZERO)
```

**No mask on store:** cuTile's `store` writes the whole tile. If the tile
covers OOB elements, undefined writes occur. Two options:
1. Pick a `BLOCK` that divides the array size (then no OOB tile exists).
2. Use `load_advanced_indexing` + `scatter` for irregular writes.

## Reshape / transpose / permute

* `ct.reshape(x, shape)` — one element may be `-1` to be inferred.
* `ct.transpose(x)` — for 2D only. For >2D pass explicit axes.
* `ct.permute(x, (2, 0, 1))` — general permutation.
* `ct.expand_dims(x, axis)` — insert size-1 axis.

## Reductions

* `ct.sum(x, axis=None|k)`, `ct.max`, `ct.min`, `ct.prod`
* `ct.argmax(x, axis=k)`, `ct.argmin(x, axis=k)`
* `keepdims=True` to keep the reduced dim.
* No `mean` primitive — do `ct.sum(...) * (1.0 / N)`.

## Broadcasting

Broadcasting follows NumPy: align by trailing dims; 1 broadcasts. For an
outer-product-like combine of a `(BM,)` row-stat and a `(BM, BH)` tile:

```python
row_stat = ct.reshape(row_stat, (BM, 1))  # -> broadcasts to (BM, BH)
out = row_stat * tile2d
```

## Autotune / launch config

There is no automatic autotuner exposed as a decorator. Pick reasonable
BLOCK sizes based on the shape; you can call `.replace_hints(occupancy=..)`
on a `@ct.kernel` if needed, but the defaults are usually fine.

## Kernel-launch pattern (with PyTorch tensors)

```python
stream = torch.cuda.current_stream()
grid = (ct.cdiv(total_work, BLOCK), 1, 1)  # up to 3D grid
ct.launch(stream, grid, my_kernel, (arg0, arg1, ..., BLOCK))
```

The `grid` argument must always be a tuple of exactly 3 ints. Pad with 1.

## Common gotchas

1. **0-dimension tiles are NOT supported.** Use `shape=(1,)` and `.view(1)`.
2. **Tile shapes must be compile-time constants and powers of 2.** Round up
   as needed and mask/pad the excess.
3. **`ct.transpose(x)` for a 2D tile has NO argument.** Passing tuples
   raises a type error. Use `ct.permute(x, axes)` for higher rank.
4. **`ct.zeros((N,))` requires `dtype=...`.** Same for `ct.full`.
5. **The `index=` param of `ct.load` is a tile-space index, not an element
   offset.** For a 1D array of length N loaded in tiles of BLOCK, `pid` is
   the tile-space index; no `* BLOCK` multiplication.
6. **cuTile lacks Triton's `.to(dtype, fp_downcast_rounding=...)`.** Use
   `ct.astype(x, ct.bfloat16)`; rounding is round-to-nearest by default.
7. **cuTile lacks `tl.rand` / seeded on-device RNG.** For oracles that use
   `tl.rand`, generate the random tensor with `torch.ops.prims.inductor_random`
   ahead of the kernel and pass it as an input.
8. **The Triton oracle usually declares `num_warps=...` in `@oracle_impl`.**
   cuTile does not need this — omit those kwargs from the registration.
9. **Strided input tensors are fine** as long as `.contiguous()` isn't
   forced; cuTile respects strides via its `Array` view of the passed tensor.
10. **`torch.empty_strided` is fine** for shaping outputs to match the
    Repro's expected strides.

## Registration

Copy the `@oracle_impl(hardware="B200", point="<hash>", ...)` decorators
from the Triton oracle verbatim. **DROP** any `num_warps=`, `num_stages=`
kwargs — cuTile doesn't take them, and passing them will cause a runtime
error when the oracle is dispatched (as they'd be passed to `oracle_forward`).

Keep only kwargs that YOUR `oracle_forward` accepts (e.g. `BLOCK`,
`BLOCK_M`, `BLOCK_H`, etc.).

## When a port is impossible

Some Triton oracles use features cuTile does not have (e.g. seeded on-device
RNG, `tl.inline_asm_elementwise` custom PTX, atomic scatter with masks,
grouped-argmax bit-packing). In those cases, write an oracle stub that
raises `TileUnsupportedFeatureError` at dispatch:

```python
"""cuTile port UNSUPPORTED: <reason>."""

from oracle_harness import oracle_impl


@oracle_impl(hardware="B200", point="<hash>")
def oracle_forward(inputs):
    raise NotImplementedError(
        "cuTile port unsupported: <one-line reason>"
    )
```

The bench harness records this as `NO_ORACLE_FOR_SHAPE` and it's excluded
from the aggregate. Do NOT copy the Triton kernel as-is — that would report
Triton results under the "cuTile" label.
