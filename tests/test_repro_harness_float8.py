import pytest
import torch

from repro_harness import make_inputs_from_config


@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("stride", [None, [1, 3]])
def test_make_inputs_from_config_generates_float8_via_supported_dtype(
    dtype: str,
    stride: list[int] | None,
) -> None:
    (value,) = make_inputs_from_config(
        {
            "inputs": [
                {
                    "kind": "tensor",
                    "shape": [2, 3],
                    "dtype": dtype,
                    "stride": stride,
                    "device": "cpu",
                }
            ]
        }
    )

    assert value.dtype == getattr(torch, dtype)
    assert value.shape == (2, 3)
    assert value.stride() == ((3, 1) if stride is None else tuple(stride))
    assert torch.isfinite(value.float()).all()
