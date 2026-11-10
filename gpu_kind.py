"""GPU kind names shared by the harnesses and the scripts/ tools.

A root module (listed in pyproject py-modules) so that an installed
oracle or repro can import it; scripts/gpu_lock.py re-exports it.
"""

from __future__ import annotations

# GB300 before B300: "NVIDIA GB300" contains "B300".
KNOWN_DEVICE_KINDS = ("GB300", "B300", "B200", "H200", "H100", "A100", "V100")


def device_kind_from_name(name: str) -> str:
    for token in KNOWN_DEVICE_KINDS:
        if token.lower() in name.lower():
            return token
    return name.strip().replace(" ", "_") or "UNKNOWN"
