"""Device management utilities."""

from __future__ import annotations

import torch


def get_device(device_str: str) -> torch.device:
    """Return the torch device for the provided string."""
    return torch.device(device_str)


def get_available_devices() -> list[str]:
    """List available devices in preference order."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices
