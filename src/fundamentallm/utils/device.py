"""Device management utilities."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def get_available_devices() -> list[str]:
    """List available devices in preference order."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def get_device(device_str: str) -> torch.device:
    """Return the torch device for the provided string.

    Args:
        device_str: Device identifier ("cpu", "cuda", "mps", or "auto").

    Returns:
        A torch.device instance.

    Raises:
        ValueError: If device is requested but not available (except "auto").
    """
    if device_str == "auto":
        best_device = get_best_device()
        logger.info(f"Auto-selected device: {best_device}")
        return torch.device(best_device)

    return torch.device(device_str)


def get_best_device() -> str:
    """Get the best available device with intelligent fallback.

    Priority order: cuda > mps > cpu

    Returns:
        Device name as string.
    """
    available = get_available_devices()
    # Return last (best) available device
    return available[-1]


def validate_device(device_str: str) -> str:
    """Validate and potentially fallback a device choice.

    Args:
        device_str: Requested device identifier.

    Returns:
        Validated device identifier (possibly different from input).

    Warns:
        If requested device is unavailable and fallback is used.
    """
    if device_str == "auto":
        return get_best_device()

    available = get_available_devices()

    # Special handling for cuda to check for specific issues
    if device_str == "cuda" and "cuda" in available:
        try:
            # Verify cuda is actually usable
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            # Quick sanity check
            _ = torch.zeros(1, device="cuda")
        except (RuntimeError, Exception) as e:
            logger.warning(
                f"CUDA requested but unavailable or misconfigured: {e}. " f"Falling back to CPU."
            )
            return "cpu"
        return device_str

    if device_str not in available:
        fallback = available[-1]
        logger.warning(
            f"Device '{device_str}' not available. "
            f"Available devices: {available}. "
            f"Using '{fallback}'."
        )
        return fallback

    return device_str


def get_device_info() -> dict[str, bool | int]:
    """Get information about available devices.

    Returns:
        Dictionary with device availability information.
    """
    return {
        "cpu_available": True,
        "cuda_available": torch.cuda.is_available(),
        "cuda_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": torch.backends.mps.is_available(),
    }
