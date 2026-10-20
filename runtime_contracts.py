"""Model-free validation shared by the editor and protection pipeline."""
from __future__ import annotations

import math


MODES = ("phantomseal_faceshield", "phantomseal", "faceshield", "nullswap")
DEFAULTS = {
    "mode": "phantomseal_faceshield",
    "max_size": 1280,
    "iterations": 30,
    "noise_clamp": 12,
    "step_size": 1,
    "phantom_iterations": 1000,
    "nullswap_strength": 1.0,
    "nullswap_seed": 0,
}
PARAMETER_LIMITS = {
    "max_size": (128, 1280, True),
    "iterations": (1, 200, True),
    "noise_clamp": (1, 32, True),
    "step_size": (.25, 4, False),
    "phantom_iterations": (10, 2000, True),
    "nullswap_strength": (0, 4, False),
    "nullswap_seed": (0, 2**32 - 1, True),
}


def number(value, name, low, high, integer=False):
    """Reject booleans, strings and nonfinite values before model loading."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name}: a finite number is required.")
    if not low <= value <= high or integer and int(value) != value:
        raise ValueError(f"{name}: expected {'an integer' if integer else 'a number'} between {low} and {high}.")
    return int(value) if integer else float(value)


def protection_parameters(raw):
    if not isinstance(raw, dict):
        raise ValueError("params must be an object.")
    mode = raw.get("mode", DEFAULTS["mode"])
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError("mode must be " + ", ".join(MODES) + ".")
    return {
        "mode": mode,
        **{name: number(raw.get(name, DEFAULTS[name]), name, *limits)
           for name, limits in PARAMETER_LIMITS.items()},
    }


def nullswap_parameters(strength=1.0, seed=0):
    return (number(strength, "nullswap_strength", *PARAMETER_LIMITS["nullswap_strength"]),
            number(seed, "nullswap_seed", *PARAMETER_LIMITS["nullswap_seed"]))


def analysis_size(value):
    size = number(value, "analysis_size", 64, 1280, integer=True)
    if size % 64:
        raise ValueError("analysis_size must be a multiple of 64 (normally 512).")
    return size
