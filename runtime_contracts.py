"""Model-free validation shared by the editor and protection pipeline."""
from __future__ import annotations

import math


MODES = ("phantomseal_faceshield", "phantomseal", "faceshield", "nullswap", "custom")
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
STAGE_DEFAULTS = {
    "nullswap": {"strength": 1.0, "seed": 0},
    "phantomseal": {"iterations": 1000},
    "faceshield": {"iterations": 30, "noise_clamp": 12, "step_size": 1.0, "analysis_size": 512},
}
STAGE_PARAMETER_LIMITS = {
    "nullswap": {"strength": (0, 4, False), "seed": (0, 2**32 - 1, True)},
    "phantomseal": {"iterations": (10, 2000, True)},
    "faceshield": {"iterations": (1, 200, True), "noise_clamp": (1, 32, True),
                   "step_size": (.25, 4, False), "analysis_size": (64, 1280, True)},
}
MAX_PIPELINE_STAGES = 8


def number(value, name, low, high, integer=False):
    """Reject booleans, strings and nonfinite values before model loading."""
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or isinstance(value, float) and not math.isfinite(value)):
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
    params = {
        "mode": mode,
        **{name: number(raw.get(name, DEFAULTS[name]), name, *limits)
           for name, limits in PARAMETER_LIMITS.items()},
    }
    if mode == "custom":
        params["stages"] = pipeline_stages(raw.get("stages"))
    elif "stages" in raw:
        raise ValueError("stages may only be supplied when mode is custom.")
    return params


def pipeline_stages(raw):
    """Validate and copy an ordered recipe without importing model runtimes."""
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_PIPELINE_STAGES:
        raise ValueError(f"stages must be a list of 1 to {MAX_PIPELINE_STAGES} stages.")
    normalized = []
    for index, stage in enumerate(raw, start=1):
        label = f"stages[{index - 1}]"
        if not isinstance(stage, dict):
            raise ValueError(f"{label} must be an object.")
        if set(stage) - {"type", "params"}:
            raise ValueError(f"{label}: unknown stage fields.")
        kind = stage.get("type")
        if not isinstance(kind, str) or kind not in STAGE_DEFAULTS:
            raise ValueError(f"{label}.type must be nullswap, phantomseal, or faceshield.")
        raw_params = stage.get("params", {})
        if not isinstance(raw_params, dict):
            raise ValueError(f"{label}.params must be an object.")
        if set(raw_params) - set(STAGE_DEFAULTS[kind]):
            raise ValueError(f"{label}.params: unknown parameters for {kind}.")
        params = {name: number(raw_params.get(name, default), f"{label}.params.{name}",
                               *STAGE_PARAMETER_LIMITS[kind][name])
                  for name, default in STAGE_DEFAULTS[kind].items()}
        if kind == "faceshield":
            params["analysis_size"] = analysis_size(params["analysis_size"])
        normalized.append({"type": kind, "params": params})
    return normalized


def nullswap_parameters(strength=1.0, seed=0):
    return (number(strength, "nullswap_strength", *PARAMETER_LIMITS["nullswap_strength"]),
            number(seed, "nullswap_seed", *PARAMETER_LIMITS["nullswap_seed"]))


def analysis_size(value):
    size = number(value, "analysis_size", 64, 1280, integer=True)
    if size % 64:
        raise ValueError("analysis_size must be a multiple of 64 (normally 512).")
    return size
