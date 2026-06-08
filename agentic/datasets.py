"""Geometry-agnostic dataset registry.

Each entry describes WHERE a target's raw field files live and HOW to turn the
per-file ordering into a frequency coordinate. Nothing here is specific to the
surrogate model — adding a brand-new geometry is a one-entry change, which is
exactly the generalization story: the agents adapt to whatever this points at.

A dataset "spec" (a plain dict, serialized to dataset.json inside each run dir)
is what the agents actually consume. Build it with `build_spec(name)`.
"""

import re
from pathlib import Path

# agentic/ lives directly under the repo root (CV_DeepONet/).
REPO_ROOT = Path(__file__).resolve().parent.parent

# ─── Registry ────────────────────────────────────────────────────────────────
# data_dir is relative to REPO_ROOT. freq_unit is either:
#   "GHz"   -> frequency coordinate = linspace(fmin, fmax, n_freqs)
#   "index" -> frequency coordinate = 0, 1, ..., n_freqs-1  (the original solver
#              only exposed an index for this target, so we keep it honest)
REGISTRY = {
    "sphere": {
        "data_dir": "sphere/data/pec",
        "e_glob": "eall_pec_f_*.dat",
        "h_glob": "hall_pec_f_*.dat",
        "freq_unit": "GHz",
        "fmin": 0.05,
        "fmax": 0.6,
        "description": "Metallic (PEC) sphere.",
    },
    "almond": {
        "data_dir": "almond/E-H_files",
        "e_glob": "eall_pec.*",
        "h_glob": "hall_pec.*",
        "freq_unit": "index",
        "description": "Metallic (PEC) NASA-almond target (has a sharp tip).",
    },
}


# ─── Natural-alphanumeric sort (so f_2 precedes f_10, .001 precedes .010) ────
def sorted_alphanumeric(items):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def key(s):
        return [convert(c) for c in re.split(r"([0-9]+)", s)]

    return sorted(items, key=key)


def _list_files(data_dir: Path, pattern: str):
    return sorted_alphanumeric([p.name for p in data_dir.glob(pattern)])


def build_spec(name: str) -> dict:
    """Resolve a registry entry into a concrete dataset spec.

    Counts the actual files on disk (does not trust a hardcoded count) and
    builds the frequency coordinate list.
    """
    if name not in REGISTRY:
        raise KeyError(f"Unknown geometry '{name}'. Known: {list(REGISTRY)}")
    entry = REGISTRY[name]
    data_dir = (REPO_ROOT / entry["data_dir"]).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    e_files = _list_files(data_dir, entry["e_glob"])
    h_files = _list_files(data_dir, entry["h_glob"])
    if not e_files or not h_files:
        raise FileNotFoundError(
            f"No data files matched in {data_dir} "
            f"(e_glob={entry['e_glob']!r}, h_glob={entry['h_glob']!r})"
        )
    if len(e_files) != len(h_files):
        raise ValueError(
            f"{name}: {len(e_files)} E-files vs {len(h_files)} H-files — must match."
        )
    n_freqs = len(e_files)

    if entry["freq_unit"] == "GHz":
        fmin, fmax = entry["fmin"], entry["fmax"]
        if n_freqs == 1:
            freqs = [fmin]
        else:
            step = (fmax - fmin) / (n_freqs - 1)
            freqs = [round(fmin + i * step, 10) for i in range(n_freqs)]
    else:  # "index"
        freqs = [float(i) for i in range(n_freqs)]

    return {
        "name": name,
        "description": entry["description"],
        "data_dir": str(data_dir),
        "e_files": e_files,
        "h_files": h_files,
        "n_freqs": n_freqs,
        "freq_unit": entry["freq_unit"],
        "freqs": freqs,
    }


def subsample_for_smoke(spec: dict, k: int = 10) -> dict:
    """Return a copy of the spec keeping only ~k frequency samples.

    Keeps the two endpoints (so the train split never has to extrapolate) and
    spreads the rest evenly. Used by `--smoke` to make an end-to-end run finish
    in minutes instead of hours.
    """
    n = spec["n_freqs"]
    if n <= k:
        return dict(spec)
    # Evenly spaced indices including both endpoints.
    idx = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
    out = dict(spec)
    out["e_files"] = [spec["e_files"][i] for i in idx]
    out["h_files"] = [spec["h_files"][i] for i in idx]
    out["freqs"] = [spec["freqs"][i] for i in idx]
    out["n_freqs"] = len(idx)
    out["smoke_subsampled_from"] = n
    return out
