import numpy as np
from pathlib import Path

"""
Validates a stats.npz file for NaN/Inf values and basic sanity.

Parameters
----------
stats_path : Path
    Path to stats.npz file
strict : bool
    If True -> raises RuntimeError on invalid values
    If False -> only logs warnings
"""
stats_path = Path("/hpc/gpfs2/home/u/boehrijo/run6/feature/full_tit/stats/train_stats.npz")
strict = False
if not stats_path.exists():
    raise FileNotFoundError(f"Stats file not found: {stats_path}")

stats = np.load(stats_path)

has_error = False

for k in stats.files:
    arr = stats[k]

    if not np.isfinite(arr).all():
        has_error = True
        msg = (
            f"[INVALID] {k} contains NaN/Inf | "
            f"min={np.nanmin(arr)}, max={np.nanmax(arr)}"
        )

        if strict:
            raise RuntimeError(msg)
        else:
            print(msg)

    # optional: zusätzliche Checks
    if "std" in k and (arr <= 0).any():
        has_error = True
        msg = f"[INVALID] {k} has non-positive std values"

        if strict:
            raise RuntimeError(msg)
        else:
            print(msg)

if not has_error:
    print(f"Stats file OK: {stats_path}")