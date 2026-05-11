import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# ========= KONFIG =========
INPUT_FILE = Path("/hpc/gpfs2/home/u/boehrijo/final_runs/run1/clean/test/CPAP-014-3/TITeit.npz")  # anpassen!
OUTPUT_DIR = Path("")
MODALITY = "TIT"  # oder "TIT"
FRAME_IDX = 3000
# =========================


def save_png(arr, path):
    arr = arr.copy()
    arr = np.nan_to_num(arr, nan=0.0)

    arr_min = arr.min()
    arr_max = arr.max()

    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr = np.zeros_like(arr)

    arr = (arr * 255).astype(np.uint8)
    plt.imsave(path, arr, cmap="viridis")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(INPUT_FILE)

    frames = data["EITdata"]
    frame = frames[FRAME_IDX]

    print("Shape:", frame.shape)
    print("NaNs:", np.isnan(frame).sum())
    print("Min/Max:", np.nanmin(frame), np.nanmax(frame))

    # ===== BEFORE =====
    save_png(frame, OUTPUT_DIR / "before_nan.png")

    # ===== NaN MASK =====
    nan_mask = np.isnan(frame).astype(np.uint8) * 255
    plt.imsave(OUTPUT_DIR / "nan_mask.png", nan_mask, cmap="viridis")

    # ===== FAKE STATS (wichtig!) =====
    # Hier kannst du echte Stats einsetzen, falls vorhanden
    mean = np.nanmean(frame)
    std = np.nanstd(frame)

    # Schutz gegen Division durch 0
    std = max(std, 1e-6)

    print("Mean:", mean, "Std:", std)

    # ===== NORMALIZATION =====
    norm = frame.copy()
    mask = np.isfinite(norm)

    norm[mask] = (norm[mask] - mean) / std
    norm[~mask] = 0.0

    # ===== AFTER =====
    save_png(norm, OUTPUT_DIR / "after_nan.png")

    print(f"Saved images to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()