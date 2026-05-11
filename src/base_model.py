import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
# from scipy.stats import spearmanr, pearsonr, kendalltau
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors   # retained: used for TwoSlopeNorm below
# %matplotlib widget

# ── 1. Load the data ──────────────────────────────────────────────────────────
try:
    df = pd.read_csv("/hpc/gpfs2/home/u/boehrijo/final_runs/run1/raw/MLF/peep_tabular_data.csv") # baseline_peep.csv
except FileNotFoundError:
    raise FileNotFoundError(
        "The file 'peep_tabular_data.csv' was not found. Please check the path."
    )
except Exception as e:
    raise RuntimeError(f"An error occurred while loading the CSV: {e}") from e

# Define the target variable
target_variable = "PEEP"

if target_variable not in df.columns:
    raise ValueError(
        f"Target variable '{target_variable}' not found in the dataset. "
        f"Available columns: {list(df.columns)}"
    )

peep_values = df[target_variable].to_numpy(dtype=float)  # work with a plain NumPy array

# ── 2. Guard against degenerate dataset ──────────────────────────────────────
n = len(peep_values)
if n < 2:
    raise ValueError(
        f"LOOCV requires at least 2 observations; only {n} found."
    )

# ── 3. LOOCV – leave-one-out mean predictor ───────────────────
total_sum = peep_values.sum()

# Vectorised computation avoids the Python loop entirely
# Prediction for sample i = mean of all other (n-1) samples
prediction_arr = (total_sum - peep_values) / (n - 1)
test_arr       = peep_values                           # kept as alias for clarity

# ── 4. Metric calculation ─────────────────────────────────────────────────────
mse  = mean_squared_error(test_arr, prediction_arr)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(test_arr, prediction_arr)

# Report all p-values.
# spearman_r,  spearman_p  = spearmanr(test_arr, prediction_arr)
# pearson_r,   pearson_p   = pearsonr(test_arr,  prediction_arr)
# kendall_tau, kendall_p   = kendalltau(test_arr, prediction_arr)

# Print results – include p-values for scientific transparency
print(f"\n{'='*45}")
print(f"  LOOCV Baseline – Mean Predictor  (n={n})")
print(f"{'='*45}")
print(f"  MSE  : {mse:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  MAE  : {mae:.4f}")
# print(f"  Spearman r : {spearman_r:.4f}  (p={spearman_p:.4e})")
# print(f"  Pearson  r : {pearson_r:.4f}  (p={pearson_p:.4e})")
# print(f"  Kendall τ  : {kendall_tau:.4f}  (p={kendall_p:.4e})")
print(f"{'='*45}\n")

# ── 5. Visualization ──────────────────────────────────────────────────────────
residuals = test_arr - prediction_arr
max_abs_res = np.max(np.abs(residuals))
norm = mcolors.TwoSlopeNorm(vmin=-max_abs_res, vcenter=0, vmax=max_abs_res)

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")   # fallback for older Matplotlib

fig, ax = plt.subplots(figsize=(10, 8), dpi=100)

scatter = ax.scatter(
    test_arr,
    prediction_arr,
    c=residuals,
    cmap="coolwarm",
    norm=norm,          # FIX: apply symmetric normalisation
    s=40,
    alpha=0.75,
    edgecolor="white",
    linewidth=0.5,
    zorder=3,
)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
cbar.set_label("Residual (True − Predicted) [cm $H_2O$]", fontsize=12)
cbar.ax.tick_params(labelsize=10)

# The identity line (y = x) 
x_min, x_max = test_arr.min(), test_arr.max()
padding = (x_max - x_min) * 0.05
ref_vals = np.linspace(x_min - padding, x_max + padding, 200)
ax.plot(ref_vals, ref_vals, "k--", linewidth=1.5, alpha=0.8, label="Identity line (y = x)")

# Axis limits with padding guarantee the identity line is fully
# visible and consistent with `set_aspect('equal')`.
ax.set_xlim(x_min - padding, x_max + padding)
ax.set_ylim(x_min - padding, x_max + padding)

# Annotation box
metrics_text = ( 
    f"MSE : {mse:.2f}\n"
    f"RMSE : {rmse:.2f}\n"
    f"MAE  : {mae:.2f}\n"
    # f"Pearson r  : {pearson_r:.3f}\n"
    # f"Spearman r : {spearman_r:.3f}"
)
ax.text(
    0.05, 0.97, metrics_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray", boxstyle="round,pad=0.5"),
)

# Axes labels and title
ax.set_xlabel("True PEEP (cm $H_2O$)", fontsize=12, labelpad=10)
ax.set_ylabel("Predicted PEEP (cm $H_2O$)", fontsize=12, labelpad=10)
ax.set_title(
    "LOOCV Baseline (Leave-One-Out Mean): True vs Predicted PEEP",
    fontsize=13, pad=15, fontweight="bold",
)

# Equal aspect ratio – meaningful only when combined with explicit axis limits
ax.set_aspect("equal", adjustable="box")
ax.grid(True, linestyle="--", alpha=0.4, which="both")
ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=11)

plt.tight_layout()
plt.savefig("baseline_peep_loocv.png", dpi=150, bbox_inches="tight")  # save for reproducibility
plt.show()