import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

root = Path("/hpc/gpfs2/home/u/boehrijo/final_runs")

data = {
    "mlf": [],
    "tit": [],
    "tab": []
}

for subdir in root.iterdir():
    if subdir.is_dir():
        file = subdir / "importance_results.json"
        
        if file.exists():
            with open(file, "r") as f:
                d = json.load(f)
            
            for split in d.values():          # full_tit, third_tit, half_tit
                for method in split.values(): # VP, 3D
                    for model in method.values(): # RF
                        data["mlf"].append(model["mlf"])
                        data["tit"].append(model["tit"])
                        data["tab"].append(model["tab"])

stats = {}
for key in data:
    values = np.array(data[key])
    stats[key] = {
        "mean": values.mean(),
        "std": values.std()
    }
    print(f"{key}: mean={stats[key]['mean']:.4f}, std={stats[key]['std']:.4f}")

labels = ["mlf", "tit", "tab"]
means = [stats[k]["mean"] for k in labels]
stds  = [stats[k]["std"] for k in labels]

plt.figure()
plt.bar(labels, means, yerr=stds, capsize=5)
plt.xlabel("Metric")
plt.ylabel("Value")
plt.title("Mean and Standard Deviation")
plt.tight_layout()
plt.show()