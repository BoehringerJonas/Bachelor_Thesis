import logging
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

logger = logging.getLogger("SplitTIT")

class SplitTIT:
    def __init__(self, path:Path):
        self.path = Path(path)

    def run(self):
        print("start")
        files = list(self.path.rglob("TITeit.npz"))
        print(f"Found files: {len(files)}")

        for path in files:
            p = path.parent

            path = p / "TITeit.npz"

            if not path.exists():
                logger.warning(f"Missing file, skipping: {path}")
                continue

            signal = np.load(path)

            pre = signal["pressureData"][0,:]
            step_point = [0]
            peaks, _ = find_peaks(-pre, height=-27, prominence=5)

            for i in range(len(peaks)-1):
                if pre[peaks[i]] - pre[peaks[i+1]] >= 1:
                    step_point.append(peaks[i])
            step_point.append(len(pre)-1) 

            
            half_eit, half_pres, half_vol = [], [], []
            third_eit, third_pres, third_vol = [], [], []
             

            for j in range(len(step_point) - 1):
                start = step_point[j]
                end = step_point[j + 1]
                if end <= start:
                    continue

                if j % 2 == 0:
                    b = pre[start:end]
                    if b.size > 0:
                        half_pres.append(b)

                if j % 3 == 0:
                    b = pre[start:end]
                    if b.size > 0:
                        third_pres.append(b)

            half_pres = np.concatenate(half_pres, axis=0) if half_pres else np.array([])
            third_pres = np.concatenate(third_pres, axis=0) if third_pres else np.array([])


            fig, axes = plt.subplots(4, 1, figsize=(10, 8))

            manager = plt.get_current_fig_manager()

            try:
                manager.window.wm_geometry("+1800+500")
            except:
                try:
                    manager.window.move(1800, 500)
                except:
                    pass

            axes[0].plot(pre)
            axes[0].plot(peaks, pre[peaks], "rx")
            axes[0].set_title(f"low peaks for {p.name}")

            axes[1].plot(pre)
            axes[1].plot(step_point, pre[step_point], "rx")
            axes[1].set_title(f"splitt points for {p.name}")

            axes[2].plot(half_pres)
            axes[2].set_title(f"Half Shorten Titration for {p.name}")

            axes[3].plot(third_pres)
            axes[3].set_title(f"Third Shorten Titration for {p.name}")

            plt.tight_layout()
            plt.savefig(p / f"exampleTIT.png", bbox_inches='tight')
            plt.close()

            print("finish")

def entrypoint():
    import argparse

    parser = argparse.ArgumentParser(description="path for input")

    parser.add_argument(
        "--path",
        help="Path to parent directory of the input .npz files",
        required=True,
    )


    args = parser.parse_args()

    SplitTIT(
        path=args.path,
    ).run()

if __name__ == "__main__":
    entrypoint()

