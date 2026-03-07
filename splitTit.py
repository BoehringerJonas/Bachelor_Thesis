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
        pat_folders = sorted(
            p for p in self.path.rglob("*")
            if p.is_dir() and not any(c.is_dir() for c in p.iterdir()))
        
        for idx, p in enumerate(pat_folders):

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

            new_sig = []
            for j in range(len(step_point)-1):
                if j % 2 == 0:
                    new_sig.extend(pre[step_point[j]:step_point[j+1]])


            fig, axes = plt.subplots(3, 1, figsize=(10, 8))

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

            axes[2].plot(new_sig)
            axes[2].set_title(f"Shorten Titration for {p.name}")

            plt.tight_layout()
            plt.show()

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

