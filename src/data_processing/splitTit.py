import logging
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks

logger = logging.getLogger("SplitTIT")

class SplitTit:
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

            eit = signal["EITdata"]
            pres = signal["pressureData"]
            vol = signal["volumeData"]

            pre = pres[0,:]
            step_point = [0]

            peaks, _ = find_peaks(-pre, height=-27, prominence=5)

            for i in range(len(peaks) - 1):
                if pre[peaks[i]] - pre[peaks[i+1]] >= 1:
                    step_point.append(peaks[i])
            step_point.append(len(pre)-1)                    

            eit_sig = []
            pres_sig = []
            vol_sig = []

            for j in range(len(step_point) - 1):
                if j % 2 == 0:
                    start = step_point[j]
                    end = step_point[j + 1]

                    if end <= start:
                        continue

                    a = eit[start:end]
                    b = pres[:, start:end]
                    c = vol[:, start:end]
                    if len(a) > 0 and len(b) > 0 and len(c) > 0:
                        eit_sig.append(a)
                        pres_sig.append(b)
                        vol_sig.append(c)
                    
                    else:
                        logger.warning(f"No valid EIT segment found: {path}")

            eit_sig = np.concatenate(eit_sig, axis=0)
            pres_sig = np.concatenate(pres_sig, axis=1)
            vol_sig = np.concatenate(vol_sig, axis=1)
            
            self.save_npz(eit_sig, pres_sig, vol_sig, p,"TITeit")

    def save_npz(self, arr:np.ndarray, pData:np.ndarray, vData:np.ndarray, path:Path, modality:str) -> None:
        """
        Saves the arrays in an npz file. For every patient it own folder and file
        
        Input:
        arr : np.ndarray
            picture data
        pData : np.ndarray
            pressure data
        vData : np.ndarray
            volume data
        w : str
            patient id
        modality : str
            dataset identifier ("TITeit" or "MLFeit")
        """
        id = path.name
        save_path = path.parent.parent.parent  / 'half_tit' / path.parent.name / id
        save_path.mkdir(parents = True, exist_ok = True)
        np.savez_compressed(
            save_path / f"{modality}.npz",
            EITdata=arr,
            pressureData=pData,
            volumeData=vData,
        )
        (save_path / "MLFeit.npz").write_bytes((self.path / path.parent.name / id / "MLFeit.npz").read_bytes())
        (save_path / "table.npz").write_bytes((self.path / path.parent.name / id / "table.npz").read_bytes())

def entrypoint():
    import argparse

    parser = argparse.ArgumentParser(description="path for input")

    parser.add_argument(
        "--path",
        help="Path to parent directory of the input .npz files",
        required=True,
    )


    args = parser.parse_args()

    SplitTit(
        path=args.path,
    ).run()

if __name__ == "__main__":
    entrypoint()

