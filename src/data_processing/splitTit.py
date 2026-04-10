import logging
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks
import time

logger = logging.getLogger("workflow")

class SplitTit:
    def __init__(self, path:Path):
        self.path = Path(path)

    def run(self) -> None:
        """
        Takes the Titration data of each patient, detects all low peaks with a drop of over 1 cmH2O
        to the next peak. The titration gets devidet into half and third of the length by excluding 
        levels of the graph.
        """
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

            half_eit, half_pres, half_vol = [], [], []
            third_eit, third_pres, third_vol = [], [], []
             

            for j in range(len(step_point) - 1):
                start = step_point[j]
                end = step_point[j + 1]
                if end <= start:
                    continue

                if j % 2 == 0:
                    a = eit[start:end]
                    b = pres[:, start:end]
                    c = vol[:, start:end]
                    if len(a) > 0 and len(b) > 0 and len(c) > 0:
                        half_eit.append(a)
                        half_pres.append(b)
                        half_vol.append(c)
                    
                    else:
                        logger.warning(f"No valid EIT segment found: {path}")

                if j % 3 == 0:
                    a = eit[start:end]
                    b = pres[:, start:end]
                    c = vol[:, start:end]
                    if len(a) > 0 and len(b) > 0 and len(c) > 0:
                        third_eit.append(a)
                        third_pres.append(b)
                        third_vol.append(c)
                    
                    else:
                        logger.warning(f"No valid EIT segment found: {path}")


            half_eit = np.concatenate(half_eit, axis=0)
            half_pres = np.concatenate(half_pres, axis=1)
            half_vol = np.concatenate(half_vol, axis=1)           
            self.save_npz(half_eit, half_pres, half_vol, p,"half_tit")

            third_eit = np.concatenate(third_eit, axis=0)
            third_pres = np.concatenate(third_pres, axis=1)
            third_vol = np.concatenate(third_vol, axis=1)
            self.save_npz(third_eit, third_pres, third_vol, p,"third_tit")


    def save_npz(self, eitData:np.ndarray, pData:np.ndarray, vData:np.ndarray, path:Path, 
                modality:str) -> None:
        """
        Saves the data in an npz file. For every patient its own folder and file, titeled by 
        half_tit or third_tit.

        Parameters
        ----------
        eitData : np.ndarray
            picture data
        pData : np.ndarray
            pressure data
        vData : np.ndarray
            volume data
        path : str
            path to full titration folder
        modality : str
            dataset identifier ("half_tit" or "third_tit")
        """
        id = path.name
        save_path = path.parent.parent.parent  / modality / path.parent.name / id
        save_path.mkdir(parents = True, exist_ok = True)
        np.savez_compressed(
            save_path / "TITeit.npz",
            EITdata=eitData,
            pressureData=pData,
            volumeData=vData,
        )
        (save_path / "MLFeit.npz").write_bytes((self.path / path.parent.name / id / "MLFeit.npz").read_bytes())
        (save_path / "table.npz").write_bytes((self.path / path.parent.name / id / "table.npz").read_bytes())

def entrypoint():
    """
    Starting point for the file.
    Parameters
    ----------
    --path    : str
        Takes the path to the parent directory of the input .npz files
    """
    import argparse

    parser = argparse.ArgumentParser(description="Path to parent directory of the input .npz files")

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

