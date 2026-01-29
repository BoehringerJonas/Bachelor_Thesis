import gc
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import scipy

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("clean")

class CleanZone:
    """
    Pipeline for preprocessing raw EIT, pressure, and volume data for each patient.

    This class implements a "silver layer" clean stage that standardizes
    multi-modal patient data, normalizes it using global statistics, and
    saves it in structured `.npz` files suitable for downstream machine
    learning models (e.g., CNNs or LSTMs).

    Pipeline Overview
    -----------------

    2. **Patient-wise Data Preprocessing**
       For each patient:
       - Load raw `.mat` files for EIT images, pressure, and volume.
       - Convert EIT images to shape `(T, H, W)` and standardize per pixel.

    3. **Saving Cleaned Data**
       - Saves processed data in a structured per-patient folder under `clean/`.
       - Each patient folder contains:
           * `MLMeit.npz` or `TITeit.npz`: normalized EIT, pressure, and volume arrays.
           * `table.npz`: patient metadata including age, BMI, sex, weight, height,
             smoking status, Ppeak, Pmean, and target PEEP.
       - Files are saved using `np.savez_compressed` to reduce storage usage.

    4. **Design Considerations**
       - Handles missing files gracefully by skipping them with a warning.
       - Uses global normalization to standardize across all patients.
       - Ensures that each patient has a dedicated folder for reproducible
         downstream training.
       - Supports both MLM (maximal flow maneuver) and TIT (titration) datasets.

    Attributes
    ----------
    input_path : Path
        Base directory containing raw EIT, pressure, volume, and tabular data.
    MLMmean, MLMstd, MLMpmean, MLMpstd, MLMvmean, MLMvstd : float
        Global mean and standard deviation for MLM images, pressure, and volume.
    TITmean, TITstd, TITpmean, TITpstd, TITvmean, TITvstd : float
        Global mean and standard deviation for TIT images, pressure, and volume.
    """
    def __init__(self, 
        input_path : Path, 
        ) -> None:
        """
        Init global mean and std for MLM and TIT bevor running the clean stage.
        """
        self.input_path = input_path


    def run(self) -> None:
        """
        Read .mat files and preprozess them and sve them to npz files.

        raw -> clean

        |- clean
            |- MLMeit.npz
            |- table.npz
            |- TITeit.npz
        """
        csv_path = self.input_path/ 'raw' / 'MLM' / 'peep_tabular_data.csv'
        table = pd.read_csv(csv_path)
        table = table[~table["Pat-ID"].str.startswith("AD")]
        table = table.set_index("Pat-ID")
        for id in table.index:
            logger.info("Starting itteration for: " + str(id))

            path = self.input_path / 'raw' / 'MLM' / id / f"{id}_images_part_sb.mat"

            if not path.exists():
                logger.warning(f"Missing file, skipping: {path}")
                continue
                
            MLMmat = scipy.io.loadmat(path)
            key = list(MLMmat.keys())[3]
            MLM = np.transpose(MLMmat[key], (2,0,1)).astype(np.float32)
            MLM = np.nan_to_num(MLM, nan=0.0)

            pMLMmat = scipy.io.loadmat(self.input_path / 'raw' / 'MLM' / id / f"{id}_pressure_filtered_sb.mat")
            pkey = list(pMLMmat.keys())[3]
            pMLM = pMLMmat[pkey].astype(np.float32) 
            vMLMmat = scipy.io.loadmat(self.input_path / 'raw' / 'MLM' / id / f"{id}_volume_sb.mat")
            vkey = list(vMLMmat.keys())[3]
            vMLM = vMLMmat[vkey].astype(np.float32) 
        
            logger.info("Save MLM")
            self.save_npz(MLM, pMLM, vMLM, id, 'MLMeit')
            
            del MLM, MLMmat, pMLMmat, vMLMmat, pkey, vkey, key

            TITmat = scipy.io.loadmat(self.input_path / 'raw' / 'Titration' / id / f"{id}_images_part_tit.mat")
            key = list(TITmat.keys())[3]
            TIT = np.transpose(TITmat[key], (2,0,1)).astype(np.float32)
            TIT = np.nan_to_num(TIT, nan=0.0)
            
            pTITmat = scipy.io.loadmat(self.input_path / 'raw' / 'Titration' / id / f"{id}_pressure_filtered_tit.mat")
            pkey = list(pTITmat.keys())[3]
            pTIT = pTITmat[pkey].astype(np.float32)
            vTITmat = scipy.io.loadmat(self.input_path / 'raw' / 'Titration' / id / f"{id}_volume_tit.mat")
            vkey = list(vTITmat.keys())[3]
            vTIT = vTITmat[vkey].astype(np.float32)


            logger.info("Save TIT")
            self.save_npz(TIT, pTIT, vTIT, id, 'TITeit')
            del TITmat, TIT, pTITmat, vTITmat, pkey, vkey, key
            

            logger.info("save table")

            sex_map = {"male": 0, "female": 1}
            smoke_map = {
                "Never":   [1, 0, 0],
                "Former":  [0, 1, 0],
                "Current": [0, 0, 1]
                }

            save_path = self.input_path / 'clean' / id
            save_path.mkdir(parents = True, exist_ok = True)
            row = table.loc[id]
            np.savez(
                save_path / "table.npz",
                sex=sex_map.get(row["sex"]),
                age=row["age"],
                BMI=row["BMI"],
                weight=row["weight (kg)"],
                height=row["height (cm)"],
                smoke=smoke_map.get(row["smoke"]),
                targetPEEP=row["PEEP"]
            )

    def save_npz(self, arr:np.ndarray, pData:np.ndarray, vData:np.ndarray, id:str, modality:str) -> None:
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
            dataset identifier ("TITeit" or "MLMeit")
        """
        save_path = Path(self.input_path) / 'clean' / id
        save_path.mkdir(parents = True, exist_ok = True)
        np.savez_compressed(
            save_path / f"{modality}.npz",
            EITdata=arr,
            pressureData=pData,
            volumeData=vData,
        )

