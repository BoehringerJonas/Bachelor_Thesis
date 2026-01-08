import gc
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple
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
    1. **Global Statistics Calculation**
       - Iterates over all patients in a dataset (MLM or TIT).
       - Computes per-pixel mean and standard deviation for EIT images.
       - Computes mean and standard deviation for pressure and volume time series.
       - These scalers are used for normalization of each patient's data.

    2. **Patient-wise Data Preprocessing**
       For each patient:
       - Load raw `.mat` files for EIT images, pressure, and volume.
       - Normalize using the global statistics computed in step 1.
       - Convert EIT images to shape `(T, H, W)` and standardize per pixel.
       - Normalize pressure and volume series independently.

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
        self.MLMmean, self.MLMstd, self.MLMpmean, self.MLMpstd, self.MLMvmean, self.MLMvstd = self.get_global("MLM")
        self.TITmean, self.TITstd, self.TITpmean, self.TITpstd, self.TITvmean, self.TITvstd = self.get_global("Titration")


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
            logger.info("Starting itteration for: " + str(table.loc[id, "PEEP"]))

            path = self.input_path / 'raw' / 'MLM' / id / f"{id}_images_part_sb.mat"

            if not path.exists():
                logger.warning(f"Missing file, skipping: {path}")
                continue
                
            MLMmat = scipy.io.loadmat(path)
            key = list(MLMmat.keys())[3]
            MLM = self.get_norm(MLMmat[key], "MLM")

            pMLMmat = scipy.io.loadmat(self.input_path / 'raw' / 'MLM' / id / f"{id}_pressure_sb.mat")
            pkey = list(pMLMmat.keys())[3]
            pMLM = (pMLMmat[pkey] - self.MLMpmean) / (self.MLMpstd + 1e-6)

            vMLMmat = scipy.io.loadmat(self.input_path / 'raw' / 'MLM' / id / f"{id}_volume_sb.mat")
            vkey = list(vMLMmat.keys())[3]
            vMLM = (vMLMmat[vkey] - self.MLMvmean) / (self.MLMvstd + 1e-6)
        
            logger.info("Save MLM")
            self.save_npz(MLM, pMLM, vMLM, id, 'MLMeit')
            
            del MLM, MLMmat, pMLMmat, vMLMmat, pkey, vkey, key

            TITmat = scipy.io.loadmat(self.input_path / 'raw' / 'Titration' / id / f"{id}_images_part_tit.mat")
            key = list(TITmat.keys())[3]
            TIT = self.get_norm(TITmat[key], "TIT")
            
            pTITmat = scipy.io.loadmat(self.input_path / 'raw' / 'Titration' / id / f"{id}_pressure_tit.mat")
            pkey = list(pTITmat.keys())[3]
            pTIT = (pTITmat[pkey] - self.TITpmean) / (self.TITpstd + 1e-6)

            vTITmat = scipy.io.loadmat(self.input_path / 'raw' / 'Titration' / id / f"{id}_volume_tit.mat")
            vkey = list(vTITmat.keys())[3]
            vTIT = (vTITmat[vkey] - self.TITvmean) / (self.TITvstd + 1e-6)


            logger.info("Save TIT")
            self.save_npz(TIT, pTIT, vTIT, id, 'TITeit')
            del TITmat, TIT, pTITmat, vTITmat, pkey, vkey, key

            logger.info("save table row for this patient (Pat_ID, sex, age, BMI, weight, height, smoke, Ppeak, Pmean, PEEP)")
            save_path = Path(self.input_path) / 'clean' / id
            save_path.mkdir(parents = True, exist_ok = True)
            row = table.loc[id]
            np.savez(
                save_path / "table.npz",
                Pat_ID=id,
                sex=row["sex"],
                age=row["age"],
                BMI=row["BMI"],
                weight_kg=row["weight (kg)"],
                height_cm=row["height (cm)"],
                smoke=row["smoke"],
                Ppeak=row["Ppeak"],
                Pmean=row["Pmean"],
                targetPEEP=row["PEEP"]
            )
        logger.info("Finish Cleanstage")

    def get_norm(self, arr:np.ndarray, modality:str) -> np.ndarray:
        """
        Takes a raw picture array and normalized the pixel for the input into a resnet50 CNN.
        
        Input:
        arr : np.ndarray
        modality : str
            dataset identifier ("TIT" or "MLM")

        Returns:       
        arr : np.ndarray
        """

        arr = np.transpose(arr, (2,0,1)).astype(np.float32)
        if modality == "MLM":
            arr = (arr - self.MLMmean) / (self.MLMstd + 1e-6)
        else:
            arr = (arr - self.TITmean) / (self.TITstd + 1e-6)
        return arr


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

    def get_global(self, modality:str) -> Tuple[np.float32, np.float32, np.float32, np.float32, np.float32, np.float32]:
        """
        Iterrating over all raw data picture arrays for the longest breath or for titration. 
        Collect total sum, total square sum and count the total number of pixels.
        Returning the per pixel mean and standart diviation
        
        Input:
        modality : str
            dataset identifier ("MLM", "TIT")

        Returns:
        arr_mean : numpy.float32
            global scaler
        arr_std : numpy.float32
            global scalar
        p_mean : numpy.float32
            global scaler
        p_std : numpy.float32
            global scalar
        v_mean : numpy.float32
            global scaler
        v_std : numpy.float32
            global scalar
        """
        logger.info("Start get_global for: " + modality)

        spez = "sb" if modality == "MLM" else "tit"

        arr_sum = p_sum = v_sum = 0.0
        arr_sq_sum = p_sq_sum = v_sq_sum = 0.0
        arr_count = p_count = v_count =0

        table = pd.read_csv(self.input_path/'raw'/'MLM'/'peep_tabular_data.csv')
        table = table[~table["Pat-ID"].str.startswith("AD")]

        for id in table['Pat-ID']:
            path = self.input_path/'raw'/modality/id/f"{id}_images_part_{spez}.mat"

            if not path.exists():
                logger.warning(f"Missing file, skipping: {path}")
                continue

            mat = scipy.io.loadmat(path)
            key = list(mat.keys())[3]

            arr_sum += mat[key].sum()
            arr_sq_sum += np.square(mat[key]).sum()
            arr_count += mat[key].size

            del mat, key

            mat = scipy.io.loadmat(
                self.input_path/'raw'/modality/id/f"{id}_pressure_{spez}.mat"
            )
            key = list(mat.keys())[3]

            p_sum += mat[key].sum()
            p_sq_sum += np.square(mat[key]).sum()
            p_count += mat[key].size

            del mat, key

            mat = scipy.io.loadmat(
                self.input_path/'raw'/modality/id/f"{id}_volume_{spez}.mat"
            )
            key = list(mat.keys())[3]

            v_sum += mat[key].sum()
            v_sq_sum += np.square(mat[key]).sum()
            v_count += mat[key].size

            del mat, key

        arr_mean = arr_sum / arr_count
        arr_var = arr_sq_sum / arr_count - arr_mean**2
        arr_var = np.maximum(arr_var, 0.0)
        arr_std = np.sqrt(arr_var) + 1e-6

        p_mean = p_sum / p_count
        p_var = p_sq_sum / p_count - p_mean**2
        p_var = np.maximum(p_var, 0.0)
        p_std = np.sqrt(p_var) + 1e-6

        v_mean = v_sum / v_count
        v_var = v_sq_sum / v_count - v_mean**2
        v_var = np.maximum(v_var, 0.0)
        v_std = np.sqrt(v_var) + 1e-6

        return np.float32(arr_mean), np.float32(arr_std), np.float32(p_mean), np.float32(p_std), np.float32(v_mean), np.float32(v_std)
