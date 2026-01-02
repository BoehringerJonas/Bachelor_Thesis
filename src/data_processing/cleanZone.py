import gc
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Union, Tuple
import scipy

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("clean")

class CleanZone:
    def __init__(self, 
        input_path: Union[str, Path], 
        ) -> None:
        """
        Init global mean and std for MLM and TIT bevor running the clean stage.
        """
        self.input_path = input_path
        self.MLMmean, self.MLMstd = self.get_global("MLM")
        self.TITmean, self.TITstd = self.get_global("Titration")


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
        for w in table.index:
            logger.info("Starting itteration for: " + str(table.loc[w, "PEEP"]))
            
            MLMmat = scipy.io.loadmat(self.input_path / 'raw' / 'MLM' / w / f"{w}_images_part_sb.mat")
            key = list(MLMmat.keys())[3]
            MLM = self.get_norm(MLMmat[key], "MLM")

            pMLMmat = scipy.io.loadmat(self.input_path / 'raw' / 'MLM' / w / f"{w}_pressure_sb.mat")
            pkey = list(pMLMmat.keys())[3]

            vMLMmat = scipy.io.loadmat(self.input_path / 'raw' / 'MLM' / w / f"{w}_volume_sb.mat")
            vkey = list(vMLMmat.keys())[3]
            
            logger.info("Save MLM")
            self.save_npz(MLM, pMLMmat[pkey], vMLMmat[vkey], w, 'MLMeit')
            
            del MLM, MLMmat, pMLMmat, vMLMmat, pkey, vkey, key

            TITmat = scipy.io.loadmat(self.input_path / 'raw' / 'Titration' / w / f"{w}_images_part_tit.mat")
            key = list(TITmat.keys())[3]
            TIT = self.get_norm(TITmat[key], "TIT")
            
            pTITmat = scipy.io.loadmat(self.input_path / 'raw' / 'Titration' / w / f"{w}_pressure_tit.mat")
            pkey = list(pTITmat.keys())[3]

            vTITmat = scipy.io.loadmat(self.input_path / 'raw' / 'Titration' / w / f"{w}_volume_tit.mat")
            vkey = list(vTITmat.keys())[3]

            logger.info("Save TIT")
            self.save_npz(TIT, pTITmat[pkey], vTITmat[vkey], w, 'TITeit')
            del TITmat, TIT, pTITmat, vTITmat, pkey, vkey, key

            logger.info("save table row for this patient (Pat_ID, sex, age, BMI, weight, height, smoke, Ppeak, Pmean, PEEP)")
            save_path = Path(self.input_path) / 'clean' / w
            save_path.mkdir(parents = True, exist_ok = True)
            row = table.loc[w]
            np.savez(
                save_path / "table.npz",
                Pat_ID=w,
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
        
        Input
        arr : np.ndarray
        modality : str
            dataset identifier ("TIT" or "MLM")
        
        arr : np.ndarray
        """

        arr = np.transpose(arr, (2,0,1)).astype(np.float32)
        if modality == "MLM":
            arr = (arr - self.MLMmean) / (self.MLMstd + 1e-6)
        else:
            arr = (arr - self.TITmean) / (self.TITstd + 1e-6)
        return arr


    def save_npz(self, arr:np.ndarray, pData:np.ndarray, vData:np.ndarray, w:str, modality:str) -> None:
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
        save_path = Path(self.input_path) / 'clean' / w
        save_path.mkdir(parents = True, exist_ok = True)
        np.savez_compressed(
            save_path / f"{modality}.npz",
            EITdata=arr,
            pressureData=pData,
            volumeData=vData,
        )

    def get_global(self, modality:str) -> Tuple[np.float32, np.float32]:
        """
        Iterrating over all raw data picture arrays for the longest breath or for titration. 
        Collect total sum, total square sum and count the total number of pixels.
        Returning the per pixel mean and standart diviation
        
        Input:
        modality : str
            dataset identifier ("MLM", "TIT")

        Return:
        mean : numpy.float32
            global scaler
        std : numpy.float32
            global scalar
        """
        logger.info("Start get_global for: " + modality)

        spez = "sb" if modality == "MLM" else "tit"

        total_sum = 0.0
        total_sq_sum = 0.0
        total_count = 0

        table = pd.read_csv(self.input_path/'raw'/'MLM'/'peep_tabular_data.csv')
        table = table[~table["Pat-ID"].str.startswith("AD")]

        for w in table['Pat-ID']:
            mat = scipy.io.loadmat(
                self.input_path/'raw'/modality/w/f"{w}_images_part_{spez}.mat"
            )
            key = list(mat.keys())[3]
            arr = np.transpose(mat[key], (2,0,1)).astype(np.float64)

            total_sum += arr.sum()
            total_sq_sum += np.square(arr).sum()
            total_count += arr.size

            del mat, arr

        mean = total_sum / total_count
        var = total_sq_sum / total_count - mean**2
        std = np.sqrt(var) + 1e-6

        return np.float32(mean), np.float32(std)
