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

    def __init__(self, 
        input_path : Path, 
        ) -> None:
        """
        Init global mean and std for MLF and TIT bevor running the clean stage.
        """
        self.input_path = input_path


    def run(self) -> None:
        """
        Read .mat files and preprozess them and sve them to npz files.

        raw -> clean

        |- clean
            |- MLFeit.npz
            |- table.npz
            |- TITeit.npz
        """
        csv_path = self.input_path/ 'raw' / 'MLF' / 'peep_tabular_data.csv'
        table = pd.read_csv(csv_path)
        table = table[~table["Pat-ID"].str.startswith("AD")]
        table = table.set_index("Pat-ID")
        for id in table.index:
            logger.info("Starting itteration for: " + str(id))

            path = self.input_path / 'raw' / 'MLF' / id / f"{id}_images_part_sb.mat"

            if not path.exists():
                logger.warning(f"Missing file, skipping: {path}")
                continue
                
            MLFmat = scipy.io.loadmat(path)
            key = list(MLFmat.keys())[3]
            MLF = np.transpose(MLFmat[key], (2,0,1)).astype(np.float32)
            #MLF = np.nan_to_num(MLF, nan=0.0)

            pMLFmat = scipy.io.loadmat(self.input_path / 'raw' / 'MLF' / id / f"{id}_pressure_filtered_sb.mat")
            pkey = list(pMLFmat.keys())[3]
            pMLF = pMLFmat[pkey].astype(np.float32) 
            vMLFmat = scipy.io.loadmat(self.input_path / 'raw' / 'MLF' / id / f"{id}_volume_sb.mat")
            vkey = list(vMLFmat.keys())[3]
            vMLF = vMLFmat[vkey].astype(np.float32) 
        
            logger.info("Save MLF")
            self.save_npz(MLF, pMLF, vMLF, id, 'MLFeit')
            
            del MLF, MLFmat, pMLFmat, vMLFmat, pkey, vkey, key

            TITmat = scipy.io.loadmat(self.input_path / 'raw' / 'Titration' / id / f"{id}_images_part_tit.mat")
            key = list(TITmat.keys())[3]
            TIT = np.transpose(TITmat[key], (2,0,1)).astype(np.float32)
            #TIT = np.nan_to_num(TIT, nan=0.0)
            
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
            dataset identifier ("TITeit" or "MLFeit")
        """
        save_path = self.input_path / 'clean' / id
        save_path.mkdir(parents = True, exist_ok = True)
        np.savez_compressed(
            save_path / f"{modality}.npz",
            EITdata=arr,
            pressureData=pData,
            volumeData=vData,
        )

