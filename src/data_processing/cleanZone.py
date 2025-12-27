import gc
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Union
import scipy


class CleanZone:
    def __init__(self, 
        input_path: Union[str, Path], 
        ) -> None:
        self.input_path = input_path
        self.MLMmean, self.MLMstd = self.get_global("MLM")
        self.TITmean, self.TITstd = self.get_global("Titration")


    def run(self) -> None:
        csv_path = self.input_path/ 'raw' / 'MLM' / 'peep_tabular_data.csv'
        table = pd.read_csv(csv_path)
        table = table[~table["Pat-ID"].str.startswith("AD")]
        table = table.set_index("Pat-ID")
        for w in table.index:
            print(table.loc[w, "PEEP"])
            
            MLMmat = scipy.io.loadmat(self.input_path / 'raw' / 'MLM' / w / f"{w}_images_part_sb.mat")
            key = list(MLMmat.keys())[3]
            MLM = self.get_norm(MLMmat[key], "MLM")
            print('MLM: ' + str(MLM.shape))

            pMLMmat = scipy.io.loadmat(self.input_path / 'raw' / 'MLM' / w / f"{w}_pressure_sb.mat")
            pkey = list(pMLMmat.keys())[3]
            print(pkey)
            print('pMLM: ' + str(pMLMmat[pkey].shape))

            vMLMmat = scipy.io.loadmat(self.input_path / 'raw' / 'MLM' / w / f"{w}_volume_sb.mat")
            vkey = list(vMLMmat.keys())[3]
            print(vkey)
            print('vMLM: ' + str(vMLMmat[vkey].shape))

            self.save_npz(MLM, pMLMmat[pkey], vMLMmat[vkey], w, table, 'MLMeit')
            
            del MLM, MLMmat, pMLMmat, vMLMmat, pkey, vkey, key

            TITmat = scipy.io.loadmat(self.input_path / 'raw' / 'Titration' / w / f"{w}_images_part_tit.mat")
            key = list(TITmat.keys())[3]
            TIT = self.get_norm(TITmat[key], "TIT")
            print('TIT: ' + str(TIT.shape))
            
            pTITmat = scipy.io.loadmat(self.input_path / 'raw' / 'Titration' / w / f"{w}_pressure_tit.mat")
            pkey = list(pTITmat.keys())[3]
            print(pkey)
            print('pTIT: ' + str(pTITmat[pkey].shape))

            vTITmat = scipy.io.loadmat(self.input_path / 'raw' / 'Titration' / w / f"{w}_volume_tit.mat")
            vkey = list(vTITmat.keys())[3]
            print(vkey)
            print('vTIT: ' + str(vTITmat[vkey].shape))

            print("start save")
            self.save_npz(TIT, pTITmat[pkey], vTITmat[vkey], w, table, 'TITeit')
            print("finish save")
            del TITmat, TIT, pTITmat, vTITmat, pkey, vkey, key

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


    def get_norm(self, arr, modality:str):
        arr = np.transpose(arr, (2,0,1)).astype(np.float32)
        if modality == "MLM":
            arr = (arr - self.MLMmean) / (self.MLMstd + 1e-6)
        else:
            arr = (arr - self.TITmean) / (self.TITstd + 1e-6)
        return arr


    def save_npz(self, arr, pData, vData, w, table, name):
            save_path = Path(self.input_path) / 'clean' / w
            save_path.mkdir(parents = True, exist_ok = True)
            np.savez_compressed(
                save_path / f"{name}.npz",
                EITdata=arr,
                pressureData=pData,
                volumeData=vData,
            )


    def get_global(self, type: str, batch_size=256):
        spez = "sb" if type == "MLM" else "tit"
        
        sum_arr = 0
        sum_sq_arr = 0
        count = 0

        table = pd.read_csv(self.input_path / 'raw' / 'MLM' / 'peep_tabular_data.csv')
        table = table[~table["Pat-ID"].str.startswith("AD")]
        
        for w in table['Pat-ID']:
            mat = scipy.io.loadmat(self.input_path / 'raw' / type / w / f"{w}_images_part_{spez}.mat")
            key = list(mat.keys())[3]
            arr = np.transpose(mat[key], (2,0,1))  # shape (N,H,W)

            for i in range(0, arr.shape[0]):
                sum_arr += arr.sum(axis=0)
                sum_sq_arr += (arr**2).sum(axis=0)
                count += arr.shape[0]

            del mat, arr

        mean = sum_arr / count
        var  = sum_sq_arr / count - mean**2
        std = np.sqrt(var) + 1e-6
        return mean.astype(np.float32), std.astype(np.float32)
