import gc
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Union
import scipy
import torch
import torch.nn.functional as F


class CleanZone:
    def __init__(self, 
                 input_path: Union[str, Path], 
                 ) -> None:
        self.input_path = input_path


    def run(self) -> None:
        csv_path = self.input_path/ 'raw' / 'MLM' / 'peep_tabular_data.csv'
        table = pd.read_csv(csv_path)
        table = table[~table["Pat-ID"].str.startswith("AD")]
        for w in table['Pat-ID']:
            print(table.loc[table['Pat-ID'] == w, 'PEEP'].values[0])
            """
            MLMmat = scipy.io.loadmat(self.input_path / 'raw' / 'MLM' / w / f"{w}_images_part_sb.mat")
            key = list(MLMmat.keys())[3]
            MLMrgb = self.get_norm_rgb(MLMmat[key])
            print('MLM: ' + str(MLMrgb.shape))

            pMLMmat = scipy.io.loadmat(self.input_path / 'raw' / 'MLM' / w / f"{w}_pressure_sb.mat")
            pkey = list(pMLMmat.keys())[3]
            print(pkey)
            print('pMLM: ' + str(pMLMmat[pkey].shape))

            vMLMmat = scipy.io.loadmat(self.input_path / 'raw' / 'MLM' / w / f"{w}_volume_sb.mat")
            vkey = list(vMLMmat.keys())[3]
            print(vkey)
            print('vMLM: ' + str(vMLMmat[vkey].shape))

            self.save_npz(MLMrgb, pMLMmat[pkey], vMLMmat[vkey], w, table, 'MLMeit')
            
            del MLMrgb, MLMmat, pMLMmat, vMLMmat, pkey, vkey, key
            gc.collect()
            torch.cuda.empty_cache()

            TITmat = scipy.io.loadmat(self.input_path / 'raw' / 'Titration' / w / f"{w}_images_part_tit.mat")
            key = list(TITmat.keys())[3]
            TITrgb = self.get_norm_rgb(TITmat[key])
            print('TIT: ' + str(TITrgb.shape))

            
            pTITmat = scipy.io.loadmat(self.input_path / 'raw' / 'Titration' / w / f"{w}_pressure_tit.mat")
            pkey = list(pTITmat.keys())[3]
            print(pkey)
            print('pTIT: ' + str(pTITmat[pkey].shape))

            vTITmat = scipy.io.loadmat(self.input_path / 'raw' / 'Titration' / w / f"{w}_volume_tit.mat")
            vkey = list(vTITmat.keys())[3]
            print(vkey)
            print('vTIT: ' + str(vTITmat[vkey].shape))

            self.save_npz(TITrgb, pTITmat[pkey], vTITmat[vkey], w, table, 'TITeit')
            del TITmat, TITrgb, pTITmat, vTITmat, pkey, vkey, key
            gc.collect()
            torch.cuda.empty_cache()
            """
            save_path = Path(self.input_path) / 'clean' / w
            save_path.mkdir(parents = True, exist_ok = True)
            np.savez(
                save_path / f"table.npz",
                Pat_ID = w,
                sex = table.loc[table['Pat-ID'] == w, 'sex'].values[0],
                age = table.loc[table['Pat-ID'] == w, 'age'].values[0],
                BMI = table.loc[table['Pat-ID'] == w, 'BMI'].values[0],
                weight_kg = table.loc[table['Pat-ID'] == w, 'weight (kg)'].values[0],
                height_cm = table.loc[table['Pat-ID'] == w, 'height (cm)'].values[0],
                smoke = table.loc[table['Pat-ID'] == w, 'smoke'].values[0],
                Ppeak = table.loc[table['Pat-ID'] == w, 'Ppeak'].values[0],
                Pmean = table.loc[table['Pat-ID'] == w, 'Pmean'].values[0],
                targetPEEP = table.loc[table['Pat-ID'] == w, 'PEEP'].values[0]
            )
            break


    def get_norm_rgb(self, arr):
        """
        Takes: an array arr of an EIT image with the shape (32,32,N)
        Returns: an array of an upsampled and normalizet EIT image with added rgb  and transposed shape = (N,3,224,224)
        """
        arr = np.transpose(arr, (2,0,1))
        rgb = np.repeat(arr[:, None, :, :], 3,  axis = 1)
        rgb = torch.from_numpy(rgb).float()
        rgb = F.interpolate(
            rgb,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )
        mean = rgb.mean(dim=(0, 2, 3), keepdim=True)
        std  = rgb.std(dim=(0, 2, 3), keepdim=True) + 1e-6
        rgb = (rgb - mean) / std
        return rgb


    def save_npz(self, eitData, pData, vData, w, table, name):
            save_path = Path(self.input_path) / 'clean' / w
            save_path.mkdir(parents = True, exist_ok = True)
            np.savez(
                save_path / f"{name}.npz",
                EITdata=eitData,
                pressureData=pData,
                volumeData=vData,
                patient_id=w,
                targetPEEP=table.loc[table['Pat-ID'] == w, 'PEEP'].values[0]
            )
