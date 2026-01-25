import logging
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Tuple


logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


logger = logging.getLogger("dataLoader")


class PatientDataset(Dataset):
    def __init__(self, path:Path, batch_size) -> None:
        self.path = path
        self.batch_size = batch_size
    
    def run(self) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        x_train = []
        x_val = []
        x_test = []
        y_train = []
        y_val = []
        y_test = []

        pat_folders = sorted([f for f in self.path.iterdir() if f.is_dir()])        
        for p in pat_folders:

            eit = [np.load(f) for f in p.glob("*_LSTM.npz")]
            for data in eit:
                tit = torch.tensor(data["tit"], dtype=torch.float32)
                mlm = torch.tensor(data["mlm"], dtype=torch.float32)

                full_tab = np.load(self.path.parent / "table" / p.name / f"{data['patient_id']}.npz")
                tab = []
                tab.append(full_tab["sex"])
                tab.append(full_tab["age"])
                tab.append(full_tab["BMI"])
                tab.append(full_tab["weight"])
                tab.append(full_tab["height"])
                tab.append(full_tab["smoke"])
                tab = torch.tensor(tab, dtype=torch.float32).unsqueeze(0)

                if p.name == "train":
                    x_train.append(torch.cat([tit, mlm, tab], dim=0))
                    y_train.append(torch.tensor(full_tab["targetPEEP"], dtype=torch.float32))
                elif p.name == "val":
                    x_val.append(torch.cat([tit, mlm, tab], dim=0))
                    y_val.append(torch.tensor(full_tab["targetPEEP"], dtype=torch.float32))
                else:
                    x_test.append(torch.cat([tit, mlm, tab], dim=0))
                    y_test.append(torch.tensor(full_tab["targetPEEP"], dtype=torch.float32)) 


        x_train = torch.stack(x_train, dim=0)
        y_train = torch.stack(y_train, dim=0)
        x_val   = torch.stack(x_val, dim=0)
        y_val   = torch.stack(y_val, dim=0)
        x_test  = torch.stack(x_test, dim=0)
        y_test  = torch.stack(y_test, dim=0)

        train_set = TensorDataset(x_train, y_train)
        train = DataLoader(train_set, self.batch_size, shuffle=True)
        val_set = TensorDataset(x_val, y_val)
        val = DataLoader(val_set, self.batch_size)
        test_set = TensorDataset(x_test, y_test)
        test = DataLoader(test_set, self.batch_size)

        return train, val, test, x_train.shape[1]
