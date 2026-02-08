import logging
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Tuple


logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


logger = logging.getLogger("PatientDataModule")


class PatientDataModule:
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
                tit = torch.tensor(data["tit"], dtype=torch.float32).squeeze(0)
                mlm = torch.tensor(data["mlm"], dtype=torch.float32).squeeze(0)

                # nach tit / mlm Laden
                #if not torch.isfinite(tit).all():
                #    raise ValueError(
                #        f"NaN/Inf in TIT for patient {data['patient_id']} "
                #        f"min={tit.min()}, max={tit.max()}"
                #    )

                #if not torch.isfinite(mlm).all():
                #    raise ValueError(
                #        f"NaN/Inf in MLM for patient {data['patient_id']} "
                #        f"min={mlm.min()}, max={mlm.max()}"
                #    )

                full_tab = np.load(self.path.parent / "table" / p.name / f"{data['patient_id']}.npz")
                tab = np.concatenate(
                    [
                        np.array([full_tab["sex"].item()], dtype=np.float32),
                        np.array([full_tab["age"].item()], dtype=np.float32),
                        np.array([full_tab["BMI"].item()], dtype=np.float32),
                        np.array([full_tab["weight"].item()], dtype=np.float32),
                        np.array([full_tab["height"].item()], dtype=np.float32),
                        full_tab["smoke"].astype(np.float32).reshape(-1), 
                    ],
                    axis=0,
                )



                tab = torch.from_numpy(tab)#.unsqueeze(0)
                #if not torch.isfinite(tab).all():
                #    raise ValueError(
                #        f"NaN/Inf in TAB for patient {data['patient_id']} "
                #        f"values={tab}"
                #    )

                if p.name == "train":
                    x_train.append(torch.cat([tit, mlm, tab], dim=0))
                    y_train.append(
                        torch.tensor([full_tab["targetPEEP"].item()], dtype=torch.float32)
                    )
                elif p.name == "val":
                    x_val.append(torch.cat([tit, mlm, tab], dim=0))
                    y_val.append(
                        torch.tensor([full_tab["targetPEEP"].item()], dtype=torch.float32)
                    )
                else:
                    x_test.append(torch.cat([tit, mlm, tab], dim=0))
                    y_test.append(
                        torch.tensor([full_tab["targetPEEP"].item()], dtype=torch.float32)
                    )

        assert len(x_train) > 0, "Train set empty"
        
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
