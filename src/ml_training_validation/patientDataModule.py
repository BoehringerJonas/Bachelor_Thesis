import logging
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple

logger = logging.getLogger("workflow")


class PatientDataModule:
    def __init__(self, path:Path, batch_size) -> None:
        self.path = path
        self.batch_size = batch_size
    
    def run(self) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        """
        Checks for wrong values. 
        Stackes the data to train, val, test and returns them as dataloader
        
        Returns
        -------
        DataLoader  : train dataset as dataloader
        DataLoader  : val dataset as dataloader
        DataLoader  : test dataset as dataloader
        DataLoader  : train + val dataset as dataloader
        """

        x_train_tit = []
        x_train_mlf = []
        x_train_tab = []
        x_train_tab_cat = []
        x_val_tit = []
        x_val_mlf = []
        x_val_tab = []
        x_val_tab_cat = []
        x_test_tit = []
        x_test_mlf = []
        x_test_tab = []
        x_test_tab_cat = []

        y_train = []
        y_val = []
        y_test = []

        pat_folders = sorted([f for f in self.path.iterdir() if f.is_dir()])        
        for p in pat_folders:

            eit = [np.load(f) for f in sorted(p.glob("*_LSTM.npz"))]
            for data in eit:
                tit = torch.tensor(data["tit"], dtype=torch.float32).squeeze(0)
                mlf = torch.tensor(data["mlf"], dtype=torch.float32).squeeze(0)

                if not torch.isfinite(tit).all():
                    logger.warning(f"Invalid TIT {data['patient_id']}")
                    continue

                if not torch.isfinite(mlf).all():
                    logger.warning(f"Invalid MLF {data['patient_id']}")
                    continue

                assert tit.shape == mlf.shape, "TIT / MLF mismatch"

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

                tab = torch.from_numpy(tab)

                if p.name == "train":
                    x_train_tit.append(tit)
                    x_train_mlf.append(mlf)
                    x_train_tab.append(tab[:5])
                    x_train_tab_cat.append(tab[5:])

                    y_train.append(
                        torch.tensor([full_tab["targetPEEP"].item()], dtype=torch.float32)
                    )
                elif p.name == "val":
                    x_val_tit.append(tit)
                    x_val_mlf.append(mlf)
                    x_val_tab.append(tab[:5])
                    x_val_tab_cat.append(tab[5:])
                    y_val.append(
                        torch.tensor([full_tab["targetPEEP"].item()], dtype=torch.float32)
                    )
                else:
                    x_test_tit.append(tit)
                    x_test_mlf.append(mlf)
                    x_test_tab.append(tab[:5])
                    x_test_tab_cat.append(tab[5:])
                    y_test.append(
                        torch.tensor([full_tab["targetPEEP"].item()], dtype=torch.float32)
                    )

        assert len(x_train_tit) > 0, "Train set empty"
        
        x_train_tit = torch.stack(x_train_tit, dim=0)
        x_train_mlf = torch.stack(x_train_mlf, dim=0)
        x_train_tab = torch.stack(x_train_tab, dim=0)
        x_train_tab_cat = torch.stack(x_train_tab_cat, dim=0).numpy()
        x_val_tit = torch.stack(x_val_tit, dim=0)
        x_val_mlf = torch.stack(x_val_mlf, dim=0)
        x_val_tab = torch.stack(x_val_tab, dim=0)
        x_val_tab_cat = torch.stack(x_val_tab_cat, dim=0).numpy()
        x_test_tit = torch.stack(x_test_tit, dim=0)
        x_test_mlf = torch.stack(x_test_mlf, dim=0)
        x_test_tab = torch.stack(x_test_tab, dim=0)
        x_test_tab_cat = torch.stack(x_test_tab_cat, dim=0).numpy()
        y_train = torch.stack(y_train, dim=0)
        y_val = torch.stack(y_val, dim=0)
        y_test = torch.stack(y_test, dim=0)

        x_train_tab = x_train_tab.detach().cpu().numpy()
        x_val_tab = x_val_tab.detach().cpu().numpy()
        x_test_tab = x_test_tab.detach().cpu().numpy()

        x_train_tab = np.concatenate([x_train_tab, x_train_tab_cat], axis=1)
        x_val_tab   = np.concatenate([x_val_tab, x_val_tab_cat], axis=1)
        x_test_tab  = np.concatenate([x_test_tab, x_test_tab_cat], axis=1)

        x_train_tab = torch.from_numpy(x_train_tab).float()
        x_val_tab = torch.from_numpy(x_val_tab).float()
        x_test_tab = torch.from_numpy(x_test_tab).float()

        train_set = TensorDataset(x_train_tit, x_train_mlf, x_train_tab, y_train)
        train = DataLoader(train_set, self.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_set = TensorDataset(x_val_tit, x_val_mlf, x_val_tab, y_val)
        val = DataLoader(val_set, self.batch_size, num_workers=2, pin_memory=True)
        test_set = TensorDataset(x_test_tit, x_test_mlf, x_test_tab, y_test)
        test = DataLoader(test_set, self.batch_size, num_workers=2, pin_memory=True)

        return train, val, test
