import logging
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Tuple

from patientDataModule import PatientDataModule
from randomForest import RandomForest
from xgBoost import XGBoost
from trainer import Trainer
from kfoldCV import KfoldCV


logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("Workflow")

class Workflow:
    def __init__(self, input_path:str, output_path:str) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        (self.output_path / "full_tit" / "3D").mkdir(parents = True, exist_ok = True)
        (self.output_path / "full_tit" / "VP").mkdir(parents = True, exist_ok = True)        
        (self.output_path / "half_tit" / "3D").mkdir(parents = True, exist_ok = True)
        (self.output_path / "half_tit" / "VP").mkdir(parents = True, exist_ok = True)

        full_stats = np.load(self.input_path / "full_tit" / "stats" / "train_stats.npz")
        half_stats = np.load(self.input_path / "half_tit" / "stats" / "train_stats.npz")

        self.fpmean = full_stats["pmean"]
        self.fpstd = full_stats["pstd"]
        self.hpmean = half_stats["pmean"]
        self.hpstd = half_stats["pstd"]

    def run(self) -> None:
        a = ["full_tit", "half_tit"]
        b = ["3D", "VP"]
        for i in a:
            for j in b:
                logger.info(f"\n##########\nStart Process for {i} {j}\n##########")
                if i == "full_tit":
                    mean = self.fpmean
                    std = self.fpstd
                elif i == "half_tit":
                    mean = self.hpmean
                    std = self.hpstd

                logger.info(f"\n##########\nStart DataLoader\n##########")
                train_loader, val_loader, test_loader = (
                    PatientDataModule(self.input_path / i / j, 8).run()
                )
                logger.info(f"\n##########\nStart MLP Trainer\n##########")
                Trainer(train_loader, val_loader, test_loader, mean, std, self.output_path / i / j).run()

                logger.info(f"\n##########\nMaschine Learning\n##########")
                x_train, y_train = self.get_numpy(train_loader)
                x_val, y_val = self.get_numpy(val_loader)
                x_test, y_test = self.get_numpy(test_loader)

                x_new_train = np.concatenate([x_train, x_val], axis=0)
                y_new_train = np.concatenate([y_train, y_val], axis=0)

                RandomForest(x_new_train, y_new_train, x_test, y_test, mean, std, self.output_path / i / j).run()
                XGBoost(x_train, y_train, x_val, y_val, x_test, y_test, mean, std, self.output_path / i / j).run()

                logger.info(f"\n##########\K Fold CV\n##########")
                #KfoldCV(x_new_train, y_new_train, x_test, y_test, mean, std, self.output_path / i / j).run()

        logger.info("\n##########\nFinished Process\n##########")


    def get_numpy(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        mlfs, tits, tabs, ys = [], [], [], []

        for mlf, tit, tab, y in loader:
            mlfs.append(mlf.cpu().numpy())
            tits.append(tit.cpu().numpy())
            tabs.append(tab.cpu().numpy())
            ys.append(y.cpu().numpy())
        
        x_mlf = np.vstack(mlfs)
        x_tit = np.vstack(tits)
        x_tab = np.vstack(tabs)
        y = np.vstack(ys).ravel()  # shape: (n_samples,)

        # Early Fusion
        x = np.concatenate([x_mlf, x_tit, x_tab], axis=1)
        return x, y


def entrypoint():
    import argparse

    parser = argparse.ArgumentParser(description="paths for in and output")

    parser.add_argument(
        "--input_path",
        help="Path to parent directory of the input .npz files",
        required=True,
    )
    parser.add_argument(
        "--output_path",
        help="Path to safe model and evaluation",
        required=True,
    )

    args = parser.parse_args()

    Workflow(
        input_path=args.input_path,
        output_path=args.output_path
    ).run()

if __name__ == "__main__":
    entrypoint()

