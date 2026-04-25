import json
import logging
import numpy as np
from pathlib import Path
import time
from torch.utils.data import DataLoader
from typing import Tuple

from patientDataModule import PatientDataModule
from randomForest import RandomForest
from xgBoost import XGBoost
from trainer import Trainer
from cross_validation import CVtest
from feature_ranking import FeatureRanking


log_file = f"/hpc/gpfs2/home/u/boehrijo/logs/workflow_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Workflow")

class Workflow:
    def __init__(self, input_path:str, output_path:str) -> None:
        """
        Parameters
        ----------

        Returns
        -------

        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

        full_stats = np.load(self.input_path / "full_tit" / "stats" / "train_stats.npz")
        half_stats = np.load(self.input_path / "half_tit" / "stats" / "train_stats.npz")
        third_stats = np.load(self.input_path / "third_tit" / "stats" / "train_stats.npz")

        self.fpmean = full_stats["pmean"]
        self.fpstd = full_stats["pstd"]
        self.hpmean = half_stats["pmean"]
        self.hpstd = half_stats["pstd"]
        self.tpmean = third_stats["pmean"]
        self.tpstd = third_stats["pstd"]

    def run(self) -> None:
        """
        Starts Data loader and the different training models for the differnt 
        combinations of split and picture kind. Combines the different json files
        to a whole one.
        """
        a = ["full_tit", "half_tit", "third_tit"]
        b = ["3D", "VP"]
        
        for i in a:
            for j in b:        
                (self.output_path / i / j).mkdir(parents = True, exist_ok = True)
                logger.info(f"\n##########\nStart Process for {i} {j}\n##########")
                if i == "full_tit":
                    mean = self.fpmean
                    std = self.fpstd
                elif i == "half_tit":
                    mean = self.hpmean
                    std = self.hpstd
                elif i == "third_tit":
                    mean = self.tpmean
                    std = self.tpstd

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


                RandomForest(x_train, y_train, x_test, y_test, mean, std, self.output_path / i / j).run()
                XGBoost(x_train, y_train, x_train, y_train, x_val, y_val, x_test, y_test, mean, std, self.output_path / i / j).run()

                CVtest(x_train, y_train, std, self.output_path / i / j).run()
                FeatureRanking(self.output_path / i / j, test_loader).run()

        
        logger.info("\n##########\nCreate ovre all json\n##########\n")

        results_path = self.output_path / "full_results.json"
        full_results = {}

        for result_file in self.output_path.rglob("result*.json"):
            result = json.loads(result_file.read_text())

            for dataset_key, models in result.items():
                if dataset_key not in full_results:
                    full_results[dataset_key] = {}

                for model_key, metrics in models.items():
                    full_results[dataset_key][model_key] = metrics

        results_path.write_text(json.dumps(full_results, indent=4))
        logger.info("\n##########\nFinished Process\n##########")


    def get_numpy(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts Dataloader to numpy.

        Parameters
        ----------
        loader  : DataLoader

        Returns
        -------
        np.ndarray  : input values
        np.ndarray  : target value
        """
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
    """
    Starting point of the whole ml pipeline.

    Parameters
    ----------
    --input_path:   str
        Path to input parent directory of the input feature .npz files
    --output_path:  str
        Path to safe model and evaluation
    """
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

