import logging
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Tuple

from patientDataModule import PatientDataModule
from randomForest import RandomForest
from trainer import Trainer


logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("Workflow")

class Workflow:
    def __init__(self, input_path:str, output_path:str) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        (self.output_path / "clean" / "3D").mkdir(parents = True, exist_ok = True)
        (self.output_path / "clean" / "VP").mkdir(parents = True, exist_ok = True)        
        (self.output_path / "splitTit" / "3D").mkdir(parents = True, exist_ok = True)
        (self.output_path / "splitTit" / "VP").mkdir(parents = True, exist_ok = True)

    def run(self) -> None:
        logger.info("\n##########\nStart Process Full Data 3D\n##########")
        logger.info("\n##########\nStart DataLoader\n##########")
        train_loader, val_loader, test_loader, input_size = (
            PatientDataModule(self.input_path / "clean" / "3D", 4).run()
        )
        logger.info("\n##########\Start MLP Trainer\n##########")
        Trainer(train_loader, val_loader, test_loader, input_size, self.output_path / "clean" / "3D").run()

        logger.info("\n##########\nRandom Forest\n##########")
        x_train, y_train = self.get_numpy(train_loader)
        x_val, y_val = self.get_numpy(val_loader)
        x_test, y_test = self.get_numpy(test_loader)

        x_new_train = np.concatenate([x_train, x_val], axis=0)
        y_new_train = np.concatenate([y_train, y_val], axis=0)

        RandomForest(x_new_train, y_new_train, x_test, y_test, self.output_path / "clean" / "3D").run()


        logger.info("\n##########\nStart Process Full Data VP\n##########")
        logger.info("\n##########\nStart DataLoader\n##########")
        train_loader, val_loader, test_loader, input_size = (
            PatientDataModule(self.input_path / "clean" / "VP", 4).run()
        )
        logger.info("\n##########\Start MLP Trainer\n##########")
        Trainer(train_loader, val_loader, test_loader, input_size, self.output_path / "clean" / "VP").run()

        logger.info("\n##########\nRandom Forest\n##########")
        x_train, y_train = self.get_numpy(train_loader)
        x_val, y_val = self.get_numpy(val_loader)
        x_test, y_test = self.get_numpy(test_loader)

        x_new_train = np.concatenate([x_train, x_val], axis=0)
        y_new_train = np.concatenate([y_train, y_val], axis=0)

        RandomForest(x_new_train, y_new_train, x_test, y_test, self.output_path / "clean" / "VP").run()


        logger.info("\n##########\nStart Process Half Data 3D\n##########")
        logger.info("\n##########\nStart DataLoader\n##########")
        train_loader, val_loader, test_loader, input_size = (
            PatientDataModule(self.input_path / "splitTit" / "3D", 4).run()
        )
        logger.info("\n##########\Start MLP Trainer\n##########")
        Trainer(train_loader, val_loader, test_loader, input_size, self.output_path / "splitTit" / "3D").run()

        logger.info("\n##########\nRandom Forest\n##########")
        x_train, y_train = self.get_numpy(train_loader)
        x_val, y_val = self.get_numpy(val_loader)
        x_test, y_test = self.get_numpy(test_loader)

        x_new_train = np.concatenate([x_train, x_val], axis=0)
        y_new_train = np.concatenate([y_train, y_val], axis=0)

        RandomForest(x_new_train, y_new_train, x_test, y_test, self.output_path / "splitTit" / "3D").run()


        logger.info("\n##########\nStart Process Half Data VP\n##########")
        logger.info("\n##########\nStart DataLoader\n##########")
        train_loader, val_loader, test_loader, input_size = (
            PatientDataModule(self.input_path / "splitTit" / "VP", 4).run()
        )
        logger.info("\n##########\Start MLP Trainer\n##########")
        Trainer(train_loader, val_loader, test_loader, input_size, self.output_path / "splitTit" / "VP").run()

        logger.info("\n##########\nRandom Forest\n##########")
        x_train, y_train = self.get_numpy(train_loader)
        x_val, y_val = self.get_numpy(val_loader)
        x_test, y_test = self.get_numpy(test_loader)

        x_new_train = np.concatenate([x_train, x_val], axis=0)
        y_new_train = np.concatenate([y_train, y_val], axis=0)

        RandomForest(x_new_train, y_new_train, x_test, y_test, self.output_path / "splitTit" / "VP").run()

        logger.info("\n##########\nFinished Process\n##########")


    def get_numpy(self, arr:DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        x_list = []
        y_list = []

        for xb, yb in arr:
            x_list.append(xb.numpy())
            y_list.append(yb.numpy())
        
        x_np = np.concatenate(x_list, axis=0)
        y_np = np.concatenate(y_list, axis=0).ravel()

        return x_np, y_np


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

