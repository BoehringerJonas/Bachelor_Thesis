import pandas as pd
import numpy as np
import logging
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_absolute_error, mean_squared_error
from typing import Tuple

from mlp import MLP
from dataLoader import DataLoader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("workflow")

class Workflow:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.patience = 10

        self.best_val_loss = float("inf")

        

    def run(self):
        #data
        train_load, val_load, test_load, input_size = DataLoader(self.path)

        #model
        model = MLP(
                input_size = input_size, 
                output_size = 1, 
                hidden_sizes = [64, 32]
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
        self.model.parameters(),
        lr=self.lr,
        weight_decay=self.weight_decay
        )

        self.criterion = torch.nn.L1Loss()

        # Training
        self.train(train_load, val_load)

        # Testing
        self.test(test_load)
    



def entrypoint():
    import argparse

    parser = argparse.ArgumentParser(description="Arguments for segmentator evaluation")

    parser.add_argument(
        "--input_path",
        help="Path to parent directory of the nifti files",
        required=True,
    )

    args = parser.parse_args()

    Workflow(
        path=args.input_path,
    ).run()


if __name__ == "__main__":
    entrypoint()