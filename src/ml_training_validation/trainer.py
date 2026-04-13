import json
import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset
from typing import List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import KFold
from mlp import MLP

logger = logging.getLogger("workflow")


class Trainer:
    def __init__(self, train_loader:DataLoader, val_loader:DataLoader, test_loader:DataLoader,
                full_loader:DataLoader, pmean:float, pstd:float, path:Path) -> None:
        """
        Parameters
        ----------

        Returns
        -------

        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.full_loader = full_loader
        self.pmean = pmean
        self.pstd = pstd
        self.path = path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lr = 5e-4
        self.weight_decay = 1e-4
        self.patience = 40

    def run(self) -> None:
        """
        Inits the model and optimizer. Starts training, evaluation and safes plot for early stopping
        """
        logger.info("Init MLP")
        for mlf, tit, tab, targets in self.train_loader:
            mlf_size = mlf.shape[1]
            tit_size = tit.shape[1]
            tab_size = tab.shape[1]
            break
        model = MLP(
            mlf_size=mlf_size,
            tit_size=tit_size,
            tab_size=tab_size
        ).to(self.device)

        for mlf, tit, tab, targets in self.train_loader:
            outputs = model(
                mlf.to(self.device),
                tit.to(self.device),
                tab.to(self.device)
            )

            logger.info(f"mlf: {mlf.shape}")
            logger.info(f"tit: {tit.shape}")
            logger.info(f"tab: {tab.shape}")
            logger.info(f"targets: {targets.shape}")
            logger.info(f"outputs: {outputs.shape}")
            break

        optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=self.lr,
        weight_decay=self.weight_decay
        )

        criterion = torch.nn.MSELoss()

        logger.info("Train Model")
        train_losses, val_losses = self.train_model(model, 
                                                    self.train_loader, 
                                                    self.val_loader, 
                                                    criterion, 
                                                    optimizer,
                                                    ""
                                                    )


        # Evaluate the model
        test_loss, test_mae, test_mse = self.evaluate_model(model, self.test_loader, criterion, "")

        plt.figure()
        plt.plot(train_losses, label="Train")
        plt.plot(val_losses, label="Validation")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("MLP Training with Early Stopping")
        plt.savefig(self.path / "training_curveMLP.png")

        self.crossVal(mlf_size, tit_size, tab_size)


    def crossVal(self, mlf_size:int, tit_size:int, tab_size:int) -> None:
        """
        Get the best model from the folds.

        Parameters
        ----------
        mlf_size    : int
            Size of MLF features
        tit_size    : int
            Size of MLF features
        tab_size    : int
            Size of MLF features
        """
        kfolds = 6
        logger.info(f"MLP CrossFoldVal folds: {kfolds}")
        best_split = {}
        best_loss = np.inf

        kfold = KFold(n_splits=kfolds, shuffle=True)
        dataset = self.full_loader.dataset
        for (train_ids, val_ids) in kfold.split(dataset):
            model = MLP(
                mlf_size=mlf_size,
                tit_size=tit_size,
                tab_size=tab_size
            ).to(self.device)

            train_subsampler = Subset(dataset, train_ids)
            val_subsampler = Subset(dataset, val_ids)
            train_loader = DataLoader(train_subsampler, batch_size=64, shuffle=True) # batchsize 32 test
            val_loader = DataLoader(val_subsampler, batch_size=64, shuffle=False)

            optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
            )

            criterion = torch.nn.MSELoss()
            train_losses, val_losses = self.train_model(model, 
                                                        train_loader, 
                                                        val_loader, 
                                                        criterion, 
                                                        optimizer,
                                                        "", 
                                                        safe=False)
            if(val_losses[-1] < best_loss):
                best_loss = val_losses[-1]
                best_split = {
                    "train_loader": train_loader,
                    "val_loader": val_loader
                }

        model = MLP(
            mlf_size=mlf_size,
            tit_size=tit_size,
            tab_size=tab_size
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
            )

        train_losses, val_losses = self.train_model(model, 
                                    best_split["train_loader"], 
                                    best_split["val_loader"], 
                                    criterion, 
                                    optimizer, 
                                    "CV"
                                    )
        
        test_loss, test_mae, test_mse = self.evaluate_model(model, self.test_loader, criterion, "CV")

        plt.figure()
        plt.plot(train_losses, label="Train")
        plt.plot(val_losses, label="Validation")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("CV MLP Training with Early Stopping")
        plt.savefig(self.path / "CVtraining_curveMLP.png")
        

    def train_model(self, model:Module, train_loader:DataLoader, val_loader:DataLoader, criterion:Module,
                     optimizer:Optimizer, mode:str, num_epochs:int=150, safe:bool=True) -> Tuple[List, List]:
        """
        Trains MLP model and checks for best moddel with a given patiens or max epochs.

        Parameters
        ----------
        model:Module
            MLP
        train_loader:DataLoader
        val_loader:DataLoader
        criterion:Module
            MSE Loss
        optimizer:Optimizer
        mode:str
            CV or not CV
        num_epochs:int=150
        safe:bool=True
            Not safe for CV
        
        Returns
        -------
        List : training_losses
        List : val_losses

        """
        logger.info("Train Model")
        train_losses = []
        val_losses = []

        best_val_loss = float("inf")
        #min_delta = 1e-8

        patience_counter = 0

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for mlf, tit, tab, targets in train_loader:
                mlf = mlf.to(self.device, non_blocking=True)
                tit = tit.to(self.device, non_blocking=True)
                tab = tab.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                
                outputs = model(mlf, tit, tab)                
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * mlf.size(0)
            
            train_loss = running_loss / len(train_loader.dataset)
            val_loss = self.validate_model(model, val_loader, criterion)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            logger.info(
            f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}"
            )

            if val_loss < best_val_loss:# - min_delta: 
                best_val_loss = val_loss
                patience_counter = 0
                if safe:
                    torch.save(model.state_dict(), self.path / f"{mode}MLP_model.pt")
                    logger.info("Validation improved → save model, reset patience")
            else:
                patience_counter += 1
                logger.info(f"No significant improvement (patience {patience_counter}/{self.patience})")


            if patience_counter >= self.patience:
                logger.info("Early stopping triggered")
                break

        return train_losses * self.pstd + self.pmean, val_losses * self.pstd + self.pmean
    
    def validate_model(self, model:Module, val_loader:DataLoader, criterion:Module) -> float:
        """
        Checks the validation Loss

        Parameters
        ----------
        model       : Module
            MLP
        val_loader  : DataLoader
        criterion   : Module
            MSE Loss
        Returns
        -------
        float : mean loss of MLP

        """
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for mlf, tit, tab, targets in val_loader:
                mlf = mlf.to(self.device)
                tit = tit.to(self.device)
                tab = tab.to(self.device)
                targets = targets.to(self.device)

                outputs = model(mlf, tit, tab)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * mlf.size(0)

        return val_loss / len(val_loader.dataset)


    def evaluate_model(self, model:Module, test_loader:DataLoader, 
                       criterion:Module, mode:str) -> Tuple[float, float, float]:
        """
        Tests the performance of the end model.

        Parameters
        ----------
        model       : Module
            MLP
        test_loader : DataLoader
        criterion   : Module
            MSE Loss
        mode        : str
            CV or not CV

        Returns
        -------
        float : acg_loss
        float : mae
        float : mse
        """
        #logger.info("Evaluate Model")
        model.eval() 
        test_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for mlf, tit, tab, targets in test_loader:
                mlf = mlf.to(self.device)
                tit = tit.to(self.device)
                tab = tab.to(self.device)
                targets = targets.to(self.device)

                outputs = model(mlf, tit, tab)
                
                loss = criterion(outputs, targets)
                test_loss += loss.item() * mlf.size(0)
                test_loss = test_loss
                
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        avg_loss = test_loss / len(test_loader.dataset) * self.pstd + self.pmean
        
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        
        y_pred_real = y_pred * self.pstd + self.pmean
        y_true_real = y_true * self.pstd + self.pmean

        mae = mean_absolute_error(y_true_real, y_pred_real)
        mse = mean_squared_error(y_true_real, y_pred_real)
        rmse = np.sqrt(mse)
        medae = median_absolute_error(y_true_real, y_pred_real)
        r2 = r2_score(y_true_real, y_pred_real)
        
        model_key = f"{mode}MLP_{self.path.name}"
        result = {
            self.path.parent.name: {
                model_key: {
                    "mae": float(mae),
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "medae": float(medae),
                    "r2": float(r2)
                }
            }
        }
        with open(self.path / f"{mode}result_{model_key}.json", "w") as f:
            json.dump(result, f, indent=4)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true_real, y_pred_real)
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predictions")
        ax.set_title(f"{mode}MLP Prediction vs True")
        fig.savefig(self.path / f"{mode}MLP_prediction.png", bbox_inches='tight')
        plt.close(fig)

        logger.info(f"{mode} MLP prediction saved to {self.path} / {mode}MLP_prediction.png")       

        return avg_loss, mae, mse

