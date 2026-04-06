import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import  mean_absolute_error, root_mean_squared_error

from mlp import MLP
from patientDataModule import PatientDataModule


logger = logging.getLogger("workflow")


class Trainer:
    def __init__(self, train_loader:DataLoader, val_loader:DataLoader, test_loader:DataLoader, pmean:float, pstd:float, path:Path) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.pmean = pmean
        self.pstd = pstd
        self.path = path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.patience = 100

    def run(self):
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

        criterion = torch.nn.L1Loss()

        logger.info("Train Model")
        train_losses, val_losses = self.train_model(model, self.train_loader, self.val_loader, criterion, optimizer, num_epochs=100)


        # Evaluate the model
        test_loss, test_mae, test_rmse = self.evaluate_model(model, self.test_loader, criterion)
        logger.info(f'Test Loss: {test_loss:.4f}')
        logger.info(f"Test MAE:  {test_mae:.2f}")
        logger.info(f"Test RMSE: {test_rmse:.2f}")

        plt.figure()
        plt.plot(train_losses, label="Train")
        plt.plot(val_losses, label="Validation")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.title("Training with Early Stopping")
        plt.savefig(self.path / "training_curveMLP.png")



    def train_model(self, model, train_loader, val_loader, criterion, optimizer, num_epochs=100):
        logger.info("Train Model")
        train_losses = []
        val_losses = []

        best_val_loss = float("inf")
        min_delta = 1e-5

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

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), self.path / "MLP_model.pt")
                logger.info("Validation improved → save model, reset patience")
            else:
                patience_counter += 1
                logger.info(f"No significant improvement (patience {patience_counter}/{self.patience})")


            if patience_counter >= self.patience:
                logger.info("Early stopping triggered")
                break

        model.load_state_dict(torch.load(self.path / "MLP_model.pt", map_location=self.device))
        return train_losses, val_losses
    
    def validate_model(self, model, val_loader, criterion):
        logger.info("Validate Model")
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


    def evaluate_model(self, model, test_loader, criterion):
        logger.info("Evaluate Model")
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
                
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        avg_loss = test_loss / len(test_loader.dataset)
        
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        
        y_pred_real = y_pred * self.pstd + self.pmean
        y_true_real = y_true * self.pstd + self.pmean

        mae = mean_absolute_error(y_true_real, y_pred_real)
        rmse = root_mean_squared_error(y_true_real, y_pred_real)

        return avg_loss, mae, rmse
