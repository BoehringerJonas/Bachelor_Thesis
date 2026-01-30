import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from sklearn.metrics import  mean_absolute_error, mean_squared_error

from mlp import MLP
from patientDataModule import PatientDataModule

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("trainer")


class Workflow:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.patience = 10

        self.best_val_loss = float("inf")


    def run(self):
        self.best_val_loss = float("inf")

        train_loader, val_loader, test_loader, input_size = (
            PatientDataModule(self.path, 4).run()
        )


        model = MLP(
                input_size = input_size, 
                output_size = 1, 
                hidden_sizes = [64, 32]
        ).to(self.device)

        for inputs, targets in train_loader:
            outputs = model(inputs.to(self.device))

            logger.info(f"inputs:  {inputs.shape}")
            logger.info(f"targets: {targets.shape}")
            logger.info(f"outputs: {outputs.shape}")
            break

        optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=self.lr,
        weight_decay=self.weight_decay
        )

        criterion = torch.nn.L1Loss()

        train_losses, val_losses = self.train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100)


        # Evaluate the model
        test_loss, test_mae, test_rmse = self.evaluate_model(model, test_loader, criterion)
        logger.info(f'Test Loss: {test_loss:.4f}')
        logger.info(f"Test MAE:  {test_mae:.2f}")
        logger.info(f"Test RMSE: {test_rmse:.2f}")

        plt.plot(train_losses, label="Train")
        plt.plot(val_losses, label="Validation")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.title("Training with Early Stopping")
        plt.savefig("training_curve.png")



    def train_model(self, model, train_loader, val_loader, criterion, optimizer, num_epochs=100):
        logger.info("Train Model")
        train_losses = []
        val_losses = []

        patience_counter = 0

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
            
            train_loss = running_loss / len(train_loader.dataset)
            val_loss = self.validate_model(model, val_loader, criterion)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            logger.info(
            f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}"
            )

            if val_loss < self.best_val_loss  - 1e-4:
                self.best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "best_model.pt")
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                logger.info("Early stopping triggered")
                break

        model.load_state_dict(torch.load("best_model.pt"))
        return train_losses, val_losses
    
    def validate_model(self, model, val_loader, criterion):
        logger.info("Validate Model")
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)

        return val_loss / len(val_loader.dataset)


    def evaluate_model(self, model, test_loader, criterion):
        logger.info("Evaluate Model")
        model.eval()  # Set model to evaluation mode
        test_loss = 0.0
        all_preds = []
        all_targets = []
        
        # No gradient calculation needed for evaluation
        with torch.no_grad():
            for inputs, targets in test_loader:
                # Forward pass
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # Calculate metrics
        avg_loss = test_loss / len(test_loader.dataset)
        
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)

        return avg_loss, mae, rmse


def entrypoint():
    import argparse

    parser = argparse.ArgumentParser(description="Arguments for segmentator evaluation")

    parser.add_argument(
        "--input_path",
        help="Path to parent directory of the input .npz files",
        required=True,
    )

    args = parser.parse_args()

    Workflow(
        path=args.input_path,
    ).run()


if __name__ == "__main__":
    entrypoint()

