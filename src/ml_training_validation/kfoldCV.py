import joblib
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb


logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("KFoldCross")


class KfoldCV:
    def __init__(self, x: np.array, y: np.array, x_test: np.array, y_test: np.array, 
                 pmean:float, pstd:float, path: Path) -> None:
        self.x = x
        self.y = y
        self.pmean = pmean
        self.pstd = pstd
        self.path = path
        self.x_test = x_test
        self.y_test = y_test
        
        self.kf = KFold(n_splits=4, 
                        shuffle=True, 
                        random_state=42)
        
        self.xgb_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            early_stopping_rounds=5,
            random_state=42
        )


    def run(self) -> None:
        logger.info("K Fold Cros Validation")
        models = []
        errors = {}

        for num, (train_idx, val_idx) in enumerate(self.kf.split(self.x), start=1):
            model = clone(self.xgb_model) 
            model.fit(self.x[train_idx], 
                      self.y[train_idx], 
                      eval_set=[(self.x[val_idx], self.y[val_idx])],
                      verbose=False)
            models.append(model)

            y_pred = model.predict(self.x_test)

            y_pred_real = y_pred * self.pstd + self.pmean
            y_true_real = self.y_test * self.pstd + self.pmean

            fig, ax = plt.subplots(figsize=(6, 6))

            ax.scatter(y_true_real, y_pred_real)
            ax.set_xlabel("True Values")
            ax.set_ylabel("Predictions")
            ax.set_title(f"Prediction vs True {num}")

            fig.savefig(self.path / f"kfoldCV_{num}_prediction.png", bbox_inches='tight')
            plt.close(fig)

            logger.info(f"Fold {num} prediction plot saved to {self.path} / kfoldCV_{num}_prediction.png")

            mae = mean_absolute_error(y_true_real, y_pred_real)
            rmse = np.sqrt(mean_squared_error(y_true_real, y_pred_real))

            errors[num] = {"mae": mae,
                          "rmse": rmse}
            
        mae_mean = np.mean([v["mae"] for v in errors.values()])
        rmse_mean = np.mean([v["rmse"] for v in errors.values()])
        logger.info(f"Model mean absolut error: {mae_mean}")
        logger.info(f"Model mean squared error: {rmse_mean}")

        joblib.dump(models, self.path / "xgb_fold_models.joblib")
        logger.info(f"Fold models saved to {self.path / 'xgb_fold_models.joblib'}")

