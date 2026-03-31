import joblib
import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

import xgboost as xgb


logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("XGBoost")


class XGBoost:
    def __init__(self, x_train: np.array, y_train: np.array, x_val: np.array, y_val: np.array, x_test: np.array, y_test: np.array, pmean:float, pstd:float, path: Path) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val        
        self.x_test = x_test
        self.y_test = y_test
        self.pmean = pmean
        self.pstd = pstd
        self.path = path

        self.xgb_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            early_stopping_rounds=5,
            random_state=42
        )

    def run(self):
        logger.info("Start XGBoost")

        self.xgb_model.fit(
            self.x_train,
            self.y_train,
            eval_set=[(self.x_val, self.y_val)],
            verbose=False
        )
        
        y_pred = self.xgb_model.predict(self.x_test)
        y_pred_real = y_pred * self.pstd + self.pmean
        y_true_real = self.y_test * self.pstd + self.pmean

        mae = mean_absolute_error(y_true_real, y_pred_real)
        rmse = np.sqrt(mean_squared_error(y_true_real, y_pred_real))
        logger.info(f"Model mean absolut error: {mae}")
        logger.info(f"Model mean squared error: {rmse}")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true_real, y_pred_real)
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predictions")
        ax.set_title("Prediction vs True")
        fig.savefig(self.path / "XGB_prediction.png", bbox_inches='tight')
        plt.close(fig)

        logger.info(f"XGBoost prediction saved to {self.path} / XGB_prediction.png")       

        joblib.dump(self.xgb_model, self.path / "XGB_model.joblib")
        logger.info(f"XGBoost model saved to {self.path} / XGB_model.joblib")
