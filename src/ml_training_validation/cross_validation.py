import json
import logging
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_validate
import xgboost as xgb

logger = logging.getLogger("workflow")

class CVtest:
    def __init__(self, x_train: np.array, y_train: np.array, pstd:float, path: Path) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.pstd = pstd
        self.path = path

        self.model = xgb.XGBRegressor(
            tree_method="hist",
            device="cuda",
            objective="reg:squarederror",
            n_estimators=1000,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            reg_lambda=1.0
        )

    def run(self) -> None:
        scores = cross_validate(
            self.model,
            self.x_train,
            self.y_train,
            cv=4,
            scoring={
                "mse": "neg_mean_squared_error"
            }
        )

        mse = -scores["test_mse"] * np.square(self.pstd)
        mse_mean = np.mean(mse)
        mse_std = np.std(mse)
        rmse = np.sqrt(mse)
        rmse_mean = np.mean(rmse)
        rmse_std = np.std(rmse)
        
        model_key = f"CV_{self.path.parent.name}_{self.path.name}"
        result = {
            self.path.parent.name: {
                model_key: {
                    "mse_mean": float(mse_mean),
                    "mse_std": float(mse_std),
                    "rmse_mean": float(rmse_mean),
                    "rmse_std": float(rmse_std),
                }
            }
        }

        with open(self.path / f"result_{model_key}.json", "w") as f:
            json.dump(result, f, indent=4)

        logger.info("saved cv json")
