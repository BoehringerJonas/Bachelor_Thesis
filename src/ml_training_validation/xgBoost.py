import joblib
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_validate

import xgboost as xgb


logger = logging.getLogger("workflow")


class XGBoost:
    def __init__(self,x_full: np.array, y_full: np.array, x_train: np.array, y_train: np.array, x_val: np.array, y_val: np.array, x_test: np.array, y_test: np.array, pmean:float, pstd:float, path: Path) -> None:
        self.x_full = x_full
        self.y_full = y_full
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.pmean = pmean
        self.pstd = pstd
        self.path = path
        self.model = xgb.XGBRegressor(
            tree_method="hist",
            device="cuda",
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=42,
            reg_lambda=1.0
        )

    def run(self) -> None:
        """
        Train and test XGBoost model. Start plot and safe prozess
        """
        logger.info("Start XGBoost")
        param_dist = {
            "n_estimators": [300, 500, 1000],
            "max_depth": [3, 5, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.7, 1.0],
            "min_child_weight": [1, 3, 5],
        }

        search = RandomizedSearchCV(
            self.model,
            param_distributions=param_dist,
            n_iter=30,
            cv=4,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            random_state=42
        )
        search.fit(
            self.x_train,
            self.y_train,
        )
        self.model = search.best_estimator_
        logger.info(f"Best hyperparameters: {search.best_params_}")

        valid_params = xgb.XGBRegressor().get_params().keys()

        best_params = {
            k: v for k, v in search.best_params_.items()
            if k in valid_params
        }

        model = xgb.XGBRegressor(
            **best_params,
            tree_method="hist",
            callbacks=[xgb.callback.EarlyStopping(rounds=30)],
            device="cuda",
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=42
        )
        model.fit(
            self.x_train,
            self.y_train,
            eval_set=[(self.x_train, self.y_train), (self.x_val, self.y_val)],
            verbose=True
        )
        self.plot_early_stop(model)
        self.save_val(model)


    def plot_early_stop(self, model:xgb.XGBRegressor) -> None:
        """
        PLots the early Stopping of the normal trained XGBoost.

        Parameters
        ----------
        model:xgb.XGBRegressor
       
        """
        results = model.evals_result()

        train_loss = results["validation_0"]["rmse"]
        val_loss   = results["validation_1"]["rmse"]
        train_loss = train_loss * self.pstd
        val_loss = val_loss * self.pstd

        plt.figure(figsize=(8, 5))
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.plot(train_loss, label="Train")
        plt.plot(val_loss, label="Validation")

        best_iter = model.best_iteration

        plt.axvline(
            best_iter,
            linestyle="--",
            color="gray",
            label=f"Best Iteration ({best_iter})"
        )

        plt.title(f"XGBoost Training Curve")
        plt.xlabel("Boosting Round")
        plt.ylabel("RMSE")
        plt.legend()

        plt.savefig(self.path / f"XGB_loss.png", bbox_inches='tight')
        plt.close()

    def save_val(self, model:xgb.XGBRegressor) -> None:
        """
        Calculate Metrics, Plot prediction vs True and Safe to json, safe the model.

        Parameters
        ----------
        model:xgb.XGBRegressor

        """
        y_pred = model.predict(self.x_test)
        y_pred_real = y_pred * self.pstd + self.pmean
        y_true_real = self.y_test * self.pstd + self.pmean

        mae = mean_absolute_error(y_true_real, y_pred_real)
        mse = mean_squared_error(y_true_real, y_pred_real)
        rmse = np.sqrt(mse)
        medae = median_absolute_error(y_true_real, y_pred_real)
        r2 = r2_score(y_true_real, y_pred_real)

        model_key = f"XGB_{self.path.parent.name}_{self.path.name}"
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

        with open(self.path / f"result_{model_key}.json", "w") as f:
            json.dump(result, f, indent=4)


        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true_real, y_pred_real)
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predictions")
        ax.set_title(f"XGB Prediction vs True")
        fig.savefig(self.path / f"XGB_prediction.png", bbox_inches='tight')
        plt.close(fig)

        logger.info(f"XGBoost prediction saved to {self.path} / XGB_prediction.png")       

        joblib.dump(model, self.path / f"XGB_model.joblib")
        logger.info(f"XGBoost model saved to {self.path} / XGB_model.joblib")


