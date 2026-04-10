import joblib
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
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


    def run(self) -> None:
        """
        Train and test XGBoost model. Start plot and safe prozess
        """
        logger.info("Start XGBoost")
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            early_stopping_rounds=30,    #>5
            random_state=42
        )
        model.fit(
            self.x_train,
            self.y_train,
            eval_set=[(self.x_val, self.y_val)],
            verbose=False
        )
        self.plot_early_stop(model, "")
        self.save_val(model, "")

        logger.info("XGBoost CV")
        modelCV = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=42
        )

        results = cross_validate(
            modelCV, 
            self.x_full, self.y_full,
            cv=7,
            scoring="neg_mean_squared_error",
            return_estimator=True
        )

        best_idx = results["test_score"].argmax()
        best_model = results["estimator"][best_idx]
        self.save_val(best_model, "CV")

    def plot_early_stop(self, model:xgb.XGBRegressor, mode:str) -> None:
        """
        PLots the early Stopping of the normal trained XGBoost.

        Parameters
        ----------
        model:xgb.XGBRegressor
        mode:str
            CV or not CV
        
        """
        results = model.evals_result()

        train_metric = results['validation_0']['rmse']
        train_metric = train_metric * self.pstd + self.pmean


        plt.figure(figsize=(8, 5))
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.plot(train_metric , label="Train")
        best_iter = model.best_iteration
        plt.axvline(
            best_iter,
            linestyle="--",
            color="gray",
            label=f"Best Iteration ({best_iter})"
        )
        plt.title(f"{mode}XGBoost Training Curve")
        plt.xlabel("Boosting Round")
        plt.ylabel("metric")
        plt.savefig(self.path / f"{mode}XGB_loss.png", bbox_inches='tight')
        plt.close()

    def save_val(self, model:xgb.XGBRegressor, mode:str) -> None:
        """
        Calculate Metrics, Plot prediction vs True and Safe to json, safe the model.

        Parameters
        ----------
        model:xgb.XGBRegressor
        mode:str
            CV or not CV

        """
        y_pred = model.predict(self.x_test)
        y_pred_real = y_pred * self.pstd + self.pmean
        y_true_real = self.y_test * self.pstd + self.pmean

        mae = mean_absolute_error(y_true_real, y_pred_real)
        mse = mean_squared_error(y_true_real, y_pred_real)
        rmse = np.sqrt(mse)
        medae = median_absolute_error(y_true_real, y_pred_real)
        r2 = r2_score(y_true_real, y_pred_real)
        #logger.info(f"Model mean absolut error: {mae}")
        #logger.info(f"Model mean squared error: {mse}")
        #logger.info(f"Model median absolut error: {medae}")
        #logger.info(f"Model R2 score: {r2}")

        model_key = f"{mode}XGB_{self.path.parent.name}_{self.path.name}"
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
        ax.set_title(f"{mode}XGB Prediction vs True")
        fig.savefig(self.path / f"{mode}XGB_prediction.png", bbox_inches='tight')
        plt.close(fig)

        logger.info(f"{mode} XGBoost prediction saved to {self.path} / {mode}XGB_prediction.png")       

        joblib.dump(model, self.path / "XGB_model.joblib")
        logger.info(f"{mode} XGBoost model saved to {self.path} / {mode}XGB_model.joblib")


