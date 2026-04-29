import joblib
import json
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


logger = logging.getLogger("workflow")


class RandomForest:
    def __init__(self, x_train: np.array, y_train: np.array, x_test: np.array,
                y_test: np.array, pmean:float, pstd:float, path: Path) -> None:
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.pmean = pmean
        self.pstd = pstd
        self.path = path

        self.rf = RandomForestRegressor(
            n_estimators=500,
            oob_score=True,
            random_state=42,
            n_jobs=1
        )

    def run(self) -> None:
        """
        Tests for best parameters, train Random Forest on the best parameters and 
        renormalized the output PEEP. Calculates metrics and safe to json. Plots 
        prediction vs true.
        Parameters
        ----------
        tune_hyperparameters: bool = True
            optional for no tuned parameters
        """
        logger.info("Start Random Forest")
        param_dist = {
            "n_estimators": [300, 500, 1000, 1500],
            "max_depth": [5, 10, 20, 30, None],
            "max_features": ["sqrt", 0.8],
            "min_samples_split": [2, 3, 5],
            "min_samples_leaf": [1, 2],
            "bootstrap": [True]
        }

        search = RandomizedSearchCV(
            self.rf,
            param_distributions=param_dist,
            n_iter=50,
            cv=3,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            random_state=42
        )
        search.fit(self.x_train, self.y_train)
        self.rf = search.best_estimator_
        logger.info(f"Best hyperparameters: {search.best_params_}")

        joblib.dump(self.rf, self.path / "rf_model.joblib")
        logger.info(f"Random Forest model saved to {self.path} / rf_model.joblib")

        y_pred = self.rf.predict(self.x_test)

        y_pred_real = y_pred * self.pstd + self.pmean
        y_true_real = self.y_test * self.pstd + self.pmean

        mae = mean_absolute_error(y_true_real, y_pred_real)
        mse = mean_squared_error(y_true_real, y_pred_real)
        rmse = np.sqrt(mse)
        medae = median_absolute_error(y_true_real, y_pred_real)
        r2 = r2_score(y_true_real, y_pred_real)

        model_key = f"RF_{self.path.parent.name}_{self.path.name}"
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
        ax.set_title("RF Prediction vs True")

        fig.savefig(self.path / "RandomForest_prediction.png", bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Random Forest prediction plot saved to {self.path} / RandomForest_prediction.png")

