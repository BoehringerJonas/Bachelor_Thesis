import joblib
import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("Random Forest")


class RandomForest:
    def __init__(self, x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array, pmean:float, pstd:float, path: Path) -> None:
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
            n_jobs=-1
        )

    def run(self, tune_hyperparameters: bool = False) -> None:
        logger.info("Start Random Forest")

        if tune_hyperparameters:
            param_dist = {
                "n_estimators": randint(100, 1000),
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": randint(2, 10),
                "min_samples_leaf": randint(1, 5)
            }
            search = RandomizedSearchCV(
                self.rf,
                param_distributions=param_dist,
                n_iter=20,
                scoring='neg_mean_squared_error',
                cv=5,
                random_state=42,
                n_jobs=-1
            )
            search.fit(self.x_train, self.y_train)
            self.rf = search.best_estimator_
            logger.info(f"Best hyperparameters: {search.best_params_}")
        else:
            self.rf.fit(self.x_train, self.y_train)

        joblib.dump(self.rf, self.path / "rf_model.joblib")
        logger.info(f"Random Forest model saved to {self.path} / rf_model.joblib")

        y_pred = self.rf.predict(self.x_test)

        if getattr(self.rf, "oob_score_", None) is not None:
            logger.info(f"OOB Score: {self.rf.oob_score_}")

        y_pred_real = y_pred * self.pstd + self.pmean
        y_true_real = self.y_true * self.pstd + self.pmean

        mae = mean_absolute_error(y_true_real, y_pred_real)
        rmse = np.sqrt(mean_squared_error(y_true_real, y_pred_real))

        logger.info(f"Model mean absolut error: {mae}")
        logger.info(f"Model mean squared error: {rmse}")

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.scatter(self.y_test, y_pred)
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predictions")
        ax.set_title("Prediction vs True")

        fig.savefig(self.path / "RandomForest_prediction.png", bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Random Forest prediction plot saved to {self.path} / RandomForest_prediction.png")

