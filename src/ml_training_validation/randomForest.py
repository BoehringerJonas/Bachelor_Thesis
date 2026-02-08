import joblib
import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("Random Forest")


class RandomForest:
    def __init__(self, x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array, path: Path) -> None:
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.path = path

        self.rf = RandomForestClassifier(
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
                scoring='accuracy',
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

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')

        logger.info(f"Model accuracy: {accuracy}")
        logger.info(f"Model precision: {precision}")
        logger.info(f"Model recall: {recall}")

        if getattr(self.rf, "oob_score_", None) is not None:
            logger.info(f"OOB Score: {self.rf.oob_score_}")

        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm)

        self.path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True)
        fig.savefig(self.path / "rf_confusion_matrix.png", bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Confusion matrix saved to {self.path} / rf_confusion_matrix.png")
