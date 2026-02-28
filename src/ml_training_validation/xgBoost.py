import joblib
import logging
import matplotlib as plt
import numpy as np

from scipy.stats import uniform, randint

from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split

import xgboost as xgb


logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("Random Forest")


class XGBoost:
    def __init__(self, x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array, path: Path) -> None:
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.path = path

        self.xgb_model = xgb.XGBRegressor(objective="reg:squarederror", 
                                          eval_metric="rmse",
                                          random_state=42)

    def run(self):
        logger.info("Start XGBoost")
        xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42, eval_metric="auc")

        xgb_model.fit(self.x_train, self.y_train, early_stopping_rounds=5, eval_set=[(self.x_test, self.y_test)])

        y_pred = xgb_model.predict(self.x_test)

        joblib.dump(self.rf, self.path / "XGB_model.joblib")
        logger.info(f"XGBoost model saved to {self.path} / XGB_model.joblib")

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
        fig.savefig(self.path / "XGB_confusion_matrix.png", bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Confusion matrix saved to {self.path} / XGB_confusion_matrix.png")        