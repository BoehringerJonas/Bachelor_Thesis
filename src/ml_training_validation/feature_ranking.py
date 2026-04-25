import joblib
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from mlp import MLP


logger = logging.getLogger("Workflow")

class FeatureRanking:
    def __init__(self, path:str, test_loader:DataLoader):
        self.path = Path(path)
        self.test_loader = test_loader


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        for mlf, tit, tab, targets in self.test_loader:
            mlf_shape = mlf.shape[1]
            tit_shape = tit.shape[1]
            tab_shape = tab.shape[1]
            break

        mlf_list, tit_list, tab_list, y_list = [], [], [], []

        for mlf, tit, tab, targets in self.test_loader:
            mlf_list.append(mlf)
            tit_list.append(tit)
            tab_list.append(tab)
            y_list.append(targets)

        mlf_test = torch.cat(mlf_list).to(self.device)
        tit_test = torch.cat(tit_list).to(self.device)
        tab_test = torch.cat(tab_list).to(self.device)
        y_test = torch.cat(y_list).to(self.device)

        # zusätzlich für RF/XGB
        x_test = torch.cat([mlf_test, tit_test, tab_test], dim=1)

        for result_file in self.path.rglob("*model*"):
            if "rf" in result_file.name and result_file.suffix == ".joblib":
                kind = "RF"
                model = joblib.load(result_file)
                importance = model.feature_importances_
                imp = {"mlf": importance[:mlf_shape].sum(),
                    "tit": importance[mlf_shape:mlf_shape+tit_shape].sum(),
                    "tab": importance[-tab_shape:].sum()}
            
                logger.info(f"Feature importance sum: {np.sum(model.feature_importances_)}")
                logger.info(f"Non-zero features: {np.count_nonzero(model.feature_importances_)}")

            elif "MLP" in result_file.name and result_file.suffix == ".pt":
                kind = "MLP"
                state_dict = torch.load(result_file, map_location=self.device)
                model = MLP(mlf_shape, tit_shape, tab_shape).to(self.device)
                model.load_state_dict(state_dict)
                model.eval()
                loss_fn = torch.nn.MSELoss()
                with torch.no_grad():
                    baseline = loss_fn(model(mlf_test, tit_test, tab_test), y_test).item()
                mlf_scores, tit_scores, tab_scores = [], [], []

                n_runs = 20  # stabiler als 1x permutieren

                for _ in range(n_runs):

                    # MLF permutieren
                    idx = torch.randperm(mlf_test.shape[0])
                    mlf_perm = mlf_test[idx]
                    loss_mlf = loss_fn(model(mlf_perm, tit_test, tab_test), y_test).item()

                    # TIT permutieren
                    idx = torch.randperm(tit_test.shape[0])
                    tit_perm = tit_test[idx]
                    loss_tit = loss_fn(model(mlf_test, tit_perm, tab_test), y_test).item()

                    # TAB permutieren
                    idx = torch.randperm(tab_test.shape[0])
                    tab_perm = tab_test[idx]
                    loss_tab = loss_fn(model(mlf_test, tit_test, tab_perm), y_test).item()

                    mlf_scores.append(loss_mlf - baseline)
                    tit_scores.append(loss_tit - baseline)
                    tab_scores.append(loss_tab - baseline)

                imp = {"mlf": float(np.mean(mlf_scores)),
                    "tit": float(np.mean(tit_scores)),
                    "tab": float(np.mean(tab_scores))}

            elif "XGB" in result_file.name and result_file.suffix == ".joblib":
                kind = "XGB"
                model = joblib.load(result_file)
                score_dict = model.get_booster().get_score(importance_type="gain")

                importance = np.zeros(x_test.shape[1])

                for k, v in score_dict.items():
                    idx = int(k[1:])
                    importance[idx] = v
                
                imp = {"mlf": importance[:mlf_shape].sum(),
                    "tit": importance[mlf_shape:mlf_shape+tit_shape].sum(),
                    "tab": importance[-tab_shape:].sum()}

            else:
                continue

            save_path = self.path / f"importance"
            save_path.mkdir(parents=True, exist_ok=True)

            out_file = save_path / f"{result_file.stem}_importance.json"

            with open(out_file, "w") as f:
                json.dump(imp, f, indent=4)

            logger.info(f"Saved {self.path.parent.name} {self.path.name} importance: {out_file}")

            labels = ["MLF", "TIT", "TAB"]
            values = [imp["mlf"], imp["tit"], imp["tab"]]

            plt.figure(figsize=(6, 4))
            plt.bar(labels, values)

            plt.ylabel("Importance")
            plt.title("Group Importance")

            plt.tight_layout()
            plt.savefig(save_path / f"{result_file.stem}_{kind}_importance_group.png")
            plt.close()


def entrypoint():
    """
    Starting point of the whole ml pipeline.

    Parameters
    ----------
    --path:  str
        Path to safe model and evaluation
    """
    import argparse

    parser = argparse.ArgumentParser(description="paths for in and output")

    parser.add_argument(
        "--path",
        help="Path to safe model and evaluation",
        required=True,
    )

    args = parser.parse_args()

    FeatureRanking(
        path=args.path
    ).run()

if __name__ == "__main__":
    entrypoint()

