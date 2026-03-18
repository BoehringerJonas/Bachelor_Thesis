import pandas as pd
import numpy as np
import logging
from pathlib import Path
import shutil
from typing import Tuple


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("splitData")


class SplitData():
    def __init__(self, path:Path):
        self.path = Path(path)


    def run(self):
        table = pd.read_csv(self.path / "raw" / "MLF"/ "peep_tabular_data.csv")
        table = table[~table["Pat-ID"].str.startswith("AD")]
        pat = table[["Pat-ID", "PEEP"]].to_numpy()

        #data, pat = self.get_data()
        
        low = []
        mid = []
        high = []
        for id, p in pat:
            if(p <= 12):
                low.append(id)
            elif(12 < p < 19):
                mid.append(id)
            else:
                high.append(id)
        
        low1, low2, low3 = self.split(low, 1, 1)
        mid1, mid2, mid3 = self.split(mid, 2, 2)
        high1, high2, high3 = self.split(high, 1, 1)

        train = np.concatenate([low1, mid1, high1])
        val = np.concatenate([low2, mid2, high2])
        test = np.concatenate([low3, mid3, high3])

        print(f"train: {len(train)}, val: {len(val)}, test: {len(test)}" )

        tr_path = self.path / "clean" / "train"
        tr_path.mkdir(parents=True, exist_ok=True)
        vl_path = self.path / "clean" / "val"
        vl_path.mkdir(parents=True, exist_ok=True)
        te_path = self.path / "clean" / "test"
        te_path.mkdir(parents=True, exist_ok=True)

        for id in train:
            shutil.move(self.path / "clean" / id, tr_path)

        for id in val:
            shutil.move(self.path / "clean" / id, vl_path)

        for id in test:
            shutil.move(self.path / "clean" / id, te_path)


    def split(self, pats:np.array, n_val:int, n_test:int) -> Tuple[np.array, np.array, np.array]:
        pats = np.array(pats)
        np.random.seed(42)
        np.random.shuffle(pats)

        train = pats[n_test + n_val:]
        val = pats[n_test:n_test + n_val]
        test = pats[:n_test]

        return train, val, test
