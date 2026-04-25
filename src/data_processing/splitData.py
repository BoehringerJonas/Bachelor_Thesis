import pandas as pd
import numpy as np
import logging
from pathlib import Path
import shutil
import time
from typing import Tuple


logger = logging.getLogger("workflow")


class SplitData():
    def __init__(self, path:Path):
        self.path = Path(path)


    def run(self) -> None:
        """
        Collects all patients, devides them by low, mid, high PEEP and and splits the data 
        into train, test and val. Test and val are getting 4 patients where 1 patient 
        with low and 1 with high PEEP is in val and test. Rest of the patients is going into 
        train. Moves patients into there specified folders.
        """
        table = pd.read_csv(self.path / "raw" / "MLF"/ "peep_tabular_data.csv")
        table = table[~table["Pat-ID"].str.startswith("AD")]
        pat = table[["Pat-ID", "PEEP"]].to_numpy()
        
        lowest = []
        low = []
        mid = []
        high = []
        for id, p in pat:
            if p == 8:
                lowest.append(id)
            elif(8 < p <= 13):
                low.append(id)
            elif(13 < p < 21):
                mid.append(id)
            else:
                high.append(id)
        
        low1, low2, low3 = self.split(low, 1, 1)
        mid1, mid2, mid3 = self.split(mid, 2, 2)
        high1, high2, high3 = self.split(high, 1, 1)

        train = np.concatenate([lowest, low1, mid1, high1])
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
        """
        Parameters
        ----------
        pats    : np.array
            patients with low, mid or high PEEP
        n_val   : int
            cont how much of low, mid or high is going into val set
        n_test  : int
            cont how much of low, mid or high is going into test set

        Returns
        -------
        np.array : array with patients for train set
        np.array : array with patients for val set
        np.array : array with patients for test set

        """
        pats = np.array(pats)
        rng = np.random.RandomState(42)
        rng.shuffle(pats)

        train = pats[n_test + n_val:]
        val = pats[n_test:n_test + n_val]
        test = pats[:n_test]

        return train, val, test
