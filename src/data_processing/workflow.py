import logging
import numpy as np
from pathlib import Path
import time
import torch
import torch.multiprocessing as mp
from typing import Union

from cleanZone import CleanZone
from featureZone import FeatureZone, worker
from splitData import SplitData
from splitTit import SplitTit

log_file = f"/hpc/gpfs2/home/u/boehrijo/logs/workflow_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("workflow")


class Workflow:
    def __init__(self, input_path : Union[str, Path]) -> None:
        self.input_path = Path(input_path)

    def run(self) -> None:
        """
        Starts all needed files for the data prozessing and featuring. Does prework for 
        multyprozess of the feature Zone.
        """
        logger.info("\n##########\nStart Clean Zone\n##########")
        CleanZone(self.input_path).run()
        
        logger.info("\n##########\nStart Split Data\n##########")
        SplitData(self.input_path).run()

        logger.info("\n##########\nStart Split Titration\n##########")
        SplitTit(self.input_path / 'clean').run()

        logger.info("\n##########\nStart Feature Zone\n##########")        
        for set_name in ["full_tit", "half_tit", "third_tit"]:
            fz = FeatureZone(self.input_path, "train", set_name, device=torch.device("cpu"))
            fz.calc_stats()

        self.run_featurezone_parallel()

        logger.info("\n##########\nFinished Process\n##########")

    def run_featurezone_parallel(self):
        """
        Defines all needed prozesses for the featuring and starts the prozesses with a spezified GPU.
        """
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            logger.warning("No GPU found → fallback to single process on CPU")
            self.run_featurezone_cpu()
            return

        tasks = []

        for set_name in ["full_tit", "half_tit", "third_tit"]:
            for split in ["train", "val", "test"]:
                logger.info(f"starting: {split} for: {set_name}")
                if set_name == "full_tit":
                    pat_folders = sorted([
                        f for f in (self.input_path / "clean" / split).iterdir()
                        if f.is_dir()
                    ])
                else:
                    pat_folders = sorted([
                        f for f in (self.input_path / set_name / split).iterdir()
                        if f.is_dir()
                    ])

                chunks = list(np.array_split(pat_folders, num_gpus))

                for gpu_id in range(num_gpus):
                    tasks.append((self.input_path, split, set_name, chunks[gpu_id]))

        gpu_tasks = [[] for _ in range(num_gpus)]
        for i, task in enumerate(tasks):
            if len(chunks[gpu_id]) > 0:
                gpu_tasks[i % num_gpus].append(task)

        mp.spawn(worker, args=(gpu_tasks,), nprocs=num_gpus)

def entrypoint():
    """
    Starting point of the whole data pipeline.

    Parameters
    ----------
    --input_path:   str
        Path to input parent directory of the input .mat files
    """
    import argparse

    parser = argparse.ArgumentParser(description="Path to input parent directory of the input .mat files")

    parser.add_argument(
        "--input_path",
        help="Path to input parent directory of the input .mat files",
        required=True,
    )

    args = parser.parse_args()

    Workflow(
        input_path=args.input_path,
    ).run()


if __name__ == "__main__":
    entrypoint()
