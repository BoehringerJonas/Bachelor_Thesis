import logging
from pathlib import Path
from typing import Union

from cleanZone import CleanZone
from featureZone import FeatureZone
from splitData import SplitData
from splitTit import SplitTit

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("workflow")


class Workflow:
    def __init__(
        self,
        input_path : Union[str, Path],
    ) -> None:
        self.input_path = Path(input_path)

    def run(self) -> None:
        logger.info("\n##########\nStart Clean Zone\n##########")
        CleanZone(self.input_path).run()
        
        logger.info("\n##########\nStart Split Data\n##########")
        SplitData(self.input_path).run()

        logger.info("\n##########\nStart Split Titration\n##########")
        SplitTit(self.input_path / 'clean').run()
   
        logger.info("\n##########\nStart Feature Zone Full\n##########")
        FeatureZone(self.input_path, "train", "full_tit").run()
        FeatureZone(self.input_path, "val", "full_tit").run()
        FeatureZone(self.input_path, "test", "full_tit").run()

        logger.info("\n##########\nStart Feature Zone Half\n##########")
        FeatureZone(self.input_path, "train", "half_tit").run()
        FeatureZone(self.input_path, "val", "half_tit").run()
        FeatureZone(self.input_path, "test", "half_tit").run()

        logger.info("\n##########\nStart Feature Zone Third\n##########")
        FeatureZone(self.input_path, "train", "third_tit").run()
        FeatureZone(self.input_path, "val", "third_tit").run()
        FeatureZone(self.input_path, "test", "third_tit").run()

        logger.info("\n##########\nFinished Process\n##########")


def entrypoint():
    import argparse

    parser = argparse.ArgumentParser(description="Arguments for in- and output")

    parser.add_argument(
        "--input_path",
        help="Path to data directory",
        required=True,
    )

    args = parser.parse_args()

    Workflow(
        input_path=args.input_path,
    ).run()


if __name__ == "__main__":
    entrypoint()
