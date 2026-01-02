import logging
from pathlib import Path
from typing import Union

from cleanZone import CleanZone

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("workflow")


class Workflow:
    def __init__(
        self,
        input_path: Union[str, Path],
    ) -> None:
        self.input_path = Path(input_path)

    def run(self) -> None:
        logger.info("Start Process")
        logger.info("\n##########\nStart Clean Zone\n##########")
        CleanZone(self.input_path).run()

        logger.info("\n##########\nStart Feature Zone\n##########")


        logger.info("\n##########\nStart Shard Zone\n##########")


        logger.info("\n##########\nFinished Process\n##########")


def entrypoint():
    import argparse

    parser = argparse.ArgumentParser(description="Arguments for segmentator evaluation")

    parser.add_argument(
        "--input_path",
        help="Path to parent directory of the nifti files",
        required=True,
    )

    args = parser.parse_args()

    Workflow(
        input_path=args.input_path,
    ).run()


if __name__ == "__main__":
    entrypoint()
