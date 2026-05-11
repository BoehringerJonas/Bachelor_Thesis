# BA Code

## Installation
python -m venv venv
source venv\Scripts\activate

```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

## Usage
To start the data and traings pipeline run the slurm skripts or use the start code below. For the .slurm skripts please change starting command.

data pipeline: 
    sbatch start_data.slurm
    python workflow.py --input_path Path/to/data/dir

ml pipeline
    sbatch start_data_test.slurm
    python workflow.py --input_path PAth/to/feature/dir --output_path Path/to/outpu/dir

Other skripts: Use the starting command or the start_data_test.slurm test skript with changed starting command:

    python Path/to/base_model.py
    python Path/to/nan_images_creation.py
    python Path/to/splitTit.py --path Path/to/clean/patient/data
