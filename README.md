To create a conda environment that will run the notebooks:
```bash
conda create -n medical-v1 python=3.10
conda activate medical-v1
conda env update -f environment.yaml
# this one must be run individually since it uses git
pip install git+https://github.com/openai/CLIP.git
```
The `setup.sh` script list the conda and pip commands to create this environment.