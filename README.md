# fMRI-reconstruction-NSD

To create a conda environment that will run the notebooks and training scripts:
```bash
conda env create -f src/environment.yaml
conda activate medical-v1
```
The [setup.sh](./src/setup.sh) script list the conda and pip commands to create this environment. There's also a [Dockerfile](./src/Dockerfile) and docker image that was created with `make build push` on DockerHub at `jimgoo6/laion-fmri`.