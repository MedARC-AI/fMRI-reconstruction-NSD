# fMRI-reconstruction-NSD

To create a conda environment that will run the notebooks and training scripts:
```bash
conda env create -f src/environment.yaml
conda activate medical-v1
```
The [setup.sh](./src/setup.sh) script list the conda and pip commands to create this environment. There's also a [Dockerfile](./src/Dockerfile) and docker image that was created with `make build push` on DockerHub at `jimgoo6/laion-fmri`.

To use the pretrained diffusion prior weights from LAION 2B, run the `download.sh` script to get the files from HuggingFace. For more info on how that model was trained, see [https://huggingface.co/nousr/conditioned-prior/](https://huggingface.co/nousr/conditioned-prior/).
