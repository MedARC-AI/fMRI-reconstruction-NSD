LAION's Replication Of: *Deep image Reconstruction From Human Brain Activity*
---

As we get the ball rolling i've added a notebook for those who wish to get familiar with some of the datatypes this project will be working with. 

To follow along you will need to grab a few files:
1. A ROI binary mask, which can be found [here](https://openneuro.org/datasets/ds001506/versions/1.3.1/file-display/sourcedata:sub-01:anat:sub-01_mask_RH_HVC.nii.gz)
2. An MRI scan, located [here](https://openneuro.org/datasets/ds001506/versions/1.3.1/file-display/sub-01:ses-imagery01:anat:sub-01_ses-imagery01_inplaneT2.nii.gz)
3. A training data sample, in [this repo](https://github.com/KamitaniLab/DeepImageReconstruction/tree/master/data/fmri)

### Project outline
- [x] Sowcase of fMRI datasets in [jupyter notebook](fMRI_FileTypes.ipynb)
- [x] [WebDataset conversion](fMRI_h5_to_wds.ipynb) for easy use in exisiting training setups
- [ ] [CLIP inference](fMRI_CLIP_inference.ipynb) on a sample fMRI record
- [ ] [Whole training example](fMRI_CLIP_training.ipynb) for CLIP fine-tuning
- [ ] Scaling up training runs with WandB
- [ ] In-project evaluation metrics
- [ ] Comparison with results of the previous paper

--- 
🚧 Under Construction 🚧
