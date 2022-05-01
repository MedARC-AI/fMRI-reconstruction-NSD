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
- [ ] [Preprocessing fMRI images](https://github.com/KamitaniLab/End2EndDeepImageReconstruction/blob/master/end2end_create_lmdb.py)
- [ ] [CLIP inference](fMRI_CLIP_inference.ipynb) on a sample fMRI record
- [ ] [Whole training example](fMRI_CLIP_training.ipynb) for CLIP fine-tuning
- [ ] Scaling up training runs with WandB
- [ ] In-project evaluation metrics
- [ ] Comparison with results of the previous paper

### Literature
- [Preprocessing script](https://github.com/KamitaniLab/End2EndDeepImageReconstruction/blob/master/end2end_create_lmdb.py)
- [Repository from 2019 fMRI paper](https://github.com/KamitaniLab/End2EndDeepImageReconstruction)
- [Related paper from 2019 from the same lab](https://www.frontiersin.org/articles/10.3389/fncom.2019.00021/full)
- [Recent preprint from the same lab](https://www.biorxiv.org/content/10.1101/2021.12.31.474501v1.full.pdf+html )
- [Overview of fine-tuned CLIP models](http://sujitpal.blogspot.com/2021/10/fine-tuning-openai-clip-for-different.html)
- [Elsevier CLIP-trained model for image-search](https://github.com/elsevierlabs-os/clip-image-search)

--- 
🚧 Under Construction 🚧
