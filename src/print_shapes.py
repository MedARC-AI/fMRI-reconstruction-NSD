from glob import glob
import h5py

fs = sorted(glob('/fsx/proj-medarc/fmri/natural-scenes-dataset/hdf5/subj*_nsdgeneral.hdf5'))

print(fs)

for f in fs:
    with h5py.File(f, 'r') as hf:
        print(f, hf['voxels'].shape, list(hf.keys()))

"""
/fsx/proj-medarc/fmri/natural-scenes-dataset/hdf5/subj01_3D_nsdgeneral.hdf5 (27750, 42, 46, 61)
/fsx/proj-medarc/fmri/natural-scenes-dataset/hdf5/subj01_nsdgeneral.hdf5 (27750, 15724)
/fsx/proj-medarc/fmri/natural-scenes-dataset/hdf5/subj02_nsdgeneral.hdf5 (27750, 14278)
/fsx/proj-medarc/fmri/natural-scenes-dataset/hdf5/subj03_nsdgeneral.hdf5 (21750, 15226)
/fsx/proj-medarc/fmri/natural-scenes-dataset/hdf5/subj04_nsdgeneral.hdf5 (20250, 13153)
/fsx/proj-medarc/fmri/natural-scenes-dataset/hdf5/subj05_nsdgeneral.hdf5 (27750, 13039)
/fsx/proj-medarc/fmri/natural-scenes-dataset/hdf5/subj06_nsdgeneral.hdf5 (21750, 17907)
/fsx/proj-medarc/fmri/natural-scenes-dataset/hdf5/subj07_nsdgeneral.hdf5 (27750, 12682)
/fsx/proj-medarc/fmri/natural-scenes-dataset/hdf5/subj08_nsdgeneral.hdf5 (20250, 14386)
"""

"""
medical-v1) jimgoo@ip-172-31-32-196:/fsx/proj-medarc/fmri/natural-scenes-dataset/temp_s3$ ls
nsddata  nsddata_betas
(medical-v1) jimgoo@ip-172-31-32-196:/fsx/proj-medarc/fmri/natural-scenes-dataset/temp_s3$ cd nsddata
(medical-v1) jimgoo@ip-172-31-32-196:/fsx/proj-medarc/fmri/natural-scenes-dataset/temp_s3/nsddata$ ls
ppdata
(medical-v1) jimgoo@ip-172-31-32-196:/fsx/proj-medarc/fmri/natural-scenes-dataset/temp_s3/nsddata$ cd ppdata/
(medical-v1) jimgoo@ip-172-31-32-196:/fsx/proj-medarc/fmri/natural-scenes-dataset/temp_s3/nsddata/ppdata$ ls
subj01
(medical-v1) jimgoo@ip-172-31-32-196:/fsx/proj-medarc/fmri/natural-scenes-dataset/temp_s3/nsddata/ppdata$ cd subj01/
(medical-v1) jimgoo@ip-172-31-32-196:/fsx/proj-medarc/fmri/natural-scenes-dataset/temp_s3/nsddata/ppdata/subj01$ ls
anat  transforms
(medical-v1) jimgoo@ip-172-31-32-196:/fsx/proj-medarc/fmri/natural-scenes-dataset/temp_s3/nsddata/ppdata/subj01$ cd anat/
(medical-v1) jimgoo@ip-172-31-32-196:/fsx/proj-medarc/fmri/natural-scenes-dataset/temp_s3/nsddata/ppdata/subj01/anat$ ls
T1_0pt8_masked.nii.gz
(medical-v1) jimgoo@ip-172-31-32-196:/fsx/proj-medarc/fmri/natural-scenes-dataset/temp_s3/nsddata/ppdata/subj01/transforms$ ls
MNI-to-anat0pt5.nii.gz       anat1pt0-to-func1pt8.nii.gz  lh.MNI-to-white.mgz         lh.anat1pt0-to-layerB3.mgz  lh.func1pt8-to-white.mgz    rh.anat0pt8-to-layerB2.mgz  rh.func1pt0-to-pial.mgz
MNI-to-anat0pt8.nii.gz       func1pt0-to-MNI.nii.gz       lh.anat0pt5-to-layerB1.mgz  lh.anat1pt0-to-pial.mgz     lh.white-to-fsaverage.mgz   rh.anat0pt8-to-layerB3.mgz  rh.func1pt0-to-white.mgz
MNI-to-anat1pt0.nii.gz       func1pt0-to-anat0pt5.nii.gz  lh.anat0pt5-to-layerB2.mgz  lh.anat1pt0-to-white.mgz    rh.MNI-to-layerB1.mgz       rh.anat0pt8-to-pial.mgz     rh.func1pt8-to-layerB1.mgz
MNI-to-func1pt0.nii.gz       func1pt0-to-anat0pt8.nii.gz  lh.anat0pt5-to-layerB3.mgz  lh.fsaverage-to-white.mgz   rh.MNI-to-layerB2.mgz       rh.anat0pt8-to-white.mgz    rh.func1pt8-to-layerB2.mgz
MNI-to-func1pt8.nii.gz       func1pt0-to-anat1pt0.nii.gz  lh.anat0pt5-to-pial.mgz     lh.func1pt0-to-layerB1.mgz  rh.MNI-to-layerB3.mgz       rh.anat1pt0-to-layerB1.mgz  rh.func1pt8-to-layerB3.mgz
anat0pt5-to-MNI.nii.gz       func1pt8-to-MNI.nii.gz       lh.anat0pt5-to-white.mgz    lh.func1pt0-to-layerB2.mgz  rh.MNI-to-pial.mgz          rh.anat1pt0-to-layerB2.mgz  rh.func1pt8-to-pial.mgz
anat0pt5-to-func1pt0.nii.gz  func1pt8-to-anat0pt5.nii.gz  lh.anat0pt8-to-layerB1.mgz  lh.func1pt0-to-layerB3.mgz  rh.MNI-to-white.mgz         rh.anat1pt0-to-layerB3.mgz  rh.func1pt8-to-white.mgz
anat0pt5-to-func1pt8.nii.gz  func1pt8-to-anat0pt8.nii.gz  lh.anat0pt8-to-layerB2.mgz  lh.func1pt0-to-pial.mgz     rh.anat0pt5-to-layerB1.mgz  rh.anat1pt0-to-pial.mgz     rh.white-to-fsaverage.mgz
anat0pt8-to-MNI.nii.gz       func1pt8-to-anat1pt0.nii.gz  lh.anat0pt8-to-layerB3.mgz  lh.func1pt0-to-white.mgz    rh.anat0pt5-to-layerB2.mgz  rh.anat1pt0-to-white.mgz
anat0pt8-to-func1pt0.nii.gz  lh.MNI-to-layerB1.mgz        lh.anat0pt8-to-pial.mgz     lh.func1pt8-to-layerB1.mgz  rh.anat0pt5-to-layerB3.mgz  rh.fsaverage-to-white.mgz
anat0pt8-to-func1pt8.nii.gz  lh.MNI-to-layerB2.mgz        lh.anat0pt8-to-white.mgz    lh.func1pt8-to-layerB2.mgz  rh.anat0pt5-to-pial.mgz     rh.func1pt0-to-layerB1.mgz
anat1pt0-to-MNI.nii.gz       lh.MNI-to-layerB3.mgz        lh.anat1pt0-to-layerB1.mgz  lh.func1pt8-to-layerB3.mgz  rh.anat0pt5-to-white.mgz    rh.func1pt0-to-layerB2.mgz
anat1pt0-to-func1pt0.nii.gz  lh.MNI-to-pial.mgz           lh.anat1pt0-to-layerB2.mgz  lh.func1pt8-to-pial.mgz     rh.anat0pt8-to-layerB1.mgz  rh.func1pt0-to-layerB3.mgz
(medical-v1) jimgoo@ip-172-31-32-196:/fsx/proj-medarc/fmri/natural-scenes-dataset/temp_s3/nsddata_betas/ppdata/subj01/func1pt8mm$ ls
betas_fithrf_GLMdenoise_RR
"""