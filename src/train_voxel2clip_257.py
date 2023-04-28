# # Import packages & functions

import os
import sys
import traceback
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import kornia
from kornia.augmentation.container import AugmentationSequential

import utils
from utils import torch_to_matplotlib, torch_to_Image
from models import Clipper, OpenClipper, BrainNetworkDETR2, BrainNetworkNoDETR, BrainDiffusionPrior, BrainSD, BrainVD
from model3d import SimpleVoxel3dConvEncoder

import torch.distributed as dist
from accelerate import Accelerator

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train voxel2clip")
    parser.add_argument(
        "--model_name",
        type=str,
        default="voxel2clip-test",
        help="name of model, used for wandb logging",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="image",
        choices=["image", "text"],
        help="image or text",
    )
    parser.add_argument(
        "--clip_variant",
        type=str,
        default="ViT-L/14",
        choices=["RN50", "ViT-L/14", "ViT-B/32"],
        help='clip variant',
    )
    # parser.add_argument(
    #     "--outdir",
    #     type=str,
    #     default=None,
    #     help="output directory for logs and checkpoints",
    # )
    parser.add_argument(
        "--wandb_log",
        action="store_true",
        help="whether to log to wandb",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="stability",
        help="wandb project name",
    )
    parser.add_argument(
        "--h5_dir",
        type=str,
        default='/scratch/gpfs/KNORMAN/nsdgeneral_hdf5/',
        help="directory containing COCO h5 files (only used for modality=text)",
    )
    parser.add_argument(
        "--voxel_dims",
        type=int,
        default=1,
        choices=[1, 3],
        help="1 for flattened input, 3 for 3d input",
    )
    parser.add_argument(
        "--remote_data",
        action="store_true",
        help="whether to pull data from huggingface",
    )
    parser.add_argument(
        "--wds_cache_dir",
        type=str,
        default='/tmp/wds-cache',
        help="directory for caching webdatasets fetched from huggingface",
    )
    parser.add_argument(
        "--disable_image_aug",
        action="store_true",
        help="whether to disable image augmentation (only used for modality=image)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="output location",
    )
    parser.add_argument(
        "--use_mse",
        action="store_true",
        help="use mse loss",
    )
    parser.add_argument(
        "--cont_loss_type",
        type=str,
        default="all",
        choices=["all", "flatten"],
        help="loss type",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--mixup_pct",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--bidir_mixco",
        action="store_true",
        help="make mixco bidirectional"
    )
    return parser.parse_args()

if __name__ == '__main__':
    # raise ValueError()
    args = parse_args()
    print('args', args)

    model_name = args.model_name
    modality = args.modality
    image_var = 'images' if modality=='image' else None  # trial
    is_text = args.modality == "text"
    clip_variant = args.clip_variant  # "convnext_xxlarge"  # "ViT-L/14" # ("RN50", "ViT-L/14", "ViT-B/32")
    weights_path = None
    # weights_path = "../train_logs/models/convnext_xxlarge.bin"
    # clamp_embs = False # clamp embeddings to (-1.5, 1.5)
    remote_data = args.remote_data
    data_commit = 'avg'
    voxel_dims = args.voxel_dims # 1 for flattened 3 for 3d
    # -----------------------------------------------------------------------------
    # params for all models
    seed = 0
    batch_size = 300 if args.voxel_dims == 1 else 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

    num_epochs = 120  # 400  # 250  # 350 if data_commit=='avg' else 120
    lr_scheduler = 'cycle'
    initial_lr = 1e-3 #3e-5
    max_lr = 3e-4
    
    wandb_log = args.wandb_log
    wandb_project = 'laion-fmri'
    wandb_run_name = ''
    wandb_notes = ''
    first_batch = False
    ckpt_saving = True
    ckpt_interval = 10  # 10
    use_mp = False
    distributed = False
    save_at_end = False
    subj_id = '01'

    cache_dir = 'cache'
    n_cache_recs = 0
    mixup_pct = args.mixup_pct  # 0.5
    bidir_mixco = args.bidir_mixco
    use_full_trainset = True

    resume_from_ckpt = args.ckpt_path is not None
    use_mse = args.use_mse
    use_projector = False  # True
    normed_mse = True
    loss_type = args.cont_loss_type

    lr_scheduler = 'cycle'
    initial_lr = 5e-4 # only used if lr_scheduler is 'fixed'
    max_lr = 3e-4

    if args.outdir is None:
        # outdir = os.path.expanduser(f'../train_logs/models/{model_name}/test')
        outdir = f'../train_logs/models/{args.model_name}'
    else:
        outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    n_samples_save = 4
    
    # uses tf32 data type which is faster than standard float32
    torch.backends.cuda.matmul.allow_tf32 = True
    # need non-deterministic CuDNN for conv3D to work
    utils.seed_everything(seed, cudnn_deterministic=False)
    
    num_devices = torch.cuda.device_count()
    if num_devices==0: num_devices = 1
    num_workers = num_devices * 4

    if not args.disable_image_aug:
        # [color augs, spartial_augs]
        # train_augs = [
        #     AugmentationSequential(
        #         kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        #         kornia.augmentation.RandomSolarize(p=0.2),
        #         kornia.augmentation.RandomGrayscale(p=0.2),
        #         data_keys=["input"],
        #         # random_apply = (1,4)
        #     ),
        #     AugmentationSequential(
        #         kornia.augmentation.RandomResizedCrop((224,224), (0.5, 1), p=0.5),
        #         kornia.augmentation.RandomHorizontalFlip(p=0.5),
        #         data_keys=["input"],
        #         # random_apply = (1,4)
        #     )
        # ]
        train_augs = AugmentationSequential(
            # kornia.augmentation.RandomCrop((140, 140), p=0.3),
            # kornia.augmentation.Resize((224, 224)),
            kornia.augmentation.RandomResizedCrop((224,224), (0.5,1), p=0.3),
            kornia.augmentation.RandomHorizontalFlip(p=0.5),
            kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.3),
            kornia.augmentation.RandomGrayscale(p=0.3),
            data_keys=["input"],
            # random_apply = (1,4)
        )
    else:
        train_augs = None

    # Multi-GPU config #
    accelerator = Accelerator()
    print = accelerator.print # only print if local_rank=0

    device = accelerator.device
    print("device:",device)

    if use_mse:
        vd_cache_dir = '/fsx/proj-medarc/fmri/cache/models--shi-labs--versatile-diffusion/snapshots/2926f8e11ea526b562cd592b099fcf9c2985d0b7'
        vd_pipe =  BrainVD.from_pretrained(
            # "lambdalabs/sd-image-variations-diffusers",
            vd_cache_dir,
            safety_checker=None,
            requires_safety_checker=False,
            torch_dtype=torch.float32, # fp16 is fine if we're not training this
        ).to("cpu")

        vd_pipe.text_encoder.eval()
        vd_pipe.text_encoder.requires_grad_(False)
        vd_pipe.image_encoder.eval()
        vd_pipe.image_encoder.requires_grad_(False)
        vd_pipe.text_unet.eval()
        vd_pipe.text_unet.requires_grad_(False)
        vd_pipe.image_unet.eval()
        vd_pipe.image_unet.requires_grad_(False)
        vd_pipe.vae.eval()
        vd_pipe.vae.requires_grad_(False)
    else:
        vd_pipe = None

    num_devices = torch.cuda.device_count()
    if num_devices==0: num_devices = 1
    num_workers = num_devices

    print(accelerator.state)
    local_rank = accelerator.state.local_process_index
    world_size = accelerator.state.num_processes
    if num_devices <= 1 and world_size <= 1:
        distributed = False
    else:
        distributed = True
    
    try:
        clip_extractor = Clipper(clip_variant, clamp_embs=False, norm_embs=False, hidden_state=True, refine=False, 
            device=device, train_transforms=train_augs)
        print('Creating Clipper...')
    except AssertionError:
        clip_extractor = OpenClipper(clip_variant, weights_path, clamp_embs=False, norm_embs=False, device=device, train_transforms=train_augs)
        print('Creating Open Clipper...')
    print("distributed =",distributed,"num_devices =", num_devices, "local rank =", local_rank, "world size =", world_size)

    if modality=='text':
        print('Using CLIP-text, preparing COCO annotations...')
        import h5py
        # load COCO annotations curated in the same way as the mind_reader (Lin Sprague Singh) preprint
        f = h5py.File(os.path.join(args.h5_dir, 'COCO_73k_subj_indices.hdf5'), 'r')
        subj01_order = f['subj01'][:]
        f.close()
        annots = np.load(os.path.join(args.h5_dir, 'COCO_73k_annots_curated.npy'), allow_pickle=True)
        subj01_annots = annots[subj01_order]

    print('Pulling NSD webdataset data...')
    if remote_data:
        # pull data directly from huggingface
        train_url, val_url = utils.get_huggingface_urls(data_commit)
        meta_url = None
    else:
        # local paths
        if data_commit == 'avg':
            train_url = f"/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/train/train_subj{subj_id}_{{0..17}}.tar"
            val_url = f"/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/val/val_subj{subj_id}_0.tar"
            if use_full_trainset:
                train_url = "{/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/train/train_subj01_{0..17}.tar,/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/val/val_subj01_0.tar}"
                val_url = "/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/test/test_subj01_{0..1}.tar"
                meta_url = "/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/metadata_subj01.json"
        elif data_commit == 'indiv':
            train_url = f"/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_indiv_split/train/train_subj{subj_id}_{{0..49}}.tar"
            val_url = f"/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_indiv_split/val/val_subj{subj_id}_0.tar"
        else:
            train_url = f"/fsx/proj-medarc/fmri/natural-scenes-dataset/{data_commit}/datasets_pscotti_naturalscenesdataset_resolve_{data_commit}_webdataset_train/train_subj01_{{0..49}}.tar"
            val_url = f"/fsx/proj-medarc/fmri/natural-scenes-dataset/{data_commit}/datasets_pscotti_naturalscenesdataset_resolve_{data_commit}_webdataset_val/val_subj01_0.tar"
        meta_url = f"/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/metadata_subj{subj_id}.json"

    # which to use for the voxels
    if voxel_dims == 1:
        voxels_key = 'nsdgeneral.npy'
    elif voxel_dims == 3:
        voxels_key = 'wholebrain_3d.npy'
    else:
        raise Exception(f"voxel_dims must be 1 or 3, not {voxel_dims}")

    print('Prepping train and validation dataloaders...')
    if use_full_trainset:
        # combines train and val so meta is not valid anymore
        num_train = 8559 + 300
        num_val = 982
    else:
        num_train = None
        num_val = None
    train_dl, val_dl, num_train, num_val = utils.get_dataloaders(
        batch_size,
        num_devices=num_devices,
        num_workers=num_workers,
        train_url=train_url,
        val_url=val_url,
        meta_url=meta_url,
        val_batch_size=300,
        cache_dir=args.wds_cache_dir,
        seed=seed,
        voxels_key=voxels_key,
        local_rank=local_rank,
        num_train=num_train,
        num_val=num_val
    )

    if voxel_dims == 3:
        import nibabel as nib
        noise_ceils_path = '/fsx/proj-medarc/fmri/natural-scenes-dataset/temp_s3/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/ncsnr.nii.gz'
        noise_ceils = nib.load(noise_ceils_path).get_fdata()
        # plt.plot(np.sort(noise_ceils.flatten()))
        # plt.show()
        x_inc,y_inc,z_inc = np.where(noise_ceils > .5)

        # check that your data loader is working and save voxel shape after excluding low signal voxels
        for val_i, (voxel, img_input, key) in enumerate(val_dl):
            voxel = voxel[:,:,np.unique(x_inc),:,:]
            voxel = voxel[:,:,:,np.unique(y_inc),:]
            voxel = voxel[:,:,:,:,np.unique(z_inc)]
            print("voxel.shape", voxel.shape) # voxel.shape torch.Size([300, 3, 68, 64, 47])
            break

    print('Creating voxel2clip...')

    # size of the CLIP embedding for each variant
    clip_sizes = {"RN50": 1024, "ViT-L/14": 768, "ViT-B/32": 512}
    # output dim for voxel2clip model
    out_dim = clip_sizes[clip_variant]

    if voxel_dims == 1: # 1D data
        # voxel2clip_kwargs = dict(out_dim=768, norm_type='bn')
        # voxel2clip_kwargs = dict(out_dim=out_dim, norm_type='ln', act_first=False, encoder_tokens=64)
        # voxel2clip = BrainNetworkDETR2(**voxel2clip_kwargs)
        voxel2clip_kwargs = dict(out_dim=out_dim, norm_type='ln', act_first=False, encoder_tokens=257, use_projector=use_mse and use_projector)
        voxel2clip = BrainNetworkNoDETR(**voxel2clip_kwargs)
    elif args.voxel_dims == 3: # 3D data
        voxel2clip_kwargs = dict(
            out_dim=out_dim,
            dims=voxel.shape[2:],
            channels=[64, 128, 256, 128],
            strides=[1, 2, 3, 3],
            padding=[1, 1, 1, 1],
            dilation=[1, 1, 1, 1],
            kernel=[3, 3, 3, 3],
        )
        voxel2clip = SimpleVoxel3dConvEncoder(**voxel2clip_kwargs)  

    print("params of voxel2clip:")
    if local_rank==0:
        utils.count_params(voxel2clip)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    opt_grouped_parameters = [
        {'params': [p for n, p in voxel2clip.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in voxel2clip.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=initial_lr) # lr doesnt get used if lr_scheduler='cycle'

    if lr_scheduler == 'fixed':
        lr_scheduler = None
    elif lr_scheduler == 'cycle':
        global_batch_size = batch_size * num_devices
        total_steps = num_epochs*(num_train//global_batch_size)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=max_lr,
            total_steps=total_steps,
            final_div_factor=1000,
            last_epoch=-1, pct_start=2/num_epochs
        )

    def save_ckpt(tag):
        ckpt_path = os.path.join(outdir, f'{tag}.pth')
        print(f'saving {ckpt_path}',flush=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': voxel2clip.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': losses,
            'val_losses': val_losses,
            'fwd_percent_correct': fwd_percent_correct,
            'bwd_percent_correct': bwd_percent_correct,
            'val_fwd_percent_correct': val_fwd_percent_correct,
            'val_bwd_percent_correct': val_bwd_percent_correct,
            'lrs': lrs,
            }, ckpt_path)

    print("\nDone with model preparations!")
    
    #--------WANDB-----------------
    if local_rank==0 and args.wandb_log:
        wandb_run = args.model_name
        wandb_notes = ''

        import wandb
        print(f"wandb {args.wandb_project} run {wandb_run}")
        wandb.login(host='https://stability.wandb.io')#, relogin=True)
        wandb_config = {
            "model_name": args.model_name,
            "modality": args.modality,
            "voxel_dims": args.voxel_dims,
            "clip_variant": args.clip_variant,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "disable_image_aug": args.disable_image_aug,
            "max_lr": max_lr,
            "lr_scheduler": lr_scheduler,
            # "clamp_embs": clamp_embs,
            "mixup_pct": mixup_pct,
            "num_train": num_train,
            "num_val": num_val,
            "seed": seed,
            "distributed": distributed,
            "num_devices": num_devices,
            "world_size": world_size,
            # "resume_from_ckpt": resume_from_ckpt,
            # "ckpt_path": ckpt_path,
            "train_url": train_url,
            "val_url": val_url,
        }
        print("wandb_config:\n",wandb_config)
        wandb.init(
            project=args.wandb_project,
            name=wandb_run,
            config=wandb_config,
            notes=wandb_notes,
        )
            
    #----ACCELERATE------------
    voxel2clip, optimizer, train_dl, val_dl, lr_scheduler = accelerator.prepare(
        voxel2clip, optimizer, train_dl, val_dl, lr_scheduler
    )

    epoch = 0
    losses, mse_losses, val_losses, lrs = [], [], [], []
    best_val_loss = 1e9
    soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, num_epochs - int(mixup_pct * num_epochs))

    voxel0 = image0 = val_voxel0 = val_image0 = None

    # Optionally resume from checkpoint #
    if resume_from_ckpt:
        print("\n---resuming from ckpt_path---\n",args.ckpt_path)
        checkpoint = torch.load(args.ckpt_path, map_location=device)
        epoch = checkpoint['epoch']+1
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        try:
            voxel2clip.load_state_dict(checkpoint['model_state_dict'])
        except:
            state_dict = checkpoint['model_state_dict']
            for key in list(state_dict.keys()):
                if 'module.' in key:
                    state_dict[key.replace('module.', '')] = state_dict[key]
                    del state_dict[key]
            voxel2clip.load_state_dict(state_dict)
        global_batch_size = batch_size * num_devices
        total_steps_done = epoch*(num_train//global_batch_size)
        for _ in range(total_steps_done):
            lr_scheduler.step()
        del checkpoint
        torch.cuda.empty_cache()

    progress_bar = tqdm(range(epoch,num_epochs), disable=(local_rank!=0))
    for epoch in progress_bar:
        voxel2clip.train()

        sims = 0.
        sims_base = 0.
        val_sims = 0.
        val_sims_base = 0.
        fwd_percent_correct = 0.
        bwd_percent_correct = 0.
        val_fwd_percent_correct = 0.
        val_bwd_percent_correct = 0.

        for train_i, (voxel, image, trial) in enumerate(train_dl):
            optimizer.zero_grad()

            image = image.float()
            voxel = voxel.float()
            if voxel_dims == 1 and data_commit == 'avg':
                voxel = utils.voxel_select(voxel)
            if voxel_dims == 3:
                voxel = voxel[:,np.unique(x_inc),:,:]
                voxel = voxel[:,:,np.unique(y_inc),:]
                voxel = voxel[:,:,:,np.unique(z_inc)]

            if image0 is None:
                image0 = image.clone()
                voxel0 = voxel.clone()

            if epoch < int(mixup_pct * num_epochs):
                voxel, perm, betas, select = utils.mixco(voxel)

            if is_text:
                trial = trial.cpu().numpy()
                annots = utils.select_annotations(subj01_annots[trial], random=True)
                clip_target = clip_extractor.embed_text(annots).float()
            
            if voxel2clip.use_projector:
                clip_voxels_mse, clip_voxels = voxel2clip(voxel)
            else:
                clip_voxels = voxel2clip(voxel)
                clip_voxels_mse = clip_voxels

            if epoch < int(mixup_pct * num_epochs):
                if not is_text:
                    clip_target = clip_extractor.embed_image(image, apply_transforms=True, apply_spatial_transforms=False).float()
                clip_target.to(voxel.dtype)
                if loss_type == 'all':
                    loss = utils.mixco_nce_all(
                        nn.functional.normalize(clip_voxels, dim=-1), 
                        nn.functional.normalize(clip_target, dim=-1),
                        temp=0.006, perm=perm, betas=betas, select=select,
                        distributed=distributed, accelerator=accelerator, local_rank=local_rank)
                else:
                    loss = utils.mixco_nce(
                        nn.functional.normalize(clip_voxels.flatten(1), dim=-1), 
                        nn.functional.normalize(clip_target.flatten(1), dim=-1),
                        temp=0.006, perm=perm, betas=betas, select=select,
                        distributed=distributed, accelerator=accelerator, local_rank=local_rank,
                        bidirectional=bidir_mixco)
            else:
                if not is_text:
                    # clip_target = clip_extractor.embed_image(image, apply_transforms=False).float()
                    # clip_trans = clip_extractor.embed_image(image, apply_transforms=True, apply_spatial_transforms=True).float()
                    clip_target = clip_extractor.embed_image(image, apply_transforms=True, apply_spatial_transforms=True).float()

                epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
                if loss_type == 'all':
                    # loss = utils.soft_cont_loss_all(
                    #     nn.functional.normalize(clip_voxels, dim=-1), 
                    #     nn.functional.normalize(clip_target, dim=-1),
                    #     nn.functional.normalize(clip_trans, dim=-1),
                    #     temp=epoch_temp,
                    #     distributed=distributed, accelerator=accelerator)
                    loss = utils.soft_clip_loss_all(
                        nn.functional.normalize(clip_voxels, dim=-1), 
                        nn.functional.normalize(clip_target, dim=-1),
                        temp=epoch_temp,
                        distributed=distributed, accelerator=accelerator)
                else:
                    # loss = utils.soft_cont_loss(
                    #     nn.functional.normalize(clip_voxels.flatten(1), dim=-1), 
                    #     nn.functional.normalize(clip_target.flatten(1), dim=-1),
                    #     nn.functional.normalize(clip_trans.flatten(1), dim=-1),
                    #     temp=epoch_temp,
                    #     distributed=distributed, accelerator=accelerator)
                    loss = utils.soft_clip_loss(
                        nn.functional.normalize(clip_voxels.flatten(1), dim=-1), 
                        nn.functional.normalize(clip_target.flatten(1), dim=-1),
                        temp=epoch_temp,
                        distributed=distributed, accelerator=accelerator)
                

            if use_mse:
                if epoch < int(mixup_pct * num_epochs):
                    # apply mse to non-mixed voxels only
                    clip_target_mse = clip_extractor.embed_image(image[~select], apply_transforms=False).float()
                    if normed_mse:
                        clip_target_mse = clip_target_mse/(clip_target_mse[:, 0].norm(dim=-1).reshape(-1, 1, 1) + 1e-6)
                        # clip_voxels_mse = clip_voxels_mse/(clip_voxels_mse[:, 0].norm(dim=-1).reshape(-1, 1, 1) + 1e-6)
                    mse_loss = F.mse_loss(clip_voxels_mse[~select], clip_target_mse.float()) * (3000 if normed_mse else 30)
                else:
                    clip_target_mse = clip_target
                    if normed_mse:
                        clip_target_mse = clip_target_mse/(clip_target_mse[:, 0].norm(dim=-1).reshape(-1, 1, 1) + 1e-6)
                        # clip_voxels_mse = clip_voxels_mse/(clip_voxels_mse[:, 0].norm(dim=-1).reshape(-1, 1, 1) + 1e-6)
                    mse_loss = F.mse_loss(clip_voxels_mse, clip_target_mse.float()) * (3000 if normed_mse else 30)
                mse_losses.append(mse_loss.item())
            else:
                mse_loss = 0
            utils.check_loss(loss + mse_loss)

            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])

            if distributed:
                sims_base += F.cosine_similarity(accelerator.gather(clip_target),
                                                      accelerator.gather(clip_voxels), dim=-1).mean().item()
            else:
                sims_base += F.cosine_similarity(clip_target,clip_voxels, dim=-1).mean().item()

            # forward and backward top 1 accuracy
            labels = torch.arange(len(clip_target)).to(device)
            if loss_type == 'all':
                fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity_all(clip_target, clip_voxels), labels, k=1).item()
                bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity_all(clip_voxels, clip_target), labels, k=1).item()
            else:
                fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target.flatten(1), clip_voxels.flatten(1)), labels, k=1).item()
                bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels.flatten(1), clip_target.flatten(1)), labels, k=1).item()
            accelerator.backward(loss + mse_loss)
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

        voxel2clip.eval()
        for val_i, (voxel, image, trial) in enumerate(val_dl): 
            with torch.no_grad():
                image = image.float()
                voxel = voxel.float()
                if voxel_dims == 1 and data_commit == 'avg':
                    voxel = voxel.mean(1)
                if voxel_dims == 3:
                    voxel = voxel[:,np.unique(x_inc),:,:]
                    voxel = voxel[:,:,np.unique(y_inc),:]
                    voxel = voxel[:,:,:,np.unique(z_inc)]

                if val_image0 is None:
                    val_image0 = image.detach().clone()
                    val_voxel0 = voxel.detach().clone()

                if is_text:
                    trial = trial.cpu().numpy()
                    annots = utils.select_annotations(subj01_annots[trial], random=False)
                    clip_target = clip_extractor.embed_text(annots).float()
                else:
                    # turn off augmentations during validation
                    clip_target = clip_extractor.embed_image(image, apply_transforms=False).float()
                clip_target.to(voxel.dtype)

                if distributed:
                    module = voxel2clip.module
                else:
                    module = voxel2clip
                
                if module.use_projector:
                    _, clip_voxels = module(voxel)
                else:
                    clip_voxels = module(voxel)

                if epoch < int(mixup_pct * num_epochs):
                    if loss_type == 'all':
                        val_loss = utils.mixco_nce_all(
                            nn.functional.normalize(clip_voxels, dim=-1), 
                            nn.functional.normalize(clip_target, dim=-1),
                            temp=0.006, 
                            distributed=distributed, accelerator=accelerator, local_rank=local_rank)
                    else:
                        val_loss = utils.mixco_nce(
                            nn.functional.normalize(clip_voxels.flatten(1), dim=-1), 
                            nn.functional.normalize(clip_target.flatten(1), dim=-1),
                            temp=0.006, 
                            distributed=distributed, accelerator=accelerator, local_rank=local_rank,
                            bidirectional=bidir_mixco)
                else:
                    epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
                    if loss_type == 'all':
                        val_loss = utils.soft_clip_loss_all(
                            nn.functional.normalize(clip_voxels, dim=-1), 
                            nn.functional.normalize(clip_target, dim=-1),
                            temp=epoch_temp, 
                            distributed=distributed, accelerator=accelerator)
                    else:
                        val_loss = utils.soft_clip_loss(
                            nn.functional.normalize(clip_voxels.flatten(1), dim=-1), 
                            nn.functional.normalize(clip_target.flatten(1), dim=-1),
                            temp=epoch_temp, 
                            distributed=distributed, accelerator=accelerator)
                utils.check_loss(val_loss)

                val_losses.append(val_loss.item())

                if distributed:
                    val_sims_base += F.cosine_similarity(accelerator.gather(clip_target),
                                                          accelerator.gather(clip_voxels),dim=-1).mean().item()
                else:
                    val_sims_base += F.cosine_similarity(clip_target,clip_voxels,dim=-1).mean().item()

                labels = torch.arange(len(clip_target)).to(device)
                if loss_type == 'all':
                    val_fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity_all(clip_target, clip_voxels), labels, k=1).item()
                    val_bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity_all(clip_voxels, clip_target), labels, k=1).item()
                else:
                    val_fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target.flatten(1), clip_voxels.flatten(1)), labels, k=1).item()
                    val_bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels.flatten(1), clip_target.flatten(1)), labels, k=1).item()

        if local_rank == 0:
            if ckpt_saving:
                # save best model
                val_loss = np.mean(val_losses[-(val_i+1):])
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    try:
                        save_ckpt('best')
                    except:
                        pass
                else:
                    print(f'not best - val_loss: {val_loss:.3f}, best_val_loss: {best_val_loss:.3f}')

            # Save model checkpoint every `ckpt_interval`` epochs or on the last epoch
            if (ckpt_interval is not None and (epoch + 1) % ckpt_interval == 0) or epoch == num_epochs - 1:
                try:
                    save_ckpt(f'epoch{epoch:03d}')
                except:
                    pass

            logs = {"train/loss": np.mean(losses[-(train_i+1):]),
                    "val/loss": np.mean(val_losses[-(val_i+1):]),
                    "train/lr": lrs[-1],
                    "train/num_steps": len(losses),
                    "val/num_steps": len(val_losses),
                    "train/cosine_sim_base": sims_base / (train_i + 1),
                    "val/cosine_sim_base": val_sims_base / (val_i + 1),
                    "train/fwd_pct_correct": fwd_percent_correct / (train_i + 1),
                    "train/bwd_pct_correct": bwd_percent_correct / (train_i + 1),
                    "val/val_fwd_pct_correct": val_fwd_percent_correct / (val_i + 1),
                    "val/val_bwd_pct_correct": val_bwd_percent_correct / (val_i + 1)}
            if use_mse:
                logs['train/mse_loss'] = np.mean(mse_losses[-(train_i+1):])
            progress_bar.set_postfix(**logs)

            # sample some images
            if vd_pipe is not None:
                if (ckpt_interval is not None and (epoch + 1) % ckpt_interval == 0) or epoch == num_epochs - 1:
                    if (not save_at_end and n_samples_save > 0) or (save_at_end and epoch == num_epochs - 1):
                        vd_pipe.to(device)
                        vd_pipe.to(torch.float16)
                        # training
                        grids,_ = utils.vd_sample_images(
                            clip_extractor, module, vd_pipe, None,
                            voxel0[:n_samples_save], image0[:n_samples_save], seed=42,
                        )
                        for i, grid in enumerate(grids):
                            grid.save(os.path.join(outdir, f'samples-train-{i:03d}.png'))
                        if wandb_log:
                            logs['train/samples'] = [wandb.Image(grid) for grid in grids]

                        # validation
                        grids,_ = utils.vd_sample_images(
                            clip_extractor, module, vd_pipe, None,
                            val_voxel0[:n_samples_save], val_image0[:n_samples_save], seed=42,
                        )
                        for i, grid in enumerate(grids):
                            grid.save(os.path.join(outdir, f'samples-val-{i:03d}.png'))
                        if wandb_log:
                            logs['val/samples'] = [wandb.Image(grid) for grid in grids]
                    
                        del grids
                        vd_pipe.to(torch.float32)
                        vd_pipe.to('cpu')
                        
            
            if args.wandb_log:
                while True:
                    try:
                        wandb.log(logs)
                        break
                    except:
                        print('Wandb log failed. Retrying')
                        time.sleep(1)
            
            del clip_voxels, clip_target, image, voxel
            torch.cuda.empty_cache()

    if args.wandb_log and local_rank==0:
        wandb.finish()

    print("\n===Finished!===\n")