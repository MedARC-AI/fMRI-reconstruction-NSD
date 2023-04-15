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
from models import Clipper, OpenClipper, BrainNetworkDETR, BrainDiffusionPrior, BrainSD, VersatileDiffusionPriorNetwork
from model3d import SimpleVoxel3dConvEncoder

import torch.distributed as dist
from accelerate import Accelerator

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train prior")
    parser.add_argument(
        "--model_name",
        type=str,
        default="prior_257_test",
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
    return parser.parse_args()

if __name__ == '__main__':
    # Multi-GPU config #
    accelerator = Accelerator()
    print = accelerator.print # only print if local_rank=0

    device = accelerator.device
    print("device:",device)

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
    batch_size = 32 if args.voxel_dims == 1 else 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

    num_epochs = 250  # 350 if data_commit=='avg' else 120
    lr_scheduler = 'cycle'
    initial_lr = 1e-3 #3e-5
    max_lr = 3e-4
    
    wandb_log = args.wandb_log
    wandb_project = 'laion-fmri'
    wandb_run_name = ''
    wandb_notes = ''
    first_batch = False
    ckpt_saving = True
    ckpt_interval = 10
    use_mp = False
    distributed = False
    save_at_end = False
    subj_id = '01'

    cache_dir = 'cache'
    n_cache_recs = 0
    mixup_pct = 0.5

    resume_from_ckpt = False
    use_mse = False

    lr_scheduler = 'cycle'
    initial_lr = 5e-4 # only used if lr_scheduler is 'fixed'
    max_lr = 3e-4
    alpha = 0.01

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
        train_augs = AugmentationSequential(
            # kornia.augmentation.RandomCrop((140, 140), p=0.3),
            # kornia.augmentation.Resize((224, 224)),
            kornia.augmentation.RandomResizedCrop((224,224), (0.8, 1), p=0.3),
            kornia.augmentation.RandomHorizontalFlip(p=0.5),
            kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.3),
            kornia.augmentation.RandomGrayscale(p=0.3),
            data_keys=["input"],
            # random_apply = (1,4)
        )
    else:
        train_augs = None

    sd_cache_dir = '/fsx/proj-medarc/fmri/atom/.cache/huggingface/diffusers/models--lambdalabs--sd-image-variations-diffusers/snapshots/a2a13984e57db80adcc9e3f85d568dcccb9b29fc/'
    sd_pipe =  BrainSD.from_pretrained(
        # "lambdalabs/sd-image-variations-diffusers",
        sd_cache_dir,
        revision="v2.0",
        safety_checker=None,
        requires_safety_checker=False,
        torch_dtype=torch.float16, # fp16 is fine if we're not training this
    ).to(device)

    assert sd_pipe.image_encoder.training == False
    sd_pipe.unet.eval()
    sd_pipe.unet.requires_grad_(False)
    sd_pipe.vae.eval()
    sd_pipe.vae.requires_grad_(False)

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
        voxel2clip_kwargs = dict(out_dim=out_dim, norm_type='ln', act_first=False)
        voxel2clip = BrainNetworkDETR(**voxel2clip_kwargs)
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

    # setup prior network
    depth = 6
    dim_head = 64
    heads = 12 # heads * dim_head = 12 * 64 = 768
    timesteps = 100
    prior_network = VersatileDiffusionPriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads
    ).to(device)

    # custom version that can fix seeds
    diffusion_prior = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=out_dim,
        condition_on_text_encodings=False,
        timesteps=timesteps,
        cond_drop_prob=0.2,
        image_embed_scale=None,
        voxel2clip=voxel2clip
    ).to(device)
    
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    opt_grouped_parameters = [
        {'params': [p for n, p in diffusion_prior.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in diffusion_prior.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
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
            'model_state_dict': diffusion_prior.state_dict(),
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
    diffusion_prior, optimizer, train_dl, val_dl, lr_scheduler = accelerator.prepare(
        diffusion_prior, optimizer, train_dl, val_dl, lr_scheduler
    )

    epoch = 0
    losses, mse_losses, val_losses, lrs = [], [], [], []
    best_val_loss = 1e9
    soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, num_epochs - int(mixup_pct * num_epochs))

    voxel0 = image0 = val_voxel0 = val_image0 = None

    # Optionally resume from checkpoint #
    # if resume_from_ckpt:
    #     print("\n---resuming from ckpt_path---\n",ckpt_path)
    #     checkpoint = torch.load(ckpt_path, map_location=device)
    #     epoch = checkpoint['epoch']
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     try:
    #         voxel2clip.load_state_dict(checkpoint['model_state_dict'])
    #     except:
    #         state_dict = checkpoint['model_state_dict']
    #         for key in list(state_dict.keys()):
    #             if 'module.' in key:
    #                 state_dict[key.replace('module.', '')] = state_dict[key]
    #                 del state_dict[key]
    #         voxel2clip.load_state_dict(state_dict)

    progress_bar = tqdm(range(epoch,num_epochs), disable=(local_rank!=0))
    for epoch in progress_bar:
        diffusion_prior.train()

        sims = 0.
        sims_base = 0.
        val_sims = 0.
        val_sims_base = 0.
        fwd_percent_correct = 0.
        bwd_percent_correct = 0.
        val_fwd_percent_correct = 0.
        val_bwd_percent_correct = 0.
        loss_nce_sum = 0.
        loss_prior_sum = 0.
        val_loss_nce_sum = 0.
        val_loss_prior_sum = 0.

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
            else:
                clip_target = clip_extractor.embed_image(image).float()
            clip_target.to(voxel.dtype)

            # mixup diffusion targets as well
            if epoch < int(mixup_pct * num_epochs):
                betas_shape = [-1] + [1]*(len(clip_target.shape)-1)
                clip_target_prior = clip_target * betas.reshape(*betas_shape) + clip_target[perm] * (1-betas.reshape(*betas_shape))
            else:
                clip_target_prior = clip_target
            loss, pred, clip_voxels = diffusion_prior(image_embed=clip_target_prior, voxel=voxel)

            # distributed is not implemented for _all loss functions
            if epoch < int(mixup_pct * num_epochs):
                loss_nce = utils.mixco_nce_all(
                    nn.functional.normalize(clip_voxels, dim=-1), 
                    nn.functional.normalize(clip_target, dim=-1),
                    temp=0.006, perm=perm, betas=betas, select=select,
                    distributed=distributed, accelerator=accelerator, local_rank=local_rank)
            else:
                epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
                loss_nce = utils.soft_clip_loss_all(
                    nn.functional.normalize(clip_voxels, dim=-1), 
                    nn.functional.normalize(clip_target, dim=-1),
                    temp=epoch_temp,
                    distributed=distributed, accelerator=accelerator)

            loss_nce_sum += loss_nce.item()
            loss_prior_sum += loss.item()
            
            # MSE and NCE are weighted equally at the beginning,
            # with alpha=0.01 we'll have something like .01*300 + .99*3 = 3 + 3
            loss = alpha * loss + (1-alpha) * loss_nce
            utils.check_loss(loss)
            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])

            if distributed:
                sims_base += F.cosine_similarity(accelerator.gather(clip_target),
                                                      accelerator.gather(clip_voxels), dim=-1).mean().item()
            else:
                sims_base += F.cosine_similarity(clip_target,clip_voxels, dim=-1).mean().item()

            # forward and backward top 1 accuracy
            labels = torch.arange(len(clip_target)).to(device)
            fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target.flatten(1), clip_voxels.flatten(1)), labels, k=1).item()
            bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels.flatten(1), clip_target.flatten(1)), labels, k=1).item()

            accelerator.backward(loss)
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()
        
        if local_rank==0: 
            diffusion_prior.eval()
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
                        clip_target = clip_extractor.embed_image(image).float()
                    clip_target.to(voxel.dtype)
                    loss, pred, clip_voxels = diffusion_prior(image_embed=clip_target, voxel=voxel)
                    if epoch < int(mixup_pct * num_epochs):
                        loss_nce = utils.mixco_nce_all(
                            nn.functional.normalize(clip_voxels, dim=-1), 
                            nn.functional.normalize(clip_target, dim=-1),
                            temp=0.006, distributed=False, accelerator=accelerator, local_rank=local_rank)
                    else:
                        epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
                        loss_nce = utils.soft_clip_loss_all(
                            nn.functional.normalize(clip_voxels, dim=-1), 
                            nn.functional.normalize(clip_target, dim=-1),
                            temp=epoch_temp, distributed=False, accelerator=accelerator)

                    val_loss_nce_sum += loss_nce.item()
                    val_loss_prior_sum += loss.item()
                    val_loss = alpha * loss + (1-alpha) * loss_nce
                    val_losses.append(val_loss.item())

                    if distributed:
                        val_sims_base += F.cosine_similarity(accelerator.gather(clip_target),
                                                            accelerator.gather(clip_voxels),dim=-1).mean().item()
                    else:
                        val_sims_base += F.cosine_similarity(clip_target,clip_voxels,dim=-1).mean().item()

                    labels = torch.arange(len(clip_target)).to(device)
                    # clip, brain
                    val_fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target.flatten(1), clip_voxels.flatten(1)), labels, k=1).item()
                    # brain, clip
                    val_bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels.flatten(1), clip_target.flatten(1)), labels, k=1).item()

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
                    "val/val_bwd_pct_correct": val_bwd_percent_correct / (val_i + 1),
                    "train/loss_nce": loss_nce_sum / (train_i + 1),
                    "train/loss_prior": loss_prior_sum / (train_i + 1),
                    "val/loss_nce": val_loss_nce_sum / (val_i + 1),
                    "val/loss_prior": val_loss_prior_sum / (val_i + 1),
                    "train/alpha": alpha,
                }
            progress_bar.set_postfix(**logs)

            # sample some images
            if sd_pipe is not None:
                if (ckpt_interval is not None and (epoch + 1) % ckpt_interval == 0) or epoch == num_epochs - 1:
                    if (not save_at_end and n_samples_save > 0) or (save_at_end and epoch == num_epochs - 1):
                        # training
                        grids,_ = utils.sample_images(
                            clip_extractor, diffusion_prior.voxel2clip, sd_pipe, diffusion_prior,
                            voxel0[:n_samples_save], image0[:n_samples_save], seed=42,
                        )
                        for i, grid in enumerate(grids):
                            grid.save(os.path.join(outdir, f'samples-train-{i:03d}.png'))
                        if wandb_log:
                            logs['train/samples'] = [wandb.Image(grid) for grid in grids]

                        # validation
                        grids,_ = utils.sample_images(
                            clip_extractor, diffusion_prior.voxel2clip, sd_pipe, diffusion_prior,
                            val_voxel0[:n_samples_save], val_image0[:n_samples_save], seed=42,
                        )
                        for i, grid in enumerate(grids):
                            grid.save(os.path.join(outdir, f'samples-val-{i:03d}.png'))
                        if wandb_log:
                            logs['val/samples'] = [wandb.Image(grid) for grid in grids]
            
            if args.wandb_log:
                while True:
                    try:
                        wandb.log(logs)
                        break
                    except:
                        print('Wandb log failed. Retrying')
                        time.sleep(1)

    if args.wandb_log and local_rank==0:
        wandb.finish()

    print("\n===Finished!===\n")