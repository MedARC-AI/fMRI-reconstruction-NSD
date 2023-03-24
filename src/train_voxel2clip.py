# # Import packages & functions

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from info_nce import InfoNCE
from dalle2_pytorch import DiffusionPriorNetwork
import kornia
from kornia.augmentation.container import AugmentationSequential

import utils
from utils import torch_to_matplotlib, torch_to_Image
from models import Clipper, BrainNetwork, BrainDiffusionPrior
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
        help="image or text",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="output directory for logs and checkpoints",
    )
    parser.add_argument(
        "--wandb_log",
        type=bool,
        default=False,
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
        help="1 for flattened input, 3 for 3d input",
    )
    parser.add_argument(
        "--remote_data",
        type=bool,
        default=False,
        help="whether to pull data from huggingface",
    )
    parser.add_argument(
        "--wds_cache_dir",
        type=str,
        default='/tmp/wds-cache',
        help="directory for caching webdatasets fetched from huggingface",
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print('args', args)

    if args.modality == "text":
        is_text = True
    else:
        is_text = False
    clip_variant = "ViT-L/14" # ("RN50", "ViT-L/14", "ViT-B/32")
    clamp_embs = False # clamp embeddings to (-1.5, 1.5)
    seed = 42
    mixup_pct = 0.5
    use_image_aug = True
    
    resume_from_ckpt = False

    num_epochs = 120
    if args.voxel_dims == 1:
        batch_size = 300
    else:
        batch_size = 128

    lr_scheduler = 'cycle'
    initial_lr = 5e-4 # only used if lr_scheduler is 'fixed'
    max_lr = 3e-4

    ckpt_saving = True
    ckpt_interval = 10
    save_at_end = True
    if args.outdir is None:
        outdir = f'../train_logs/{args.model_name}'
    else:
        outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir,exist_ok=True)

    if use_image_aug:
        train_augs = AugmentationSequential(
            kornia.augmentation.RandomResizedCrop((224,224), (0.6,1), p=0.3),
            kornia.augmentation.Resize((224, 224)),
            kornia.augmentation.RandomHorizontalFlip(p=0.5),
            kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.3),
            kornia.augmentation.RandomGrayscale(p=0.3),
            data_keys=["input"],
        )
    else:
        train_augs = None
    
    # uses tf32 data type which is faster than standard float32
    torch.backends.cuda.matmul.allow_tf32 = True

    # Resume from ckpt? #
    if resume_from_ckpt:
        ckpt_path = '../train_logs/vox2clip_indiv/ckpt-voxel2clip-epoch029.pth'
    else:
        ckpt_path = 'none'

    # Multi-GPU config #
    accelerator = Accelerator()
    print = accelerator.print # only print if local_rank=0

    device = accelerator.device
    print("device:",device)

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
    print("distributed =",distributed,"num_devices =", num_devices, "local rank =", local_rank, "world size =", world_size)

    # need non-deterministic CuDNN for conv3D to work
    utils.seed_everything(seed, cudnn_deterministic=False)

    if args.modality=='text':
        print('Using CLIP-text, preparing COCO annotations...')
        import h5py
        # load COCO annotations curated in the same way as the mind_reader (Lin Sprague Singh) preprint
        f = h5py.File(os.path.join(args.h5_dir, 'COCO_73k_subj_indices.hdf5'), 'r')
        subj01_order = f['subj01'][:]
        f.close()
        annots = np.load(os.path.join(args.h5_dir, 'COCO_73k_annots_curated.npy'), allow_pickle=True)
        subj01_annots = annots[subj01_order]

    print('Pulling NSD webdataset data...')
    # local paths
    # data_commit = '9947586218b6b7c8cab804009ddca5045249a38d'
    # train_url = f"/fsx/proj-medarc/fmri/natural-scenes-dataset/{data_commit}/datasets_pscotti_naturalscenesdataset_resolve_{data_commit}_webdataset_train/train_subj01_{{0..49}}.tar"
    # val_url = f"/fsx/proj-medarc/fmri/natural-scenes-dataset/{data_commit}/datasets_pscotti_naturalscenesdataset_resolve_{data_commit}_webdataset_val/val_subj01_0.tar"
    # meta_url = None
    # num_train = num_val = None # None means use all samples as specified in webdataset metadata.json

    train_url = "{/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/train/train_subj01_{0..17}.tar,/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/val/val_subj01_0.tar}"
    val_url = "/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/test/test_subj01_{0..1}.tar"
    meta_url = "/fsx/proj-medarc/fmri/natural-scenes-dataset/webdataset_avg_split/metadata_subj01.json"
    num_train = 8559 + 300
    num_val = 982

    if args.remote_data:
        # data loaders will fetch chunks from huggingface
        old = '/fsx/proj-medarc/fmri/natural-scenes-dataset/'
        new = 'https://huggingface.co/datasets/pscotti/naturalscenesdataset/resolve/a8c6147/'
        train_url = train_url.replace(old, new)
        val_url = val_url.replace(old, new)
        meta_url = meta_url.replace(old, new)

    # which to use for the voxels
    if args.voxel_dims == 1:
        voxels_key = 'nsdgeneral.npy'
    elif args.voxel_dims == 3:
        voxels_key = 'wholebrain_3d.npy'
    else:
        raise Exception(f"voxel_dims must be 1 or 3, not {args.voxel_dims}")

    print('Prepping train and validation dataloaders...')
    train_dl, val_dl, num_train, num_val = utils.get_dataloaders(
        batch_size,
        num_devices=num_devices,
        num_workers=num_workers,
        train_url=train_url,
        val_url=val_url,
        meta_url=meta_url,
        num_train=num_train,
        num_val=num_val,
        val_batch_size=300,
        cache_dir=args.wds_cache_dir,
        seed=seed,
        voxels_key=voxels_key,
        local_rank=local_rank,
    )

    if args.voxel_dims == 3:
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

        print('Creating Clipper...')

    # Don't L2 norm the extracted CLIP embeddings since we want the prior 
    # to learn un-normed embeddings for usage with the SD image variation pipeline.
    clip_extractor = Clipper(clip_variant, clamp_embs=False, norm_embs=False, device=device, train_transforms=train_augs)

    print('Creating voxel2clip...')

    if args.voxel_dims == 1: # 1D data
        voxel2clip_kwargs = dict(out_dim=768)
        voxel2clip = BrainNetwork(**voxel2clip_kwargs)
    elif args.voxel_dims == 3: # 3D data
        voxel2clip_kwargs = dict(
            out_dim=768,
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

    no_decay = ['bias']
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
        state_dict = voxel2clip.state_dict()
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

        if args.wandb_log: 
            import wandb
            print(f"wandb {args.wandb_project} run {wandb_run}")
            wandb.login(host='https://stability.wandb.io')#, relogin=True)
            wandb_config = {
              "model_name": args.model_name,
              "modality": args.modality,
              "voxel_dims": args.voxel_dims,
              "clip_variant": clip_variant,
              "batch_size": batch_size,
              "num_epochs": num_epochs,
              "use_image_aug": use_image_aug,
              "max_lr": max_lr,
              "lr_scheduler": lr_scheduler,
              "clamp_embs": clamp_embs,
              "mixup_pct": mixup_pct,
              "num_train": num_train,
              "num_val": num_val,
              "seed": seed,
              "distributed": distributed,
              "num_devices": num_devices,
              "world_size": world_size,
              "resume_from_ckpt": resume_from_ckpt,
              "ckpt_path": ckpt_path,
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
    
    #-----MAIN------------------
    # need non-deterministic CuDNN for conv3D to work
    utils.seed_everything(seed, cudnn_deterministic=False)

    epoch = 0
    losses, val_losses, lrs = [], [], []
    best_val_loss = 1e9
    soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, num_epochs - int(mixup_pct * num_epochs))

    val_voxel0 = val_image0 = None

    # Optionally resume from checkpoint #
    if resume_from_ckpt:
        print("\n---resuming from ckpt_path---\n",ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=device)
        epoch = checkpoint['epoch']
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

    progress_bar = tqdm(range(epoch,num_epochs), ncols=1200, disable=(local_rank!=0))
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

            repeat_index = train_i % 3

            image = image.float()
            voxel = voxel.float()[:,repeat_index].float()

            if args.voxel_dims == 3:
                voxel = voxel[:,np.unique(x_inc),:,:]
                voxel = voxel[:,:,np.unique(y_inc),:]
                voxel = voxel[:,:,:,np.unique(z_inc)]

            if epoch < int(mixup_pct * num_epochs):
                voxel, perm, betas, select = utils.mixco(voxel)

            if is_text:
                trial = trial.cpu().numpy()
                annots = utils.select_annotations(subj01_annots[trial], random=True)
                clip_target = clip_extractor.embed_text(annots).float()
            else:
                clip_target = clip_extractor.embed_image(image).float()
            clip_target.to(voxel.dtype)

            clip_voxels = voxel2clip(voxel)

            if epoch < int(mixup_pct * num_epochs):
                loss = utils.mixco_nce(
                    nn.functional.normalize(clip_voxels, dim=-1), 
                    nn.functional.normalize(clip_target, dim=-1),
                    temp=0.006, perm=perm, betas=betas, select=select,
                    distributed=distributed, accelerator=accelerator, local_rank=local_rank)
            else:
                epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
                loss = utils.soft_clip_loss(
                    nn.functional.normalize(clip_voxels, dim=-1), 
                    nn.functional.normalize(clip_target, dim=-1),
                    temp=epoch_temp,
                    distributed=distributed, accelerator=accelerator)
            utils.check_loss(loss)

            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])

            if distributed:
                sims_base += F.cosine_similarity(accelerator.gather(clip_target),
                                                      accelerator.gather(clip_voxels)).mean().item()
            else:
                sims_base += F.cosine_similarity(clip_target,clip_voxels).mean().item()

            # forward and backward top 1 accuracy
            labels = torch.arange(len(clip_target)).to(device)
            fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target, clip_voxels), labels, k=1)
            bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels, clip_target), labels, k=1)

            accelerator.backward(loss)
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

        voxel2clip.eval()
        for val_i, (voxel, image, trial) in enumerate(val_dl): 
            with torch.no_grad():
                repeat_index = val_i % 3

                image = image.float()
                # voxel.shape: [bs, 3, 15724]
                voxel = voxel[:,repeat_index].float()

                if args.voxel_dims == 3:
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

                if distributed:
                    clip_voxels = voxel2clip.module(voxel)
                else:
                    clip_voxels = voxel2clip(voxel)

                if epoch < int(mixup_pct * num_epochs):
                    val_loss = utils.mixco_nce(
                        nn.functional.normalize(clip_voxels, dim=-1), 
                        nn.functional.normalize(clip_target, dim=-1),
                        temp=0.006, 
                        distributed=distributed, accelerator=accelerator, local_rank=local_rank)
                else:
                    epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
                    val_loss = utils.soft_clip_loss(
                        nn.functional.normalize(clip_voxels, dim=-1), 
                        nn.functional.normalize(clip_target, dim=-1),
                        temp=epoch_temp, 
                        distributed=distributed, accelerator=accelerator)
                utils.check_loss(val_loss)

                val_losses.append(val_loss.item())

                if distributed:
                    val_sims_base += F.cosine_similarity(accelerator.gather(clip_target),
                                                          accelerator.gather(clip_voxels)).mean().item()
                else:
                    val_sims_base += F.cosine_similarity(clip_target,clip_voxels).mean().item()

                labels = torch.arange(len(clip_target)).to(device)
                # clip, brain
                val_fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels, clip_target), labels, k=1)
                # brain, clip
                val_bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target, clip_voxels), labels, k=1)

        if local_rank==0:
            if (not save_at_end and ckpt_saving) or (save_at_end and epoch == num_epochs - 1):
                # save best model
                val_loss = np.mean(val_losses[-(val_i+1):])
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_ckpt('best')
                else:
                    print(f'not best - val_loss: {val_loss:.3f}, best_val_loss: {best_val_loss:.3f}')

            # Save model checkpoint every `ckpt_interval`` epochs or on the last epoch
            if (ckpt_interval is not None and (epoch + 1) % ckpt_interval == 0) or epoch == num_epochs - 1:
                save_ckpt(f'epoch{epoch:03d}')

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
            progress_bar.set_postfix(**logs)

            if args.wandb_log:
                wandb.log(logs)

        if distributed:
            dist.barrier()

    if args.wandb_log and local_rank==0:
        wandb.finish()

    print("\n===Finished!===\n")