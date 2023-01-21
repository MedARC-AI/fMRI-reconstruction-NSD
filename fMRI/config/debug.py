batch_size = 64
# clip_aug_prob = 0.05
clip_aug_prob = 0.0
clip_aug_mode = 'y'
first_batch = True
wandb_log = False
n_samples_save = 2
ckpt_saving = False
outdir = "~/data/neuro/models/prior/test"
num_epochs = 200

# python train_prior.py --clip_aug_mode='n' --pretrained=True --wandb_log=True --remote_data=True --timesteps=1000 --n_samples_save=2 --batch_size=64 --first_batch=True --num_epochs=3 --ckpt_saving=False --save_samples_at_end=True