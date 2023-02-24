voxel_dims = 3
batch_size = 64

voxel2clip_kwargs = dict(
    out_dim=768,
    dims=[83, 104, 81],
    attention_width=64,
    channels=[64, 128, 256, 64], # last one is attention_width
    strides=[1, 2, 3, 3],
    padding=[0, 0, 0, 0],
    dilation=[1, 1, 1, 1],
    kernel=[3, 3, 3, 3],
    average_output=False,
)

## debug
# first_batch = True
# save_at_end = True
# ckpt_saving = False
# num_epochs = 500
