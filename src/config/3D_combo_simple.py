voxel_dims = 3
batch_size = 16

voxel2clip_arch = 

voxel2clip_kwargs = dict(
    arch='3dconv-simple',
    out_dim=768,
    dims=[83, 104, 81],
    # channels=[64, 128, 256, attention_width],
    # strides=[1, 2, 3, 3],
    # padding=[0, 0, 0, 0],
    # dilation=[1, 1, 1, 1],
    # kernel=[3, 3, 3, 3],
    channels=[64, 128, 256, 128],
    strides=[1, 2, 3, 3],
    padding=[1, 1, 1, 1],
    dilation=[1, 1, 1, 1],
    kernel=[3, 3, 3, 3],
)
