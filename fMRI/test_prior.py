import torch
import json
from dalle2_pytorch import DiffusionPriorNetwork, DiffusionPrior
from dalle2_pytorch.train_configs import DiffusionPriorNetworkConfig
from models import BrainDiffusionPrior
from utils import count_params

# wget https://huggingface.co/nousr/conditioned-prior/resolve/main/vit-l-14/aesthetic/best.pth
# wget https://huggingface.co/nousr/conditioned-prior/raw/main/vit-l-14/aesthetic/prior_config.json
# wget https://huggingface.co/nousr/conditioned-prior/resolve/main/vit-l-14/laion2b/ema855M.pth

if __name__=='__main__':
    
    config = json.load(open('/workspace/models/nousr/prior_config.json'))
    
    ## Missing key(s) in state_dict: "net.null_text_encodings", "net.null_text_embeds", "net.null_image_embed".
    # ckpt_path = '/workspace/models/nousr/best.pth'

    # Unexpected keys(s): "scaler", "optimizer", "model", "version", "step", "ema"
    #ckpt_path = '/workspace/models/nousr/ema855M.pth'

    device = 'cuda'

    # None is the default and breaks things
    config['prior']['net']['max_text_len'] = 256

    net_config = DiffusionPriorNetworkConfig(**config['prior']['net'])
    print('net_config')
    print(net_config)

    kwargs = config['prior']
    has_clip = kwargs.pop('clip') is not None
    kwargs.pop('net')
    clip = None
    if has_clip:
        # clip = self.clip.create()
        pass

    diffusion_prior_network = net_config.create()
    diffusion_prior = BrainDiffusionPrior(net = diffusion_prior_network, clip = clip, **kwargs).to(device)
    
    count_params(diffusion_prior)
    # param counts:
    # 101,894,032 total
    # 101,894,016 trainable

    ckpt = torch.load(ckpt_path, map_location=device)
    diffusion_prior.load_state_dict(ckpt) #, strict=False)
    
    # diffusion_prior = BrainDiffusionPrior.from_pretrained()