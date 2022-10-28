import torch, h5py, sys

sys.path.append('./stylegan_xl')
import dnnlib
import legacy

# Load pretrained model
network_pkl = 'imagenet256.pkl'
print('Loading networks from "%s"...' % network_pkl)
device = torch.device('cuda')
print('Using device:', device, file=sys.stderr)
with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema']
    G = G.eval().requires_grad_(False).to(device)  
    
import os
import torch
#os.environ['CUDA_HOME'] = '/opt/conda/pkgs/cudatoolkit-11.3.1-ha36c431_9'
x = torch.rand([1, 32, 512]).cuda()
pic = G.synthesis(x, noise_mode='const')
print(pic)
