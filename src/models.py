import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import PIL

import clip


class Clipper(torch.nn.Module):
    def __init__(self, clip_variant, device=None, train_transforms=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        print(clip_variant, self.device)
        clip_model, _ = clip.load(clip_variant, device=self.device)
        clip_model.eval() # dont want to train model
        for param in clip_model.parameters():
            param.requires_grad = False # dont need to calculate gradients
            
        self.clip = clip_model
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073])
        self.std = np.array([0.26862954, 0.26130258, 0.27577711])
        self.normalize = transforms.Normalize(self.mean, self.std)
        self.denormalize = transforms.Normalize((-self.mean / self.std).tolist(), (1.0 / self.std).tolist())
        self.clip_size = (224,224)
        self.transforms = train_transforms

    def embed_image(self, image, return_norm=False):
        """Expects images in -1 to 1 range"""
        # import pdb; pdb.set_trace()
        clip_emb = nn.functional.interpolate(image.to(self.device), self.clip_size, mode="area")
        if self.transforms is not None:
            clip_emb = self.transforms(clip_emb)
        # clip_emb = (0.5*clip_emb + 0.5).clamp(0,1)
        clip_emb = self.clip.encode_image(self.normalize(clip_emb))
        # clip_emb = torch.clamp(clip_emb, -1.5, 1.5)
        clip_emb_norm = clip_emb.norm(2,dim=-1)
        clip_emb = nn.functional.normalize(clip_emb, dim=-1)
        if not return_norm:
            return clip_emb
        else:
             return clip_emb, clip_emb_norm

    def embed_text(self, text_samples):
        clip_text = clip.tokenize(text_samples).to(self.device)
        clip_text = self.clip.encode_text(clip_text)
        clip_text = torch.clamp(clip_text, -1.5, 1.5)
        clip_text = nn.functional.normalize(clip_text, dim=-1)
        return clip_text
    
    def embed_curated_annotations(self, annots):
        for i,b in enumerate(annots):
            t = ''
            while t == '':
                rand = torch.randint(5,(1,1))[0][0]
                t = b[0,rand]
            if i==0:
                txt = np.array(t)
            else:
                txt = np.vstack((txt,t))
        txt = txt.flatten()
        return self.embed_text(txt)
    

class BrainNetwork(nn.Module):
    # 133M
    def __init__(self, out_dim, in_dim=15724, h=4096):
        super().__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=0),
        #     # nn.Dropout1d(0.1),
        #     nn.Batchnorm1d(),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2, stride=2)
        # )
        self.conv = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(h),
            nn.Dropout(0.5),
        )
            
        
        self.lin = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(h),
                nn.Dropout(0.15)
            ) for _ in range(4)
        ])  
        
        self.lin1 = nn.Linear(4096,out_dim)
        
    def forward(self, x):
        '''
            bs, 1, 15724 -> bs, 32, h
            bs, 32, h -> bs, 32h
            b2, 32h -> bs, 768
        '''
        x = self.conv(x)  # bs, 4096
        residual = x
        for res_block in range(4):
            x = self.lin[res_block](x)
            x += residual
            residual = x
        x = x.reshape(len(x),-1)
        x = self.lin1(x)
        return x
    
class BrainNetworkLarge(nn.Module):
    # 235M
    def __init__(self, out_dim, in_dim=15724, h=4096):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.GELU(),
            nn.Dropout(0.5),
        )

        self.lins = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.15) if i!=0 else nn.Identity(),
                nn.Linear(h, h),
                nn.GELU(),
                nn.BatchNorm1d(h),
                nn.Dropout(0.25),
                # nn.Dropout(0.15),
                nn.Linear(h, h),
                nn.GELU(),
                nn.BatchNorm1d(h),
                # nn.Dropout(0.15),
            ) for i in range(5)
        ])  
        
        # zero init batchnorms
        for lin in self.lins:
            nn.init.constant_(lin[-1].weight, 0.0)
            # nn.init.constant_(lin[-2].weight, 0.0)
        
        self.lin1 = nn.Linear(4096, out_dim)
        
    def forward(self, x):
        x = self.conv(x)  # bs, 4096
        residual = x
        for lin in self.lins:
            x = lin(x)
            x += residual
            residual = x
        x = self.lin1(x)
        return 