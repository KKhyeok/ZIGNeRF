import torch
import torch.nn as nn
from math import log2
from im2scene.layers import ResnetBlock
import torch.nn.utils.spectral_norm as spectral_norm
from im2scene.utils import ImgEncoderwoSN, ImgEncoder
import torch.nn.functional as F


class DCDiscriminator(nn.Module):
    ''' DC Discriminator class.

    Args:
        in_dim (int): input dimension
        n_feat (int): features of final hidden layer
        img_size (int): input image size
    '''
    def __init__(self, in_dim=3, n_feat=512, img_size=64):
        super(DCDiscriminator, self).__init__()

        self.in_dim = in_dim
        n_layers = int(log2(img_size) - 2)
        self.blocks = nn.ModuleList(
            [nn.Conv2d(
                in_dim,
                int(n_feat / (2 ** (n_layers - 1))),
                4, 2, 1, bias=False)] + [nn.Conv2d(
                    int(n_feat / (2 ** (n_layers - i))),
                    int(n_feat / (2 ** (n_layers - 1 - i))),
                    4, 2, 1, bias=False) for i in range(1, n_layers)])

        self.conv_out = nn.Conv2d(n_feat, 1, 4, 1, 0, bias=False)
        self.actvn = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, **kwargs):
        batch_size = x.shape[0]
        if x.shape[1] != self.in_dim:
            x = x[:, :self.in_dim]
        for layer in self.blocks:
            x = self.actvn(layer(x))

        out = self.conv_out(x)
        out = out.reshape(batch_size, 1)
        return out


class DCDiscriminatorCond(nn.Module):
    ''' DC Discriminator class.

    Args:
        in_dim (int): input dimension
        n_feat (int): features of final hidden layer
        img_size (int): input image size
    '''
    def __init__(self, in_dim=3, n_feat=512, img_size=64, cond=True):
        super(DCDiscriminatorCond, self).__init__()

        self.in_dim = in_dim
        self.n_feat = n_feat
        n_layers = int(log2(img_size) - 2)
        self.blocks = nn.ModuleList(
            [spectral_norm(nn.Conv2d(
                in_dim,
                int(n_feat / (2 ** (n_layers - 1))),
                4, 2, 1, bias=False))] + [spectral_norm(nn.Conv2d(
                    int(n_feat / (2 ** (n_layers - i))),
                    int(n_feat / (2 ** (n_layers - 1 - i))),
                    4, 2, 1, bias=False)) for i in range(1, n_layers)])

        self.conv_out = spectral_norm(nn.Conv2d(n_feat, 1, 4, 1, 0, bias=False))
        self.actvn = nn.LeakyReLU(0.2, inplace=True)
        if cond:
            self.cond_encoder = spectral_norm(nn.Linear(40, self.n_feat))
            nn.init.normal_(self.cond_encoder.weight, 0.0, 0.05)
            nn.init.constant_(self.cond_encoder.bias, 0.)

    def forward(self, x, cond_data = None, **kwargs):
        batch_size = x.shape[0]
        if x.shape[1] != self.in_dim:
            x = x[:, :self.in_dim]
        for layer in self.blocks:
            x = self.actvn(layer(x))
        if cond_data is not None:
            out_cond = nn.functional.avg_pool2d(x*self.cond_encoder(cond_data).view([batch_size, self.n_feat, 1, 1]).expand_as(x), 4).squeeze()
        out = self.conv_out(x)
        out = out.reshape(batch_size, 1)
        if cond_data is not None:
            return out + out_cond.sum(dim=1)
        else:
            return out

class DiscriminatorResnet(nn.Module):
    ''' ResNet Discriminator class.

    Adopted from: https://github.com/LMescheder/GAN_stability

    Args:
        img_size (int): input image size
        nfilter (int): first hidden features
        nfilter_max (int): maximum hidden features
    '''
    def __init__(self, image_size, nfilter=16, nfilter_max=512):
        super().__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        size = image_size

        # Submodules
        nlayers = int(log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks = [
            ResnetBlock(nf, nf)
        ]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        self.conv_img = nn.Conv2d(3, 1*nf, 3, padding=1)
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.nf0*s0*s0, 1)
        self.actvn = nn.LeakyReLU(0.2)

    def forward(self, x, **kwargs):
        batch_size = x.size(0)

        out = self.conv_img(x)
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0*self.s0*self.s0)
        out = self.fc(self.actvn(out))
        return out


class DiscriminatorResnetCond(nn.Module):
    ''' ResNet Discriminator class.

    Adopted from: https://github.com/LMescheder/GAN_stability

    Args:
        img_size (int): input image size
        nfilter (int): first hidden features
        nfilter_max (int): maximum hidden features
    '''
    def __init__(self, img_size, nfilter=64, nfilter_max=256, cond=False):
        super().__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        size = img_size

        # Submodules
        nlayers = int(log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)
        '''

        blocks = [
            ResnetBlock(nf, nf)
        ]
        '''
        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks += [
                ResnetBlock(nf0, nf1),
                nn.AvgPool2d(3, stride=2, padding=1),
            ]

        nf0 = min(nf * 2 ** nlayers, nf_max)
        nf1 = min(nf * 2 ** (nlayers + 1), nf_max)
        blocks += [ResnetBlock(nf0, nf1)]

        self.conv_img = spectral_norm(nn.Conv2d(3, 1*nf, 5, padding=2))
        self.resnet = nn.Sequential(*blocks)
        self.fc = spectral_norm(nn.Linear(self.nf0, 1, bias=False))
        self.actvn = nn.ReLU()

    def forward(self, x, cond_data=None, **kwargs):
        batch_size = x.size(0)

        out = self.conv_img(x)
        out = self.actvn(self.resnet(out))
        out = nn.functional.avg_pool2d(out, 4)*16
        out = out.view(batch_size, self.nf0)
        out = self.fc(out)
        return out


class Identifier(nn.Module):
    def __init__(self, img_size, z_dim=128, z_dim_bg = 128, size_h = 32, size_max = 1024,  batch_size = 32):
        super().__init__()
        self.z_dim = z_dim
        self.z_dim_bg = z_dim_bg
        self.batch_size = batch_size
        self.out_dim = 512
        self.encoder = ImgEncoderwoSN(img_size, self.out_dim , size_h = size_h, size_max = size_max)

        self.out_shape_obj = (nn.Linear(self.out_dim , z_dim))
        self.out_app_obj = (nn.Linear(self.out_dim , z_dim))
        self.out_shape_bg = (nn.Linear(self.out_dim , z_dim_bg))
        self.out_app_bg = (nn.Linear(self.out_dim , z_dim_bg))


    def forward(self, x, batch_size):
        a = F.relu

        z_hidden = self.encoder(x)

        z_shape_obj = a(self.out_shape_obj(z_hidden)).reshape(batch_size,1,self.z_dim)
        z_app_obj =  a(self.out_app_obj(z_hidden)).reshape(batch_size,1,self.z_dim)
        z_shape_bg = a(self.out_shape_bg(z_hidden))
        z_app_bg = a(self.out_app_bg(z_hidden))

        
        latents = (z_shape_obj, z_app_obj,z_shape_bg,z_app_bg)
        return latents
