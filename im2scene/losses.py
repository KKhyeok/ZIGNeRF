import torch
import torch.nn as nn
import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16, vgg19


EPS = 1e-7


class VGGPerceptualLoss(nn.Module):
    """https://github.com/elliottwu/unsup3d/blob/master/unsup3d/networks.py"""
    def __init__(self, requires_grad=False):
        super(VGGPerceptualLoss, self).__init__()
        mean_rgb = torch.FloatTensor([0.485, 0.456, 0.406])
        std_rgb = torch.FloatTensor([0.229, 0.224, 0.225])
        self.register_buffer('mean_rgb', mean_rgb)
        self.register_buffer('std_rgb', std_rgb)

        vgg_pretrained_features = torchvision.models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def normalize(self, x):
        out = x/2 + 0.5
        out = (out - self.mean_rgb.view(1,3,1,1)) / self.std_rgb.view(1,3,1,1)
        return out

    def __call__(self, im1, im2, mask=None, conf_sigma=None):
        im = torch.cat([im1,im2], 0)
        im = self.normalize(im)  # normalize input

        ## compute features
        feats = []
        f = self.slice1(im)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice2(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice3(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice4(f)
        feats += [torch.chunk(f, 2, dim=0)]

        losses = []
        for f1, f2 in feats[2:3]:  # use relu3_3 features only
            loss = (f1-f2)**2
            if conf_sigma is not None:
                loss = loss / (2*conf_sigma**2 +EPS) + (conf_sigma +EPS).log()
            if mask is not None:
                b, c, h, w = loss.shape
                _, _, hm, wm = mask.shape
                sh, sw = hm//h, wm//w
                mask0 = nn.functional.avg_pool2d(mask, kernel_size=(sh,sw), stride=(sh,sw)).expand_as(loss)
                loss = (loss * mask0).sum() / mask0.sum()
            else:
                loss = loss.mean()
            losses += [loss]
        return sum(losses)


def kld_loss(mu, var):
    log_var = torch.log(var)
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 0), dim = 0)
    return kld_loss



####



class CombinationLoss(nn.Module):
    def __init__(self, device):
        super(CombinationLoss, self).__init__()
        self.l1_lambda = 0.
        self.l2_lambda = 1.
        self.vgg_lambda = 10.
        self.vgg = VGGLoss(16, vgg_loss_type = 'L1', device = device).to(device)
        self.mse = nn.MSELoss().to(device)
        self.l1 = nn.L1Loss().to(device)


    def forward(self, x, gt):
        l1 = self.l1(x, gt)
        l2 = self.mse(x, gt)
        vgg = self.vgg(x, gt)
        # print(l1.item(), l2.item(), vgg.item())
        loss_l1 =self.l1_lambda * l1 
        loss_l2 =self.l2_lambda * l2
        loss_vgg = self.vgg_lambda * vgg
        loss_total = loss_l1 + loss_l2 + loss_vgg 
        return loss_total, loss_l1, loss_l2, loss_vgg


class VGGLoss(nn.Module):
    """
   (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
   (1): ReLU(inplace)
   (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
   (3): ReLU(inplace)
   (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
   (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
   (6): ReLU(inplace)
   (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
   (8): ReLU(inplace)
   (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
   (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
   (11): ReLU(inplace)
   (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
   (13): ReLU(inplace)
   (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
   (15): ReLU(inplace)
   (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
   (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
   (18): ReLU(inplace)
   (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
   (20): ReLU(inplace)
   (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
   (22): ReLU(inplace)
   (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
   (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
   (25): ReLU(inplace)
   (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
   (27): ReLU(inplace)
   (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
   (29): ReLU(inplace)
   (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    """
    def __init__(self, vgg_layer, vgg_loss_type = 'L1', device = None):
        super(VGGLoss, self).__init__()
        self.vgg = nn.Sequential(*list(vgg16(pretrained=True).children())[0][:vgg_layer]).to(device)
        if vgg_loss_type == 'L2':
            self.loss = F.mse_loss
        elif vgg_loss_type == 'L1':
            self.loss = F.l1_loss
        self.pre_processing_mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
        self.pre_processing_std = torch.Tensor([0.229, 0.224, 0.225]).to(device)
        self.resize = 128


    def forward(self, x, gt):
        """
        :param x: [-1.0, 1.0]
        :param gt: [-1.0, 1.0]
        :return:
        """
        x = (x * 0.5 + 0.5).sub_(self.pre_processing_mean[:, None, None]).div_(self.pre_processing_std[:, None, None])
        gt = (gt * 0.5 + 0.5).sub_(self.pre_processing_mean[:, None, None]).div_(self.pre_processing_std[:, None, None])
        x_features = self.vgg(F.interpolate(x, size=self.resize, mode='nearest'))
        gt_features = self.vgg(F.interpolate(gt, size=self.resize, mode='nearest'))
        return self.loss(x_features, gt_features, reduction='mean') * 0.001

