import torch.nn as nn
import torch
from math import log2
from im2scene.layers import Blur
import torch.nn.utils.spectral_norm as spectral_norm
from im2scene.layers import ResnetBlock, ResnetBlockAdaIN

class AdaIN(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x, y):
		eps = 1e-5	
		mean_x = torch.mean(x, dim=[2,3])
		mean_y = torch.mean(y, dim=[2,3])

		std_x = torch.std(x, dim=[2,3])
		std_y = torch.std(y, dim=[2,3])

		mean_x = mean_x.unsqueeze(-1).unsqueeze(-1)
		mean_y = mean_y.unsqueeze(-1).unsqueeze(-1)

		std_x = std_x.unsqueeze(-1).unsqueeze(-1) + eps
		std_y = std_y.unsqueeze(-1).unsqueeze(-1) + eps

		out = (x - mean_x)/ std_x * std_y + mean_y

		return out

class NeuralRenderer(nn.Module):
    ''' Neural renderer class

    Args:
        n_feat (int): number of features
        input_dim (int): input dimension; if not equal to n_feat,
            it is projected to n_feat with a 1x1 convolution
        out_dim (int): output dimension
        final_actvn (bool): whether to apply a final activation (sigmoid)
        min_feat (int): minimum features
        img_size (int): output image size
        use_rgb_skip (bool): whether to use RGB skip connections
        upsample_feat (str): upsampling type for feature upsampling
        upsample_rgb (str): upsampling type for rgb upsampling
        use_norm (bool): whether to use normalization
    '''

    def __init__(
            self, n_feat=256, input_dim=128, out_dim=3, final_actvn=True,
            min_feat=128, img_size=64, use_rgb_skip=True,
            upsample_feat="nn", upsample_rgb="bilinear", use_norm=False,
            **kwargs):
        super().__init__()
        self.final_actvn = final_actvn
        self.input_dim = input_dim
        self.use_rgb_skip = use_rgb_skip
        self.use_norm = use_norm
        n_blocks = int(log2(img_size) - 4)

        assert(upsample_feat in ("nn", "bilinear"))
        if upsample_feat == "nn":
            self.upsample_2 = nn.Upsample(scale_factor=2.)
        elif upsample_feat == "bilinear":
            self.upsample_2 = nn.Sequential(nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False), Blur())

        assert(upsample_rgb in ("nn", "bilinear"))
        if upsample_rgb == "nn":
            self.upsample_rgb = nn.Upsample(scale_factor=2.)
        elif upsample_rgb == "bilinear":
            self.upsample_rgb = nn.Sequential(nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False), Blur())

        if n_feat == input_dim:
            self.conv_in = lambda x: x
        else:
            self.conv_in = spectral_norm(nn.Conv2d(input_dim, n_feat, 1, 1, 0))

        if use_norm:
            self.conv_layers = nn.ModuleList(
                [ResnetBlockAdaIN(n_feat, n_feat // 2)] +
                [ResnetBlockAdaIN(max(n_feat // (2 ** (i + 1)), min_feat),
                           max(n_feat // (2 ** (i + 2)), min_feat))
                    for i in range(0, n_blocks - 1)]
            )
        else:
            self.conv_layers = nn.ModuleList(
                [ResnetBlock(n_feat, n_feat // 2)] +
                [ResnetBlock(max(n_feat // (2 ** (i + 1)), min_feat),
                           max(n_feat // (2 ** (i + 2)), min_feat))
                    for i in range(0, n_blocks - 1)]
            )
       
        if use_rgb_skip:
            self.conv_rgb = nn.ModuleList(
                [ResnetBlock(input_dim, out_dim)] +
                [ResnetBlock(max(n_feat // (2 ** (i + 1)), min_feat),
                           out_dim) for i in range(0, n_blocks)]
            )
        else:
            self.conv_rgb = spectral_norm(nn.Conv2d(
                max(n_feat // (2 ** (n_blocks)), min_feat), 3, 1, 1))
       
        if use_norm:
            self.norms_w = nn.ModuleList([
                spectral_norm(nn.Conv2d(self.input_dim, max(n_feat // (2 ** (i + 1)), min_feat), 1, stride=1, padding=0, bias=False))
                for i in range(n_blocks)
            ])

        self.actvn = nn.ReLU()

    def forward(self, x):

        net = self.conv_in(x)

        if self.use_rgb_skip:
            rgb = self.upsample_rgb(self.conv_rgb[0](x))

        for idx, layer in enumerate(self.conv_layers):
            if self.use_norm:
                hid = layer(self.upsample_2(net), self.norms_w[idx](x))
            else:
                hid = layer(self.upsample_2(net))


            net = hid

            if self.use_rgb_skip:
                rgb = rgb + self.conv_rgb[idx + 1](net)
                if idx < len(self.conv_layers) - 1:
                    rgb = self.upsample_rgb(rgb)

        if not self.use_rgb_skip:
            net = self.actvn(net)
            rgb = self.conv_rgb(net)

        if self.final_actvn:
            rgb = torch.sigmoid(rgb)
        return rgb
