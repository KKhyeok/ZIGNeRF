import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from kornia.filters import filter2D
import torch
import numpy as np

class ResnetBlockDown(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = True
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = spectral_norm(nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1, bias=is_bias))
        self.conv_1 = spectral_norm(nn.Conv2d(self.fhidden, self.fout, 3, stride=2, padding=1, bias=is_bias))

        if self.learned_shortcut:
            self.conv_s = spectral_norm(nn.Conv2d(self.fin, self.fout, 1, stride=2, padding=0, bias=False))

    def actvn(self, x):
        out = F.relu(x)
        return out

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(self.actvn(x))
        dx = self.conv_1(self.actvn(dx))
        out = x_s + dx
        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s

class ResnetBlockDownwoSN(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = True
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1, bias=is_bias)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=2, padding=1, bias=is_bias)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=2, padding=0, bias=False)

    def actvn(self, x):
        out = F.relu(x)
        return out

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(self.actvn(x))
        dx = self.conv_1(self.actvn(dx))
        out = x_s + dx
        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s

class ImgEncoder(nn.Module):
    def __init__(self, size_in, size_out, size_h=32, size_max=256):
        super().__init__()

        self.num_layer = int(np.log2(size_in/4))
        self.size_in = size_in
        self.size_in = size_out
        self.size_h = size_h

        self.conv_0 = spectral_norm(nn.Conv2d(3, size_h, 7, stride=1, padding=3))
        self.conv_layers = nn.ModuleList([ResnetBlockDown(min(size_max, size_h*2**i), min(size_max, size_h*2**(i+1))) for i in range(self.num_layer)])
        self.out = spectral_norm(nn.Linear(min(size_max, size_h*2**self.num_layer), size_out))

    def forward(self, x):
        h = self.conv_0(x)
        for idx, layer in enumerate(self.conv_layers):
            h = layer(h)
        h = self.out(h.mean([2, 3]))
        return h

class ImgEncoderwoSN(nn.Module):
    def __init__(self, size_in, size_out, size_h=16, size_max=512):
        super().__init__()

        self.num_layer = int(np.log2(size_in/4))
        self.size_in = size_in
        self.size_in = size_out
        self.size_h = size_h

        self.conv_0 = nn.Conv2d(3, size_h, 7, stride=1, padding=3)
        self.conv_layers = nn.ModuleList([ResnetBlockDownwoSN(min(size_max, size_h*2**i), min(size_max, size_h*2**(i+1))) for i in range(self.num_layer)])
        self.out = nn.Linear(min(size_max, size_h*2**self.num_layer), size_out)

    def forward(self, x):
        h = self.conv_0(x)
        for idx, layer in enumerate(self.conv_layers):
            h = layer(h)
        h = self.out(h.mean([2, 3]))
        return h