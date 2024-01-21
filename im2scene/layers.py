import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from kornia.filters import filter2D
import torch


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = spectral_norm(nn.Linear(size_in, size_h))
        self.fc_1 = spectral_norm(nn.Linear(size_h, size_out))
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = spectral_norm(nn.Linear(size_in, size_out, bias=False))
        # Initialization

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = spectral_norm(nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1, bias=is_bias))
        self.conv_1 = spectral_norm(nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias))

        if self.learned_shortcut:
            self.conv_s = spectral_norm(nn.Conv2d(
                self.fin, self.fout, 1, stride=1, padding=0, bias=False))

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

class ResnetBlockAdaIN(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = spectral_norm(nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1, bias=is_bias))
        self.conv_1 = spectral_norm(nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias))

        if self.learned_shortcut:
            self.conv_s = spectral_norm(nn.Conv2d(
                self.fin, self.fout, 1, stride=1, padding=0, bias=False))
        
        self.adain = AdaIN()

    def actvn(self, x):
        out = F.relu(x)
        return out

    def forward(self, x, y):
        x_s = self._shortcut(x)
        dx = self.adain(self.conv_0(self.actvn(x)), y)
        dx = self.adain(self.conv_1(self.actvn(dx)), y)
        out = x_s + dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s

class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2D(x, f, normalized=True)
