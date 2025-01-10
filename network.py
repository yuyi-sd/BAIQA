import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from torch import Tensor
from typing import Tuple
import torch.fft

import torchvision.transforms as T
from io import BytesIO
from torchvision import transforms
import random
from PIL import Image

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class U_Net(nn.Module):
    def __init__(self, epsilon = 0.03, img_ch=3,output_ch=1, multiplier = 1):
        super(U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv2 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        self.Conv3 = conv_block(ch_in=int(128 * multiplier),ch_out=int(256 * multiplier))
        self.Conv4 = conv_block(ch_in=int(256 * multiplier),ch_out=int(512 * multiplier))
        self.Conv5 = conv_block(ch_in=int(512 * multiplier),ch_out=int(1024 * multiplier))

        self.Up5 = up_conv(ch_in=int(1024 * multiplier),ch_out=int(512 * multiplier))
        self.Up_conv5 = conv_block(ch_in=int(1024 * multiplier), ch_out=int(512 * multiplier))

        self.Up4 = up_conv(ch_in=int(512 * multiplier),ch_out=int(256 * multiplier))
        self.Up_conv4 = conv_block(ch_in=int(512 * multiplier), ch_out=int(256 * multiplier))
        
        self.Up3 = up_conv(ch_in=int(256 * multiplier),ch_out=int(128 * multiplier))
        self.Up_conv3 = conv_block(ch_in=int(256 * multiplier), ch_out=int(128 * multiplier))
        
        self.Up2 = up_conv(ch_in=int(128 * multiplier),ch_out=int(64 * multiplier))
        self.Up_conv2 = conv_block(ch_in=int(128 * multiplier), ch_out=int(64 * multiplier))

        self.Conv_1x1 = nn.Conv2d(int(64 * multiplier),output_ch,kernel_size=1,stride=1,padding=0)
        self.epsilon = epsilon

    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        d = F.tanh(d1) * self.epsilon
        return d

import numpy as np
import torch
import torch.nn as nn


def dct1(x):
    """
    Discrete Cosine Transform, Type I
    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])

    return torch.rfft(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1), 1)[:, :, 0].view(*x_shape)


def idct1(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I
    Our definition if idct1 is such that idct1(dct1(x)) == x
    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

#     Vc = torch.rfft(v, 1, onesided=False)
    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))  # add this line

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

#     v = torch.irfft(V, 1, onesided=False)
    v= torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)   # add this line
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_2d(dct_2d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)

def blockify(im: Tensor, size: int) -> Tensor:
    r"""
    Breaks an image into non-overlapping blocks of equal size.

    Parameters
    ----------
    im : Tensor
        The image to break into blocks, must be in :math:`(N, C, H, W)` format.
    size : Tuple[int, int]
        The size of the blocks in :math:`(H, W)` format.

    Returns
    -------
    A tensor containing the non-overlappng blocks in :math:`(N, C, L, H, W)` format where :math:`L` is the
    number of non-overlapping blocks in the image channel indexed by :math:`(N, C)` and :math:`(H, W)` matches
    the block size.

    Note
    ----
    If the image does not split evenly into blocks of the given size, the result will have some overlap. It
    is the callers responsibility to pad the input to a multiple of the block size, no error will be thrown
    in this case.
    """
    bs = im.shape[0]
    ch = im.shape[1]
    h = im.shape[2]
    w = im.shape[3]

    im = im.reshape(bs * ch, 1, h, w)
    im = torch.nn.functional.unfold(im, kernel_size=(size, size), stride=(size, size))
    im = im.transpose(1, 2)
    im = im.reshape(bs, ch, -1, size, size)

    return im



def deblockify(blocks: Tensor, size: Tuple[int, int]) -> Tensor:
    r"""
    Reconstructs an image given non-overlapping blocks of equal size.

    Args:
        blocks (Tensor): The non-overlapping blocks in :math:`(N, C, L, H, W)` format.
        size: (Tuple[int, int]): The dimensions of the original image (e.g. the desired output)
            in :math:`(H, W)` format.

    Returns:
        The image in :math:`(N, C, H, W)` format.

    Note:
        If the blocks have some overlap, or if the output size cannot be constructed from the given number of non-overlapping
        blocks, this function will raise an exception unlike :py:func:`blockify`.

    """
    bs = blocks.shape[0]
    ch = blocks.shape[1]
    block_size = blocks.shape[3]

    blocks = blocks.reshape(bs * ch, -1, int(block_size ** 2))
    blocks = blocks.transpose(1, 2)
    blocks = torch.nn.functional.fold(blocks, output_size=size, kernel_size=(block_size, block_size), stride=(block_size, block_size))
    blocks = blocks.reshape(bs, ch, size[0], size[1])

    return blocks

def to_ycbcr(x: Tensor, data_range: float = 255) -> Tensor:
    r"""
    Converts a Tensor from RGB color space to YCbCr color space

    Parameters
    ----------
    x : Tensor
        The input Tensor holding an RGB image in :math:`(\ldots, C, H ,W)` format (where :math:`\ldots` indicates an arbitrary number of dimensions).
    data_range : float
        The range of the input/output data. i.e., 255 indicates pixels in [0, 255], 1.0 indicates pixels in [0, 1]. Only 1.0 and 255 are supported.

    Returns
    -------
    Tensor
        The YCbCr result of the same shape as the input and with the same data range.

    Note
    -----
    This function implements the "full range" conversion used by JPEG, e.g. it does **not** implement the ITU-R BT.601 standard which 
    many libraries (excluding PIL) use as the default definition of YCbCr. This conversion (for [0, 255]) is given by:

    .. math::
        \begin{aligned}
        Y&=&0&+(0.299&\cdot R)&+(0.587&\cdot G)&+(0.114&\cdot B) \\
        C_{B}&=&128&-(0.168736&\cdot R)&-(0.331264&\cdot G)&+(0.5&\cdot B) \\
        C_{R}&=&128&+(0.5&\cdot R)&-(0.418688&\cdot G)&-(0.081312&\cdot B)
        \end{aligned}
    
    """
    assert data_range in [1.0, 255]

    # fmt: off
    ycbcr_from_rgb = torch.tensor([
        0.29900, 0.58700, 0.11400,
        -0.168735892, -0.331264108, 0.50000,
        0.50000, -0.418687589, -0.081312411
    ]).view(3, 3).transpose(0, 1)
    # fmt: on

    if data_range == 255:
        b = torch.tensor([0, 128, 128]).view(3, 1, 1)
    else:
        b = torch.tensor([0, 0.5, 0.5]).view(3, 1, 1)

    if x.is_cuda:
        ycbcr_from_rgb = ycbcr_from_rgb.cuda()
        b = b.cuda()

    x = torch.einsum("cv,...cxy->...vxy", [ycbcr_from_rgb, x])
    x += b

    return x.contiguous()



def to_rgb(x: Tensor, data_range: float = 255) -> Tensor:
    r"""
    Converts a Tensor from YCbCr color space to RGB color space

    Parameters
    ----------
    x : Tensor
        The input Tensor holding a YCbCr image in :math:`(\ldots, C, H ,W)` format (where :math:`\ldots` indicates an arbitrary number of dimensions).
    data_range : float
        The range of the input/output data. i.e., 255 indicates pixels in [0, 255], 1.0 indicates pixels in [0, 1]. Only 1.0 and 255 are supported.

    Returns
    -------
    Tensor
        The RGB result of the same shape as the input and with the same data range.

    Note
    -----
    This function expects the input to be "full range" conversion used by JPEG, e.g. it does **not** implement the ITU-R BT.601 standard which 
    many libraries (excluding PIL) use as the default definition of YCbCr. If the input came from this library or from PIL it should be fine.
    The conversion (for [0, 255]) is given by:

    .. math::
        \begin{aligned}
        R&=&Y&&&+1.402&\cdot (C_{R}-128) \\
        G&=&Y&-0.344136&\cdot (C_{B}-128)&-0.714136&\cdot (C_{R}-128 ) \\
        B&=&Y&+1.772&\cdot (C_{B}-128)&
        \end{aligned}
    
    """
    assert data_range in [1.0, 255]

    # fmt: off
    rgb_from_ycbcr = torch.tensor([
        1, 0, 1.40200,
        1, -0.344136286, -0.714136286,
        1, 1.77200, 0
    ]).view(3, 3).transpose(0, 1)
    # fmt: on

    if data_range == 255:
        b = torch.tensor([-179.456, 135.458816, -226.816]).view(3, 1, 1)
    else:
        b = torch.tensor([-0.70374902, 0.531211043, -0.88947451]).view(3, 1, 1)

    if x.is_cuda:
        rgb_from_ycbcr = rgb_from_ycbcr.cuda()
        b = b.cuda()

    x = torch.einsum("cv,...cxy->...vxy", [rgb_from_ycbcr, x])
    x += b

    return x.contiguous()

class Trigger_Net(nn.Module):
    def __init__(self, img_ch=3, multiplier = 1):
        super(Trigger_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv2 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        self.Conv_1x1_1 = nn.Conv2d(int(128 * multiplier),img_ch,kernel_size=1,stride=1,padding=0)

        self.Conv3 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv4 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        self.Conv5 = conv_block(ch_in=int(128 * multiplier),ch_out=int(256 * multiplier))
        self.Conv6 = conv_block(ch_in=int(256 * multiplier),ch_out=int(512 * multiplier))
        self.Conv_1x1_2 = nn.Conv2d(int(512 * multiplier),img_ch,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = (x * 255).round()

        x_ycbcr = to_ycbcr(x, data_range = 255)
        H, W = x.shape[2], x.shape[3]
        K1 = H//16
        K2 = W//16
        x_blocks = blockify(x,16).transpose(1,2)
        N = x_blocks.shape[0]
        L = x_blocks.shape[1]
        x_blocks = x_blocks.contiguous().view(N * L, x_blocks.shape[2], 16, 16)
        x_blocks = self.Conv1(x_blocks)
        x_blocks = self.Maxpool(x_blocks)
        x_blocks = x_blocks.view(N, L, x_blocks.shape[1], x_blocks.shape[2], x_blocks.shape[3]).transpose(1,2) # N * C * L * 4 * 4
        feature_patch, _ = torch.max(x_blocks, dim = 2)
        feature_patch = self.Conv2(feature_patch)
        feature_patch = self.Conv_1x1_1(feature_patch)
        trigger_patch = F.pad(feature_patch,(4,4,4,4))
        trigger_patch = trigger_patch.repeat(1,1,K1,K2)

        trigger_weight = self.Conv3(x)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv4(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv5(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv6(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv_1x1_2(trigger_weight)
        trigger_weight = trigger_weight.repeat_interleave(16,2).repeat_interleave(16,3)

        x_ycbcr_blocks = blockify(x_ycbcr,16) - 128
        dct_blocks = dct_2d(x_ycbcr_blocks)
        dct = deblockify(dct_blocks,(H,W))
        dct_poisoned = dct + trigger_patch * trigger_weight
        dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)

        x_poisoned = x_poisoned.div(255)
       
        return x_poisoned

class Trigger_Net_v3(nn.Module):
    def __init__(self, img_ch=3, multiplier = 1, trigger_channel = 16):
        super(Trigger_Net_v3,self).__init__()
        
        self.img_ch = img_ch
        self.trigger_channel = trigger_channel
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        # self.Maxpool2 = nn.MaxPool2d(kernel_size=4,stride=4)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv2 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        # self.Conv3 = conv_block(ch_in=int(128 * multiplier),ch_out=int(256 * multiplier))
        # self.Conv4 = conv_block(ch_in=int(256 * multiplier),ch_out=int(512 * multiplier))
        self.Conv_1x1_1 = nn.Conv2d(int(16 * 128 * multiplier), img_ch * trigger_channel, kernel_size=1,stride=1,padding=0)

        self.Conv5 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv6 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        self.Conv7 = conv_block(ch_in=int(128 * multiplier),ch_out=int(256 * multiplier))
        self.Conv8 = conv_block(ch_in=int(256 * multiplier),ch_out=int(512 * multiplier))
        self.Conv_1x1_2 = nn.Conv2d(int(512 * multiplier),img_ch,kernel_size=1,stride=1,padding=0)

        self.zigzag_indices = torch.tensor([
             0,  1,  5,  6, 14, 15, 27, 28, 44, 45, 65, 66, 90, 91,119,120,
             2,  4,  7, 13, 16, 26, 29, 43, 46, 64, 67, 89, 92,118,121,150,
             3,  8, 12, 17, 25, 30, 42, 47, 63, 68, 88, 93,117,122,149,151,
             9, 11, 18, 24, 31, 41, 48, 62, 69, 87, 94,116,123,148,152,177,
            10, 19, 23, 32, 40, 49, 61, 70, 86, 95,115,124,147,153,176,178,
            20, 22, 33, 39, 50, 60, 71, 85, 96,114,125,146,154,175,179,200,
            21, 34, 38, 51, 59, 72, 84, 97,113,126,145,155,174,180,199,201,
            35, 37, 52, 58, 73, 83, 98,112,127,144,156,173,181,198,202,219,
            36, 53, 57, 74, 82, 99,111,128,143,157,172,182,197,203,218,220,
            54, 56, 75, 81,100,110,129,142,158,171,183,196,204,217,221,234,
            55, 76, 80,101,109,130,141,159,170,184,195,205,216,222,233,235,
            77, 79,102,108,131,140,160,169,185,194,206,215,223,232,236,245,
            78,103,107,132,139,161,168,186,193,207,214,224,231,237,244,246,
            104,106,133,138,162,167,187,192,208,213,225,230,238,243,247,252,
            105,134,137,163,166,188,191,209,212,226,229,239,242,248,251,253,
            135,136,164,165,189,190,210,211,227,228,240,241,249,250,254,255
        ]).long() 

    def forward(self,x):
        x = (x * 255).round()

        x_ycbcr = to_ycbcr(x, data_range = 255)
        H, W = x.shape[2], x.shape[3]
        K1 = H//16
        K2 = W//16

        x_blocks = blockify(x,16).transpose(1,2)
        N = x_blocks.shape[0]
        L = x_blocks.shape[1]
        x_blocks = x_blocks.contiguous().view(N * L, x_blocks.shape[2], 16, 16)
        x_blocks = self.Conv1(x_blocks)
        x_blocks = self.Maxpool(x_blocks)
        x_blocks = self.Conv2(x_blocks)
        x_blocks = self.Maxpool(x_blocks) # NL * C * 4 * 4
        x_blocks = x_blocks.view(N, L, x_blocks.shape[1], x_blocks.shape[2], x_blocks.shape[3]).transpose(1,2) # N * C * L * 4 * 4
        feature_patch, _ = torch.max(x_blocks, dim = 2)
        feature_patch = feature_patch.view(feature_patch.shape[0], 16 * feature_patch.shape[1], 1, 1)
        feature_patch = self.Conv_1x1_1(feature_patch)
        feature_patch = feature_patch.view(feature_patch.shape[0], self.img_ch, self.trigger_channel)

        trigger_patch = torch.zeros_like(x[:,:,:16,:16])
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 256)
        trigger_patch[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]] = feature_patch
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 16, 16).repeat(1,1,K1,K2)

        trigger_weight = self.Conv5(x)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv6(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv7(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv8(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv_1x1_2(trigger_weight)
        trigger_weight = trigger_weight.repeat_interleave(16,2).repeat_interleave(16,3)

        x_ycbcr_blocks = blockify(x_ycbcr,16) - 128
        dct_blocks = dct_2d(x_ycbcr_blocks)
        dct = deblockify(dct_blocks,(H,W))
        dct_poisoned = dct + trigger_patch * trigger_weight
        dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)

        x_poisoned = x_poisoned.div(255)
       
        return x_poisoned

class Trigger_Net_v4(nn.Module):
    def __init__(self, img_ch=3, multiplier = 1, trigger_channel = 64):
        super(Trigger_Net_v4,self).__init__()
        
        self.img_ch = img_ch
        self.trigger_channel = trigger_channel
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        # self.Maxpool2 = nn.MaxPool2d(kernel_size=4,stride=4)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv2 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        # self.Conv3 = conv_block(ch_in=int(128 * multiplier),ch_out=int(256 * multiplier))
        # self.Conv4 = conv_block(ch_in=int(256 * multiplier),ch_out=int(512 * multiplier))
        self.Conv_1x1_1 = nn.Conv2d(int(16 * 128 * multiplier), img_ch * trigger_channel, kernel_size=1,stride=1,padding=0)
        self.Conv_1x1_2 = nn.Conv2d(int(16 * 128 * multiplier), img_ch * trigger_channel, kernel_size=1,stride=1,padding=0)

        self.Conv5 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv6 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        self.Conv7 = conv_block(ch_in=int(128 * multiplier),ch_out=int(256 * multiplier))
        self.Conv8 = conv_block(ch_in=int(256 * multiplier),ch_out=int(512 * multiplier))
        self.Conv_1x1_3 = nn.Conv2d(int(512 * multiplier),img_ch,kernel_size=1,stride=1,padding=0)

        self.zigzag_indices = torch.tensor([
             0,  1,  5,  6, 14, 15, 27, 28, 44, 45, 65, 66, 90, 91,119,120,
             2,  4,  7, 13, 16, 26, 29, 43, 46, 64, 67, 89, 92,118,121,150,
             3,  8, 12, 17, 25, 30, 42, 47, 63, 68, 88, 93,117,122,149,151,
             9, 11, 18, 24, 31, 41, 48, 62, 69, 87, 94,116,123,148,152,177,
            10, 19, 23, 32, 40, 49, 61, 70, 86, 95,115,124,147,153,176,178,
            20, 22, 33, 39, 50, 60, 71, 85, 96,114,125,146,154,175,179,200,
            21, 34, 38, 51, 59, 72, 84, 97,113,126,145,155,174,180,199,201,
            35, 37, 52, 58, 73, 83, 98,112,127,144,156,173,181,198,202,219,
            36, 53, 57, 74, 82, 99,111,128,143,157,172,182,197,203,218,220,
            54, 56, 75, 81,100,110,129,142,158,171,183,196,204,217,221,234,
            55, 76, 80,101,109,130,141,159,170,184,195,205,216,222,233,235,
            77, 79,102,108,131,140,160,169,185,194,206,215,223,232,236,245,
            78,103,107,132,139,161,168,186,193,207,214,224,231,237,244,246,
            104,106,133,138,162,167,187,192,208,213,225,230,238,243,247,252,
            105,134,137,163,166,188,191,209,212,226,229,239,242,248,251,253,
            135,136,164,165,189,190,210,211,227,228,240,241,249,250,254,255
        ]).long() 

    def forward(self,x):
        # x, pads = pad_to(x, 16)
        x = (x * 255).round()

        x_ycbcr = to_ycbcr(x, data_range = 255)
        H, W = x.shape[2], x.shape[3]
        K1 = H//16
        K2 = W//16

        x_blocks = blockify(x,16).transpose(1,2)
        N = x_blocks.shape[0]
        L = x_blocks.shape[1]
        x_blocks = x_blocks.contiguous().view(N * L, x_blocks.shape[2], 16, 16)
        x_blocks = self.Conv1(x_blocks)
        x_blocks = self.Maxpool(x_blocks)
        x_blocks = self.Conv2(x_blocks)
        x_blocks = self.Maxpool(x_blocks) # NL * C * 4 * 4
        x_blocks = x_blocks.view(N, L, x_blocks.shape[1], x_blocks.shape[2], x_blocks.shape[3]).transpose(1,2) # N * C * L * 4 * 4
        feature_patch, _ = torch.max(x_blocks, dim = 2)
        feature_patch = feature_patch.view(feature_patch.shape[0], 16 * feature_patch.shape[1], 1, 1) # N * 16C * 1 * 1

        feature_patch_importance = self.Conv_1x1_2(feature_patch)
        feature_patch_importance = feature_patch_importance.view(feature_patch_importance.shape[0], self.img_ch, self.trigger_channel)
        topk_index = torch.topk(feature_patch_importance, self.trigger_channel//4 * 3, dim = 2, largest = False)[1]
        feature_patch = self.Conv_1x1_1(feature_patch)
        feature_patch = feature_patch.view(feature_patch.shape[0], self.img_ch, self.trigger_channel)
        feature_patch.scatter_(2, topk_index, 0)


        trigger_patch = torch.zeros_like(x[:,:,:16,:16], dtype = feature_patch.dtype)
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 256)
        trigger_patch[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]] = feature_patch
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 16, 16).repeat(1,1,K1,K2)

        trigger_weight = self.Conv5(x)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv6(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv7(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv8(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv_1x1_3(trigger_weight)
        trigger_weight = trigger_weight.repeat_interleave(16,2).repeat_interleave(16,3)

        x_ycbcr_blocks = blockify(x_ycbcr,16) - 128
        dct_blocks = dct_2d(x_ycbcr_blocks)
        dct = deblockify(dct_blocks,(H,W))
        dct_poisoned = dct + trigger_patch * trigger_weight
        dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)

        x_poisoned = x_poisoned.div(255)
        
        # x_poisoned = unpad(x_poisoned, pads)
        return x_poisoned
        # return x_poisoned, trigger_weight

class Trigger_Net_v5(nn.Module):
    def __init__(self, img_ch=3, multiplier = 1, trigger_channel = 64):
        super(Trigger_Net_v5,self).__init__()
        
        self.img_ch = img_ch
        self.trigger_channel = trigger_channel
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        # self.Maxpool2 = nn.MaxPool2d(kernel_size=4,stride=4)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv2 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        self.Conv3 = conv_block(ch_in=int(128 * multiplier),ch_out=int(128 * multiplier))
        # self.Conv4 = conv_block(ch_in=int(256 * multiplier),ch_out=int(512 * multiplier))
        self.Conv_1x1_0 = nn.Conv2d(int(16 * 128 * multiplier), int(4 * 128 * multiplier), kernel_size=1,stride=1,padding=0)
        self.Conv_1x1_1 = nn.Conv2d(int(4 * 128 * multiplier), img_ch * trigger_channel, kernel_size=1,stride=1,padding=0)
        self.Conv_1x1_2 = nn.Conv2d(int(4 * 128 * multiplier), img_ch * trigger_channel, kernel_size=1,stride=1,padding=0)

        self.Conv5 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv6 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        self.Conv7 = conv_block(ch_in=int(128 * multiplier),ch_out=int(256 * multiplier))
        self.Conv8 = conv_block(ch_in=int(256 * multiplier),ch_out=int(512 * multiplier))
        self.Conv_1x1_3 = nn.Conv2d(int(512 * multiplier),img_ch,kernel_size=1,stride=1,padding=0)

        self.zigzag_indices = torch.tensor([
             0,  1,  5,  6, 14, 15, 27, 28, 44, 45, 65, 66, 90, 91,119,120,
             2,  4,  7, 13, 16, 26, 29, 43, 46, 64, 67, 89, 92,118,121,150,
             3,  8, 12, 17, 25, 30, 42, 47, 63, 68, 88, 93,117,122,149,151,
             9, 11, 18, 24, 31, 41, 48, 62, 69, 87, 94,116,123,148,152,177,
            10, 19, 23, 32, 40, 49, 61, 70, 86, 95,115,124,147,153,176,178,
            20, 22, 33, 39, 50, 60, 71, 85, 96,114,125,146,154,175,179,200,
            21, 34, 38, 51, 59, 72, 84, 97,113,126,145,155,174,180,199,201,
            35, 37, 52, 58, 73, 83, 98,112,127,144,156,173,181,198,202,219,
            36, 53, 57, 74, 82, 99,111,128,143,157,172,182,197,203,218,220,
            54, 56, 75, 81,100,110,129,142,158,171,183,196,204,217,221,234,
            55, 76, 80,101,109,130,141,159,170,184,195,205,216,222,233,235,
            77, 79,102,108,131,140,160,169,185,194,206,215,223,232,236,245,
            78,103,107,132,139,161,168,186,193,207,214,224,231,237,244,246,
            104,106,133,138,162,167,187,192,208,213,225,230,238,243,247,252,
            105,134,137,163,166,188,191,209,212,226,229,239,242,248,251,253,
            135,136,164,165,189,190,210,211,227,228,240,241,249,250,254,255
        ]).long() 

    def forward(self,x):
        x = (x * 255).round()

        x_ycbcr = to_ycbcr(x, data_range = 255)
        H, W = x.shape[2], x.shape[3]
        K1 = H//16
        K2 = W//16

        x_blocks = blockify(x,16).transpose(1,2)
        N = x_blocks.shape[0]
        L = x_blocks.shape[1]
        x_blocks = x_blocks.contiguous().view(N * L, x_blocks.shape[2], 16, 16)
        x_blocks = self.Conv1(x_blocks)
        x_blocks = self.Maxpool(x_blocks)
        x_blocks = self.Conv2(x_blocks)
        x_blocks = self.Maxpool(x_blocks) # NL * C * 4 * 4
        x_blocks = x_blocks.view(N, L, x_blocks.shape[1], x_blocks.shape[2], x_blocks.shape[3]).transpose(1,2) # N * C * L * 4 * 4
        x_blocks = deblockify(x_blocks, (K1 * x_blocks.shape[3], K2 * x_blocks.shape[4]))
        x_blocks = self.Conv3(x_blocks) # N * C * (H/4) * (W/4)
        x_blocks = blockify(x_blocks,4).transpose(1,2).contiguous().view(N * L, x_blocks.shape[1], 4, 4).view(N * L, 16 * x_blocks.shape[1], 1, 1) # NL * 16C * 1 *1
        x_blocks = self.Conv_1x1_0(x_blocks) # NL * 4C * 1 *1
        x_blocks = x_blocks.view(N, L, x_blocks.shape[1], 1, 1) # N * L * 4C * 1 *1
        feature_patch, _ = torch.max(x_blocks, dim = 1) # N * 4C * 1 *1

        feature_patch_importance = self.Conv_1x1_2(feature_patch) # N * 192 * 1 *1
        feature_patch_importance = feature_patch_importance.view(feature_patch_importance.shape[0], self.img_ch, self.trigger_channel)  # N * 3 * 64
        topk_index = torch.topk(feature_patch_importance, self.trigger_channel//4 * 3, dim = 2, largest = False)[1]
        feature_patch = self.Conv_1x1_1(feature_patch) # N * 192 * 1 *1
        feature_patch = feature_patch.view(feature_patch.shape[0], self.img_ch, self.trigger_channel)  # N * 3 * 64
        feature_patch.scatter_(2, topk_index, 0)


        trigger_patch = torch.zeros_like(x[:,:,:16,:16])
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 256)
        trigger_patch[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]] = feature_patch
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 16, 16).repeat(1,1,K1,K2)

        trigger_weight = self.Conv5(x)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv6(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv7(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv8(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv_1x1_3(trigger_weight)
        trigger_weight = trigger_weight.repeat_interleave(16,2).repeat_interleave(16,3)

        x_ycbcr_blocks = blockify(x_ycbcr,16) - 128
        dct_blocks = dct_2d(x_ycbcr_blocks)
        dct = deblockify(dct_blocks,(H,W))
        dct_poisoned = dct + trigger_patch * trigger_weight
        dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)

        x_poisoned = x_poisoned.div(255)
       
        return x_poisoned

class Trigger_Net_v6(nn.Module):
    def __init__(self, img_ch=3, multiplier = 1, candidate_channel = 256, trigger_channel = 16):
        super(Trigger_Net_v6,self).__init__()
        
        self.img_ch = img_ch
        self.candidate_channel = candidate_channel
        self.trigger_channel = trigger_channel
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        # self.Maxpool2 = nn.MaxPool2d(kernel_size=4,stride=4)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv2 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        # self.Conv3 = conv_block(ch_in=int(128 * multiplier),ch_out=int(256 * multiplier))
        # self.Conv4 = conv_block(ch_in=int(256 * multiplier),ch_out=int(512 * multiplier))
        self.Conv_1x1_1 = nn.Conv2d(int(16 * 128 * multiplier), img_ch * candidate_channel, kernel_size=1,stride=1,padding=0)
        self.Conv_1x1_2 = nn.Conv2d(int(16 * 128 * multiplier), img_ch * candidate_channel, kernel_size=1,stride=1,padding=0)

        self.Conv5 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv6 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        self.Conv7 = conv_block(ch_in=int(128 * multiplier),ch_out=int(256 * multiplier))
        self.Conv8 = conv_block(ch_in=int(256 * multiplier),ch_out=int(512 * multiplier))
        self.Conv_1x1_3 = nn.Conv2d(int(512 * multiplier),img_ch,kernel_size=1,stride=1,padding=0)

        self.zigzag_indices = torch.tensor([
             0,  1,  5,  6, 14, 15, 27, 28, 44, 45, 65, 66, 90, 91,119,120,
             2,  4,  7, 13, 16, 26, 29, 43, 46, 64, 67, 89, 92,118,121,150,
             3,  8, 12, 17, 25, 30, 42, 47, 63, 68, 88, 93,117,122,149,151,
             9, 11, 18, 24, 31, 41, 48, 62, 69, 87, 94,116,123,148,152,177,
            10, 19, 23, 32, 40, 49, 61, 70, 86, 95,115,124,147,153,176,178,
            20, 22, 33, 39, 50, 60, 71, 85, 96,114,125,146,154,175,179,200,
            21, 34, 38, 51, 59, 72, 84, 97,113,126,145,155,174,180,199,201,
            35, 37, 52, 58, 73, 83, 98,112,127,144,156,173,181,198,202,219,
            36, 53, 57, 74, 82, 99,111,128,143,157,172,182,197,203,218,220,
            54, 56, 75, 81,100,110,129,142,158,171,183,196,204,217,221,234,
            55, 76, 80,101,109,130,141,159,170,184,195,205,216,222,233,235,
            77, 79,102,108,131,140,160,169,185,194,206,215,223,232,236,245,
            78,103,107,132,139,161,168,186,193,207,214,224,231,237,244,246,
            104,106,133,138,162,167,187,192,208,213,225,230,238,243,247,252,
            105,134,137,163,166,188,191,209,212,226,229,239,242,248,251,253,
            135,136,164,165,189,190,210,211,227,228,240,241,249,250,254,255
        ]).long() 

    def forward(self,x):
        x = (x * 255).round()

        x_ycbcr = to_ycbcr(x, data_range = 255)
        H, W = x.shape[2], x.shape[3]
        K1 = H//16
        K2 = W//16

        x_blocks = blockify(x,16).transpose(1,2)
        N = x_blocks.shape[0]
        L = x_blocks.shape[1]
        x_blocks = x_blocks.contiguous().view(N * L, x_blocks.shape[2], 16, 16)
        x_blocks = self.Conv1(x_blocks)
        x_blocks = self.Maxpool(x_blocks)
        x_blocks = self.Conv2(x_blocks)
        x_blocks = self.Maxpool(x_blocks) # NL * C * 4 * 4
        x_blocks = x_blocks.view(N, L, x_blocks.shape[1], x_blocks.shape[2], x_blocks.shape[3]).transpose(1,2) # N * C * L * 4 * 4

        feature_patch, _ = torch.max(x_blocks, dim = 2)
        feature_patch = feature_patch.view(feature_patch.shape[0], 16 * feature_patch.shape[1], 1, 1)

        feature_patch_importance = self.Conv_1x1_2(feature_patch)
        feature_patch_importance = feature_patch_importance.view(feature_patch_importance.shape[0], self.img_ch, self.candidate_channel)
        topk_index = torch.topk(feature_patch_importance, self.candidate_channel - self.trigger_channel, dim = 2, largest = False)[1]
        feature_patch = self.Conv_1x1_1(feature_patch)
        feature_patch = feature_patch.view(feature_patch.shape[0], self.img_ch, self.candidate_channel)
        feature_patch.scatter_(2, topk_index, 0)


        trigger_patch = torch.zeros_like(x[:,:,:16,:16])
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 256)
        trigger_patch[..., self.zigzag_indices[128 - self.candidate_channel//2: 128 - self.candidate_channel//2 + self.candidate_channel]] = feature_patch
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 16, 16).repeat(1,1,K1,K2)

        trigger_weight = self.Conv5(x)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv6(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv7(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv8(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv_1x1_3(trigger_weight)
        trigger_weight = trigger_weight.repeat_interleave(16,2).repeat_interleave(16,3)

        x_ycbcr_blocks = blockify(x_ycbcr,16) - 128
        dct_blocks = dct_2d(x_ycbcr_blocks)
        dct = deblockify(dct_blocks,(H,W))
        dct_poisoned = dct + trigger_patch * trigger_weight
        dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)

        x_poisoned = x_poisoned.div(255)
       
        return x_poisoned

class Trigger_Net_v7(nn.Module):
    def __init__(self, img_ch=3, multiplier = 1, trigger_channel = 64):
        super(Trigger_Net_v7,self).__init__()
        
        self.img_ch = img_ch
        self.trigger_channel = trigger_channel
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv2 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        self.Conv3 = conv_block(ch_in=int(128 * multiplier),ch_out=int(256 * multiplier))
        self.Conv4 = conv_block(ch_in=int(256 * multiplier),ch_out=int(512 * multiplier))
        self.Conv_1x1_1 = nn.Conv2d(int(512 * multiplier), img_ch * trigger_channel, kernel_size=1,stride=1,padding=0)
        self.Conv_1x1_2 = nn.Conv2d(int(512 * multiplier), img_ch * trigger_channel, kernel_size=1,stride=1,padding=0)

        self.Conv5 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv6 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        self.Conv7 = conv_block(ch_in=int(128 * multiplier),ch_out=int(256 * multiplier))
        self.Conv8 = conv_block(ch_in=int(256 * multiplier),ch_out=int(512 * multiplier))
        self.Conv_1x1_3 = nn.Conv2d(int(512 * multiplier),img_ch,kernel_size=1,stride=1,padding=0)

        self.zigzag_indices = torch.tensor([
             0,  1,  5,  6, 14, 15, 27, 28, 44, 45, 65, 66, 90, 91,119,120,
             2,  4,  7, 13, 16, 26, 29, 43, 46, 64, 67, 89, 92,118,121,150,
             3,  8, 12, 17, 25, 30, 42, 47, 63, 68, 88, 93,117,122,149,151,
             9, 11, 18, 24, 31, 41, 48, 62, 69, 87, 94,116,123,148,152,177,
            10, 19, 23, 32, 40, 49, 61, 70, 86, 95,115,124,147,153,176,178,
            20, 22, 33, 39, 50, 60, 71, 85, 96,114,125,146,154,175,179,200,
            21, 34, 38, 51, 59, 72, 84, 97,113,126,145,155,174,180,199,201,
            35, 37, 52, 58, 73, 83, 98,112,127,144,156,173,181,198,202,219,
            36, 53, 57, 74, 82, 99,111,128,143,157,172,182,197,203,218,220,
            54, 56, 75, 81,100,110,129,142,158,171,183,196,204,217,221,234,
            55, 76, 80,101,109,130,141,159,170,184,195,205,216,222,233,235,
            77, 79,102,108,131,140,160,169,185,194,206,215,223,232,236,245,
            78,103,107,132,139,161,168,186,193,207,214,224,231,237,244,246,
            104,106,133,138,162,167,187,192,208,213,225,230,238,243,247,252,
            105,134,137,163,166,188,191,209,212,226,229,239,242,248,251,253,
            135,136,164,165,189,190,210,211,227,228,240,241,249,250,254,255
        ]).long() 

    def forward(self,x):
        x = (x * 255).round()

        x_ycbcr = to_ycbcr(x, data_range = 255)
        H, W = x.shape[2], x.shape[3]
        K1 = H//16
        K2 = W//16

        feat = self.Conv1(x)
        feat = self.Maxpool(feat)
        feat = self.Conv2(feat)
        feat = self.Maxpool(feat)
        feat = self.Conv3(feat)
        feat = self.Maxpool(feat)
        feat = self.Conv4(feat)
        feat = feat.view(feat.shape[0],feat.shape[1],-1,1)
        feature_patch,_ = torch.max(feat, dim = 2, keepdim = True)

        feature_patch_importance = self.Conv_1x1_2(feature_patch)
        feature_patch_importance = feature_patch_importance.view(feature_patch_importance.shape[0], self.img_ch, self.trigger_channel)
        topk_index = torch.topk(feature_patch_importance, self.trigger_channel//4 * 3, dim = 2, largest = False)[1]
        feature_patch = self.Conv_1x1_1(feature_patch)
        feature_patch = feature_patch.view(feature_patch.shape[0], self.img_ch, self.trigger_channel)
        feature_patch.scatter_(2, topk_index, 0)


        trigger_patch = torch.zeros_like(x[:,:,:16,:16])
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 256)
        trigger_patch[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]] = feature_patch
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 16, 16).repeat(1,1,K1,K2)

        trigger_weight = self.Conv5(x)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv6(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv7(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv8(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv_1x1_3(trigger_weight)
        trigger_weight = trigger_weight.repeat_interleave(16,2).repeat_interleave(16,3)

        x_ycbcr_blocks = blockify(x_ycbcr,16) - 128
        dct_blocks = dct_2d(x_ycbcr_blocks)
        dct = deblockify(dct_blocks,(H,W))
        dct_poisoned = dct + trigger_patch * trigger_weight
        dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)

        x_poisoned = x_poisoned.div(255)
       
        return x_poisoned


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SEModule(nn.Module):
    '''Squeeze and Excitation Module'''
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return input * x

class CAModule(nn.Module):
    '''Channel Attention Module'''
    def __init__(self, channels, reduction):
        super(CAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        x = self.shared_mlp(avg_pool) + self.shared_mlp(max_pool)
        x = self.sigmoid(x)

        return input * x

class SAModule(nn.Module):
    '''Spatial Attention Module'''
    def __init__(self):
        super(SAModule, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        avg_c = torch.mean(x, 1, True)
        max_c, _ = torch.max(x, 1, True)
        x = torch.cat((avg_c, max_c), 1)
        x = self.conv(x)
        x = self.sigmoid(x)
        return input * x

class BottleNeck_IR(nn.Module):
    '''Improved Residual Bottlenecks'''
    def __init__(self, in_channel, out_channel, stride, dim_match):
        super(BottleNeck_IR, self).__init__()
        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channel),
                                       nn.Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       nn.PReLU(out_channel),
                                       nn.Conv2d(out_channel, out_channel, (3, 3), stride, 1, bias=False),
                                       nn.BatchNorm2d(out_channel))
        if dim_match:
            self.shortcut_layer = None
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        shortcut = x
        res = self.res_layer(x)

        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)

        return shortcut + res

class BottleNeck_IR_SE(nn.Module):
    '''Improved Residual Bottlenecks with Squeeze and Excitation Module'''
    def __init__(self, in_channel, out_channel, stride, dim_match):
        super(BottleNeck_IR_SE, self).__init__()
        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channel),
                                       nn.Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       nn.PReLU(out_channel),
                                       nn.Conv2d(out_channel, out_channel, (3, 3), stride, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       SEModule(out_channel, 16))
        if dim_match:
            self.shortcut_layer = None
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        shortcut = x
        res = self.res_layer(x)

        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)

        return shortcut + res

class BottleNeck_IR_CAM(nn.Module):
    '''Improved Residual Bottlenecks with Channel Attention Module'''
    def __init__(self, in_channel, out_channel, stride, dim_match):
        super(BottleNeck_IR_CAM, self).__init__()
        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channel),
                                       nn.Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       nn.PReLU(out_channel),
                                       nn.Conv2d(out_channel, out_channel, (3, 3), stride, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       CAModule(out_channel, 16))
        if dim_match:
            self.shortcut_layer = None
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        shortcut = x
        res = self.res_layer(x)

        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)

        return shortcut + res

class BottleNeck_IR_SAM(nn.Module):
    '''Improved Residual Bottlenecks with Spatial Attention Module'''
    def __init__(self, in_channel, out_channel, stride, dim_match):
        super(BottleNeck_IR_SAM, self).__init__()
        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channel),
                                       nn.Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       nn.PReLU(out_channel),
                                       nn.Conv2d(out_channel, out_channel, (3, 3), stride, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       SAModule())
        if dim_match:
            self.shortcut_layer = None
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        shortcut = x
        res = self.res_layer(x)

        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)

        return shortcut + res

class BottleNeck_IR_CBAM(nn.Module):
    '''Improved Residual Bottleneck with Channel Attention Module and Spatial Attention Module'''
    def __init__(self, in_channel, out_channel, stride, dim_match):
        super(BottleNeck_IR_CBAM, self).__init__()
        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channel),
                                       nn.Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       nn.PReLU(out_channel),
                                       nn.Conv2d(out_channel, out_channel, (3, 3), stride, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       CAModule(out_channel, 16),
                                       SAModule()
                                       )
        if dim_match:
            self.shortcut_layer = None
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        shortcut = x
        res = self.res_layer(x)

        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)

        return shortcut + res


filter_list = [64, 64, 128, 256, 512]
def get_layers(num_layers):
    if num_layers == 50:
        return [3, 4, 14, 3]
    elif num_layers == 100:
        return [3, 13, 30, 3]
    elif num_layers == 152:
        return [3, 8, 36, 3]

class CBAMResNet(nn.Module):
    def __init__(self, num_layers, feature_dim=512, drop_ratio=0.4, mode='ir',filter_list=filter_list):
        super(CBAMResNet, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50, 100 or 152'
        assert mode in ['ir', 'ir_se', 'ir_cam', 'ir_sam', 'ir_cbam'], 'mode should be ir, ir_se, ir_cam, ir_sam or ir_cbam'
        layers = get_layers(num_layers)
        if mode == 'ir':
            block = BottleNeck_IR
        elif mode == 'ir_se':
            block = BottleNeck_IR_SE
        elif mode == 'ir_cam':
            block = BottleNeck_IR_CAM
        elif mode == 'ir_sam':
            block = BottleNeck_IR_SAM
        elif mode == 'ir_cbam':
            block = BottleNeck_IR_CBAM

        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), stride=1, padding=1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.PReLU(64))
        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=2)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=2)

        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                          nn.Dropout(drop_ratio),
                                          Flatten(),
                                          nn.Linear(512 * 7 * 7, feature_dim),
                                          nn.BatchNorm1d(feature_dim))

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_channel, out_channel, blocks, stride):
        layers = []
        layers.append(block(in_channel, out_channel, stride, False))
        for i in range(1, blocks):
            layers.append(block(out_channel, out_channel, 1, True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output_layer(x)

        return x


class Trigger_Net_v20(nn.Module):
    def __init__(self, img_ch=3, multiplier = 1, trigger_channel = 64):
        super(Trigger_Net_v20,self).__init__()
        self.img_ch = img_ch
        self.trigger_channel = trigger_channel
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        # self.Maxpool2 = nn.MaxPool2d(kernel_size=4,stride=4)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv2 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        
    def forward(self,x):
        x = (x * 255).round()

        x_ycbcr = to_ycbcr(x, data_range = 255)
        H, W = x.shape[2], x.shape[3]
        K1 = H//32
        K2 = W//32

        x_ycbcr_blocks = blockify(x_ycbcr,32) - 128
        dct_blocks = dct_2d(x_ycbcr_blocks)
        dct_poisoned_blocks = dct_blocks
        dct_poisoned_blocks[:,1:,:,15,15] = 100
        dct_poisoned_blocks[:,1:,:,31,31] = 100
        # dct = deblockify(dct_blocks,(H,W))
        # dct_poisoned = dct + trigger_patch * trigger_weight
        # dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)

        x_poisoned = x_poisoned.div(255)
        
        return x_poisoned


class Trigger_Net_v21(nn.Module):
    def __init__(self, img_ch=3, multiplier = 1, trigger_channel = 64):
        super(Trigger_Net_v21,self).__init__()
        self.img_ch = img_ch
        self.trigger_channel = trigger_channel
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        # self.Maxpool2 = nn.MaxPool2d(kernel_size=4,stride=4)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv2 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        
    def forward(self,x):
        x = (x * 255).round()

        x_ycbcr = to_ycbcr(x, data_range = 255)
        H, W = x.shape[2], x.shape[3]
        K1 = H//32
        K2 = W//32

        x_ycbcr_blocks = blockify(x_ycbcr,32) - 128
        dct_blocks = dct_2d(x_ycbcr_blocks)
        dct_poisoned_blocks = dct_blocks
        dct_poisoned_blocks[:,1:,:,12:19,12:19] = 50
        dct_poisoned_blocks[:,1:,:,25:,25:] = 50
        # dct_poisoned_blocks[:,1:,:,12:19,12:19] = 100
        # dct_poisoned_blocks[:,1:,:,25:,25:] = 100
        # dct = deblockify(dct_blocks,(H,W))
        # dct_poisoned = dct + trigger_patch * trigger_weight
        # dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)

        x_poisoned = x_poisoned.div(255)
        
        return x_poisoned


class Trigger_Net_v21_seg(nn.Module):
    def __init__(self, img_ch=3, multiplier = 1, trigger_channel = 64):
        super(Trigger_Net_v21_seg,self).__init__()
        self.img_ch = img_ch
        self.trigger_channel = trigger_channel
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        # self.Maxpool2 = nn.MaxPool2d(kernel_size=4,stride=4)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv2 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        
    def forward(self,x):
        x = (x * 255).round()

        x_ycbcr = to_ycbcr(x, data_range = 255)
        H, W = x.shape[2], x.shape[3]
        K1 = H//32
        K2 = W//32

        x_ycbcr_blocks = blockify(x_ycbcr,32) - 128
        dct_blocks = dct_2d(x_ycbcr_blocks)
        dct_poisoned_blocks = dct_blocks
        dct_poisoned_blocks[:,1:,:,12:19,12:19] = 180
        dct_poisoned_blocks[:,1:,:,25:,25:] = 180
        # dct_poisoned_blocks[:,1:,:,12:19,12:19] = 100
        # dct_poisoned_blocks[:,1:,:,25:,25:] = 100
        # dct = deblockify(dct_blocks,(H,W))
        # dct_poisoned = dct + trigger_patch * trigger_weight
        # dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)

        x_poisoned = x_poisoned.div(255)
        
        return x_poisoned

class Trigger_Net_v21_fr(nn.Module):
    def __init__(self, img_ch=3, multiplier = 1, trigger_channel = 64):
        super(Trigger_Net_v21_fr,self).__init__()
        self.img_ch = img_ch
        self.trigger_channel = trigger_channel
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        # self.Maxpool2 = nn.MaxPool2d(kernel_size=4,stride=4)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv2 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        
    def forward(self,x):
        x = (x * 255).round()

        x_ycbcr = to_ycbcr(x, data_range = 255)
        H, W = x.shape[2], x.shape[3]
        K1 = H//32
        K2 = W//32

        x_ycbcr_blocks = blockify(x_ycbcr,32) - 128
        dct_blocks = dct_2d(x_ycbcr_blocks)
        dct_poisoned_blocks = dct_blocks
        dct_poisoned_blocks[:,1:,:,12:19,12:19] = 180
        dct_poisoned_blocks[:,1:,:,25:,25:] = 180
        # dct_poisoned_blocks[:,1:,:,12:19,12:19] = 100
        # dct_poisoned_blocks[:,1:,:,25:,25:] = 100
        # dct = deblockify(dct_blocks,(H,W))
        # dct_poisoned = dct + trigger_patch * trigger_weight
        # dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)

        x_poisoned = x_poisoned.div(255)
        
        return x_poisoned


class Trigger_Net_v22(nn.Module):
    def __init__(self, img_ch=3, multiplier = 1, trigger_channel = 64):
        super(Trigger_Net_v22,self).__init__()
        
        self.img_ch = img_ch
        self.trigger_channel = trigger_channel
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv2 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))

        self.Conv_1x1_1 = nn.Conv2d(int(16 * 128 * multiplier), img_ch * trigger_channel, kernel_size=1,stride=1,padding=0)
        self.Conv_1x1_2 = nn.Conv2d(int(16 * 128 * multiplier), img_ch * trigger_channel, kernel_size=1,stride=1,padding=0)

        self.Conv5 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv6 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        self.Conv7 = conv_block(ch_in=int(128 * multiplier),ch_out=int(256 * multiplier))
        self.Conv8 = conv_block(ch_in=int(256 * multiplier),ch_out=int(512 * multiplier))
        self.Conv_1x1_3 = nn.Conv2d(int(512 * multiplier),img_ch,kernel_size=1,stride=1,padding=0)

        self.zigzag_indices = torch.tensor([
             0,  1,  5,  6, 14, 15, 27, 28, 44, 45, 65, 66, 90, 91,119,120,
             2,  4,  7, 13, 16, 26, 29, 43, 46, 64, 67, 89, 92,118,121,150,
             3,  8, 12, 17, 25, 30, 42, 47, 63, 68, 88, 93,117,122,149,151,
             9, 11, 18, 24, 31, 41, 48, 62, 69, 87, 94,116,123,148,152,177,
            10, 19, 23, 32, 40, 49, 61, 70, 86, 95,115,124,147,153,176,178,
            20, 22, 33, 39, 50, 60, 71, 85, 96,114,125,146,154,175,179,200,
            21, 34, 38, 51, 59, 72, 84, 97,113,126,145,155,174,180,199,201,
            35, 37, 52, 58, 73, 83, 98,112,127,144,156,173,181,198,202,219,
            36, 53, 57, 74, 82, 99,111,128,143,157,172,182,197,203,218,220,
            54, 56, 75, 81,100,110,129,142,158,171,183,196,204,217,221,234,
            55, 76, 80,101,109,130,141,159,170,184,195,205,216,222,233,235,
            77, 79,102,108,131,140,160,169,185,194,206,215,223,232,236,245,
            78,103,107,132,139,161,168,186,193,207,214,224,231,237,244,246,
            104,106,133,138,162,167,187,192,208,213,225,230,238,243,247,252,
            105,134,137,163,166,188,191,209,212,226,229,239,242,248,251,253,
            135,136,164,165,189,190,210,211,227,228,240,241,249,250,254,255
        ]).long() 

    def transform_all(self, x):
        k = random.randrange(0,3)
        if k == 0:
            return self.transform_gb(x)
        elif k == 1:
            return self.transform_gn(x)
        elif k == 2:
            return self.transform_jpeg(x)

    def transform_gb(self, x):
        g_sigma = random.uniform(0.5,0.7)
        transform = T.GaussianBlur(kernel_size=(5, 5), sigma=(g_sigma, g_sigma))
        return transform(x)

    def transform_jpeg(self, input_img):
        # x = x.squeeze()
        B, C, H, W = input_img.shape
        for i in range(B):
            x = T.ToPILImage()(input_img[i,:,:,:])
            x = T.Lambda(self.randomJPEGcompression)(x)
            x = T.ToTensor()(x)
            x = x.cuda()
            input_img[i,:,:,:] = x
        return input_img

    def transform_scb(self, x):
        depth = 3
        x = (x * (2**depth - 1)).int()
        x = (x / (2**depth - 1)).float()
        return x

    def transform_gn(self, x):
        sigma = random.uniform(0.2,1.0)
        x = AddGaussianNoise(0,sigma)(x)
        return x

    def randomJPEGcompression(self, image):
        qf = 10 * random.randrange(1, 10)
        outputIoStream = BytesIO()
        image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
        outputIoStream.seek(0)
        return Image.open(outputIoStream)

    def forward(self,x):
        x = (x * 255).round()

        x_ycbcr = to_ycbcr(x, data_range = 255)
        H, W = x.shape[2], x.shape[3]
        K1 = H//16
        K2 = W//16

        trigger_weight = self.Conv5(x)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv6(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv7(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv8(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv_1x1_3(trigger_weight)
        trigger_weight = trigger_weight.repeat_interleave(16,2).repeat_interleave(16,3)

        x_blocks = blockify(x,16).transpose(1,2)
        N = x_blocks.shape[0]
        L = x_blocks.shape[1]
        x_blocks = x_blocks.contiguous().view(N * L, x_blocks.shape[2], 16, 16)
        x_blocks = self.Conv1(x_blocks)
        x_blocks = self.Maxpool(x_blocks)
        x_blocks = self.Conv2(x_blocks)
        x_blocks = self.Maxpool(x_blocks) # NL * C * 4 * 4
        x_blocks = x_blocks.view(N, L, x_blocks.shape[1], x_blocks.shape[2], x_blocks.shape[3]).transpose(1,2) # N * C * L * 4 * 4
        feature_patch, _ = torch.max(x_blocks, dim = 2)
        feature_patch = feature_patch.view(feature_patch.shape[0], 16 * feature_patch.shape[1], 1, 1) # N * 16C * 1 * 1

        feature_patch = self.Conv_1x1_1(feature_patch)
        feature_patch = feature_patch.view(feature_patch.shape[0], self.img_ch, self.trigger_channel)

        x_ycbcr_blocks = blockify(x_ycbcr,16) - 128
        dct_blocks = dct_2d(x_ycbcr_blocks)
        dct = deblockify(dct_blocks,(H,W))
        trigger_patch = torch.zeros_like(x[:,:,:16,:16], dtype = feature_patch.dtype)
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 256)
        trigger_patch[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]] = feature_patch.detach()
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 16, 16).repeat(1,1,K1,K2)
        dct_poisoned = dct + trigger_patch * trigger_weight.detach()
        dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)
        x_poisoned = x_poisoned.div(255)
        x_poisoned = self.transform_all(x_poisoned)

        x_poisoned_ycbcr = to_ycbcr(x_poisoned, data_range = 255)
        x_poisoned_ycbcr_blocks = blockify(x_poisoned_ycbcr,16) - 128
        dct_x_poisoned_ycbcr_blocks = dct_2d(x_poisoned_ycbcr_blocks)
        dct_x_poisoned_ycbcr = deblockify(dct_x_poisoned_ycbcr_blocks,(H,W))
        delta_dct = (dct_x_poisoned_ycbcr - dct_poisoned.detach())/(trigger_patch * trigger_weight.detach())
        delta_dct_blocks = blockify(delta_dct,16).abs() # N * C * L * 16 * 16
        delta_dct_blocks = delta_dct_blocks.sum(2) # N * C * 16 * 16
        delta_dct_blocks = delta_dct_blocks.view(delta_dct_blocks.shape[0],delta_dct_blocks.shape[1],256)
        feature_patch_importance = delta_dct_blocks[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]]

        topk_index = torch.topk(feature_patch_importance, self.trigger_channel//4 * 3, dim = 2, largest = True)[1]
        feature_patch.scatter_(2, topk_index, 0)


        trigger_patch = torch.zeros_like(x[:,:,:16,:16], dtype = feature_patch.dtype)
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 256)
        trigger_patch[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]] = feature_patch
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 16, 16).repeat(1,1,K1,K2)

        dct_poisoned = dct + trigger_patch * trigger_weight
        dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)

        x_poisoned = x_poisoned.div(255)
        
        return x_poisoned


class Trigger_Net_v23(nn.Module):
    def __init__(self, img_ch=3, multiplier = 1, trigger_channel = 64):
        super(Trigger_Net_v23,self).__init__()
        
        self.img_ch = img_ch
        self.trigger_channel = trigger_channel
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv2 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))

        self.Conv_1x1_1 = nn.Conv2d(int(16 * 128 * multiplier), img_ch * trigger_channel, kernel_size=1,stride=1,padding=0)
        self.Conv_1x1_2 = nn.Conv2d(int(16 * 128 * multiplier), img_ch * trigger_channel, kernel_size=1,stride=1,padding=0)

        self.Conv5 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv6 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        self.Conv7 = conv_block(ch_in=int(128 * multiplier),ch_out=int(256 * multiplier))
        self.Conv8 = conv_block(ch_in=int(256 * multiplier),ch_out=int(512 * multiplier))
        self.Conv_1x1_3 = nn.Conv2d(int(512 * multiplier),img_ch,kernel_size=1,stride=1,padding=0)

        self.zigzag_indices = torch.tensor([
             0,  1,  5,  6, 14, 15, 27, 28, 44, 45, 65, 66, 90, 91,119,120,
             2,  4,  7, 13, 16, 26, 29, 43, 46, 64, 67, 89, 92,118,121,150,
             3,  8, 12, 17, 25, 30, 42, 47, 63, 68, 88, 93,117,122,149,151,
             9, 11, 18, 24, 31, 41, 48, 62, 69, 87, 94,116,123,148,152,177,
            10, 19, 23, 32, 40, 49, 61, 70, 86, 95,115,124,147,153,176,178,
            20, 22, 33, 39, 50, 60, 71, 85, 96,114,125,146,154,175,179,200,
            21, 34, 38, 51, 59, 72, 84, 97,113,126,145,155,174,180,199,201,
            35, 37, 52, 58, 73, 83, 98,112,127,144,156,173,181,198,202,219,
            36, 53, 57, 74, 82, 99,111,128,143,157,172,182,197,203,218,220,
            54, 56, 75, 81,100,110,129,142,158,171,183,196,204,217,221,234,
            55, 76, 80,101,109,130,141,159,170,184,195,205,216,222,233,235,
            77, 79,102,108,131,140,160,169,185,194,206,215,223,232,236,245,
            78,103,107,132,139,161,168,186,193,207,214,224,231,237,244,246,
            104,106,133,138,162,167,187,192,208,213,225,230,238,243,247,252,
            105,134,137,163,166,188,191,209,212,226,229,239,242,248,251,253,
            135,136,164,165,189,190,210,211,227,228,240,241,249,250,254,255
        ]).long() 

    def transform_all(self, x):
        k = random.randrange(0,3)
        if k == 0:
            return self.transform_gb(x)
        elif k == 1:
            return self.transform_gn(x)
        elif k == 2:
            return self.transform_jpeg(x)

    def transform_gb(self, x):
        g_sigma = random.uniform(0.5,0.7)
        transform = T.GaussianBlur(kernel_size=(5, 5), sigma=(g_sigma, g_sigma))
        return transform(x)

    def transform_jpeg(self, input_img):
        # x = x.squeeze()
        B, C, H, W = input_img.shape
        for i in range(B):
            x = T.ToPILImage()(input_img[i,:,:,:])
            x = T.Lambda(self.randomJPEGcompression)(x)
            x = T.ToTensor()(x)
            x = x.cuda()
            input_img[i,:,:,:] = x
        return input_img

    def transform_scb(self, x):
        depth = 3
        x = (x * (2**depth - 1)).int()
        x = (x / (2**depth - 1)).float()
        return x

    def transform_gn(self, x):
        sigma = random.uniform(0.04,0.1)
        x = AddGaussianNoise(0,sigma)(x)
        return x

    def randomJPEGcompression(self, image):
        qf = 10 * random.randrange(1, 10)
        outputIoStream = BytesIO()
        image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
        outputIoStream.seek(0)
        return Image.open(outputIoStream)

    def forward(self,x):
        x = (x * 255).round()

        x_ycbcr = to_ycbcr(x, data_range = 255)
        H, W = x.shape[2], x.shape[3]
        K1 = H//16
        K2 = W//16

        trigger_weight = self.Conv5(x)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv6(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv7(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv8(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv_1x1_3(trigger_weight)
        trigger_weight = trigger_weight.repeat_interleave(16,2).repeat_interleave(16,3)

        x_blocks = blockify(x,16).transpose(1,2)
        N = x_blocks.shape[0]
        L = x_blocks.shape[1]
        x_blocks = x_blocks.contiguous().view(N * L, x_blocks.shape[2], 16, 16)
        x_blocks = self.Conv1(x_blocks)
        x_blocks = self.Maxpool(x_blocks)
        x_blocks = self.Conv2(x_blocks)
        x_blocks = self.Maxpool(x_blocks) # NL * C * 4 * 4
        x_blocks = x_blocks.view(N, L, x_blocks.shape[1], x_blocks.shape[2], x_blocks.shape[3]).transpose(1,2) # N * C * L * 4 * 4
        feature_patch, _ = torch.max(x_blocks, dim = 2)
        feature_patch = feature_patch.view(feature_patch.shape[0], 16 * feature_patch.shape[1], 1, 1) # N * 16C * 1 * 1

        feature_patch = self.Conv_1x1_1(feature_patch)
        feature_patch = feature_patch.view(feature_patch.shape[0], self.img_ch, self.trigger_channel)

        x_ycbcr_blocks = blockify(x_ycbcr,16) - 128
        dct_blocks = dct_2d(x_ycbcr_blocks)
        dct = deblockify(dct_blocks,(H,W))
        trigger_patch = torch.zeros_like(x[:,:,:16,:16], dtype = feature_patch.dtype)
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 256)
        trigger_patch[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]] = feature_patch.detach()
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 16, 16).repeat(1,1,K1,K2)
        dct_poisoned = dct + trigger_patch * trigger_weight.detach()
        dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)
        x_poisoned = x_poisoned.div(255)
        x_poisoned = self.transform_gn(x_poisoned)

        x_poisoned_ycbcr = to_ycbcr(x_poisoned, data_range = 255)
        x_poisoned_ycbcr_blocks = blockify(x_poisoned_ycbcr,16) - 128
        dct_x_poisoned_ycbcr_blocks = dct_2d(x_poisoned_ycbcr_blocks)
        dct_x_poisoned_ycbcr = deblockify(dct_x_poisoned_ycbcr_blocks,(H,W))
        delta_dct = (dct_x_poisoned_ycbcr - dct_poisoned.detach())/(trigger_patch * trigger_weight.detach())
        delta_dct_blocks = blockify(delta_dct,16).abs() # N * C * L * 16 * 16
        delta_dct_blocks = delta_dct_blocks.sum(2) # N * C * 16 * 16
        delta_dct_blocks = delta_dct_blocks.view(delta_dct_blocks.shape[0],delta_dct_blocks.shape[1],256)
        feature_patch_importance = delta_dct_blocks[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]]

        topk_index = torch.topk(feature_patch_importance, self.trigger_channel//4 * 3, dim = 2, largest = True)[1]
        feature_patch.scatter_(2, topk_index, 0)


        trigger_patch = torch.zeros_like(x[:,:,:16,:16], dtype = feature_patch.dtype)
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 256)
        trigger_patch[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]] = feature_patch
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 16, 16).repeat(1,1,K1,K2)

        dct_poisoned = dct + trigger_patch * trigger_weight
        dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)

        x_poisoned = x_poisoned.div(255)
        
        return x_poisoned


class Trigger_Net_v24(nn.Module):
    def __init__(self, img_ch=3, multiplier = 1, trigger_channel = 64):
        super(Trigger_Net_v24,self).__init__()
        
        self.img_ch = img_ch
        self.trigger_channel = trigger_channel
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv2 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))

        self.Conv_1x1_1 = nn.Conv2d(int(16 * 128 * multiplier), img_ch * trigger_channel, kernel_size=1,stride=1,padding=0)
        self.Conv_1x1_2 = nn.Conv2d(int(16 * 128 * multiplier), img_ch * trigger_channel, kernel_size=1,stride=1,padding=0)

        self.Conv5 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv6 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        self.Conv7 = conv_block(ch_in=int(128 * multiplier),ch_out=int(256 * multiplier))
        self.Conv8 = conv_block(ch_in=int(256 * multiplier),ch_out=int(512 * multiplier))
        self.Conv_1x1_3 = nn.Conv2d(int(512 * multiplier),img_ch,kernel_size=1,stride=1,padding=0)

        self.zigzag_indices = torch.tensor([
             0,  1,  5,  6, 14, 15, 27, 28, 44, 45, 65, 66, 90, 91,119,120,
             2,  4,  7, 13, 16, 26, 29, 43, 46, 64, 67, 89, 92,118,121,150,
             3,  8, 12, 17, 25, 30, 42, 47, 63, 68, 88, 93,117,122,149,151,
             9, 11, 18, 24, 31, 41, 48, 62, 69, 87, 94,116,123,148,152,177,
            10, 19, 23, 32, 40, 49, 61, 70, 86, 95,115,124,147,153,176,178,
            20, 22, 33, 39, 50, 60, 71, 85, 96,114,125,146,154,175,179,200,
            21, 34, 38, 51, 59, 72, 84, 97,113,126,145,155,174,180,199,201,
            35, 37, 52, 58, 73, 83, 98,112,127,144,156,173,181,198,202,219,
            36, 53, 57, 74, 82, 99,111,128,143,157,172,182,197,203,218,220,
            54, 56, 75, 81,100,110,129,142,158,171,183,196,204,217,221,234,
            55, 76, 80,101,109,130,141,159,170,184,195,205,216,222,233,235,
            77, 79,102,108,131,140,160,169,185,194,206,215,223,232,236,245,
            78,103,107,132,139,161,168,186,193,207,214,224,231,237,244,246,
            104,106,133,138,162,167,187,192,208,213,225,230,238,243,247,252,
            105,134,137,163,166,188,191,209,212,226,229,239,242,248,251,253,
            135,136,164,165,189,190,210,211,227,228,240,241,249,250,254,255
        ]).long() 

    def transform_all(self, x):
        k = random.randrange(0,3)
        if k == 0:
            return self.transform_gb(x)
        elif k == 1:
            return self.transform_gn(x)
        elif k == 2:
            return self.transform_jpeg(x)

    def transform_gb(self, x):
        g_sigma = random.uniform(0.5,0.7)
        transform = T.GaussianBlur(kernel_size=(5, 5), sigma=(g_sigma, g_sigma))
        return transform(x)

    def transform_jpeg(self, input_img):
        # x = x.squeeze()
        B, C, H, W = input_img.shape
        for i in range(B):
            x = T.ToPILImage()(input_img[i,:,:,:])
            x = T.Lambda(self.randomJPEGcompression)(x)
            x = T.ToTensor()(x)
            x = x.cuda()
            input_img[i,:,:,:] = x
        return input_img

    def transform_scb(self, x):
        depth = 3
        x = (x * (2**depth - 1)).int()
        x = (x / (2**depth - 1)).float()
        return x

    def transform_gn(self, x):
        sigma = random.uniform(0.04,0.1)
        x = AddGaussianNoise(0,sigma)(x)
        return x

    def randomJPEGcompression(self, image):
        qf = 10 * random.randrange(1, 10)
        outputIoStream = BytesIO()
        image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
        outputIoStream.seek(0)
        return Image.open(outputIoStream)

    def forward(self,x):
        x = (x * 255).round()

        x_ycbcr = to_ycbcr(x, data_range = 255)
        H, W = x.shape[2], x.shape[3]
        K1 = H//16
        K2 = W//16

        trigger_weight = self.Conv5(x)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv6(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv7(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv8(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv_1x1_3(trigger_weight)
        trigger_weight = trigger_weight.repeat_interleave(16,2).repeat_interleave(16,3)

        x_blocks = blockify(x,16).transpose(1,2)
        N = x_blocks.shape[0]
        L = x_blocks.shape[1]
        x_blocks = x_blocks.contiguous().view(N * L, x_blocks.shape[2], 16, 16)
        x_blocks = self.Conv1(x_blocks)
        x_blocks = self.Maxpool(x_blocks)
        x_blocks = self.Conv2(x_blocks)
        x_blocks = self.Maxpool(x_blocks) # NL * C * 4 * 4
        x_blocks = x_blocks.view(N, L, x_blocks.shape[1], x_blocks.shape[2], x_blocks.shape[3]).transpose(1,2) # N * C * L * 4 * 4
        feature_patch, _ = torch.max(x_blocks, dim = 2)
        feature_patch = feature_patch.view(feature_patch.shape[0], 16 * feature_patch.shape[1], 1, 1) # N * 16C * 1 * 1

        feature_patch = self.Conv_1x1_1(feature_patch)
        feature_patch = feature_patch.view(feature_patch.shape[0], self.img_ch, self.trigger_channel)

        x_ycbcr_blocks = blockify(x_ycbcr,16) - 128
        dct_blocks = dct_2d(x_ycbcr_blocks)
        dct = deblockify(dct_blocks,(H,W))
        trigger_patch = torch.zeros_like(x[:,:,:16,:16], dtype = feature_patch.dtype)
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 256)
        trigger_patch[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]] = feature_patch.detach()
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 16, 16).repeat(1,1,K1,K2)
        dct_poisoned = dct + trigger_patch * trigger_weight.detach()
        dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)
        x_poisoned = x_poisoned.div(255)
        x_poisoned = self.transform_jpeg(x_poisoned)

        x_poisoned_ycbcr = to_ycbcr(x_poisoned, data_range = 255)
        x_poisoned_ycbcr_blocks = blockify(x_poisoned_ycbcr,16) - 128
        dct_x_poisoned_ycbcr_blocks = dct_2d(x_poisoned_ycbcr_blocks)
        dct_x_poisoned_ycbcr = deblockify(dct_x_poisoned_ycbcr_blocks,(H,W))
        delta_dct = (dct_x_poisoned_ycbcr - dct_poisoned.detach())/(trigger_patch * trigger_weight.detach())
        delta_dct_blocks = blockify(delta_dct,16).abs() # N * C * L * 16 * 16
        delta_dct_blocks = delta_dct_blocks.sum(2) # N * C * 16 * 16
        delta_dct_blocks = delta_dct_blocks.view(delta_dct_blocks.shape[0],delta_dct_blocks.shape[1],256)
        feature_patch_importance = delta_dct_blocks[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]]

        topk_index = torch.topk(feature_patch_importance, self.trigger_channel//4 * 3, dim = 2, largest = True)[1]
        feature_patch.scatter_(2, topk_index, 0)

        trigger_patch = torch.zeros_like(x[:,:,:16,:16], dtype = feature_patch.dtype)
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 256)
        trigger_patch[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]] = feature_patch
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 16, 16).repeat(1,1,K1,K2)

        dct_poisoned = dct + trigger_patch * trigger_weight
        dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)

        x_poisoned = x_poisoned.div(255)
        
        return x_poisoned



class Trigger_Net_v25(nn.Module):
    def __init__(self, img_ch=3, multiplier = 1, trigger_channel = 64):
        super(Trigger_Net_v25,self).__init__()
        
        self.img_ch = img_ch
        self.trigger_channel = trigger_channel
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv2 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))

        self.Conv_1x1_1 = nn.Conv2d(int(16 * 128 * multiplier), img_ch * trigger_channel, kernel_size=1,stride=1,padding=0)
        self.Conv_1x1_2 = nn.Conv2d(int(16 * 128 * multiplier), img_ch * trigger_channel, kernel_size=1,stride=1,padding=0)

        self.Conv5 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv6 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        self.Conv7 = conv_block(ch_in=int(128 * multiplier),ch_out=int(256 * multiplier))
        self.Conv8 = conv_block(ch_in=int(256 * multiplier),ch_out=int(512 * multiplier))
        self.Conv_1x1_3 = nn.Conv2d(int(512 * multiplier),img_ch,kernel_size=1,stride=1,padding=0)

        self.zigzag_indices = torch.tensor([
             0,  1,  5,  6, 14, 15, 27, 28, 44, 45, 65, 66, 90, 91,119,120,
             2,  4,  7, 13, 16, 26, 29, 43, 46, 64, 67, 89, 92,118,121,150,
             3,  8, 12, 17, 25, 30, 42, 47, 63, 68, 88, 93,117,122,149,151,
             9, 11, 18, 24, 31, 41, 48, 62, 69, 87, 94,116,123,148,152,177,
            10, 19, 23, 32, 40, 49, 61, 70, 86, 95,115,124,147,153,176,178,
            20, 22, 33, 39, 50, 60, 71, 85, 96,114,125,146,154,175,179,200,
            21, 34, 38, 51, 59, 72, 84, 97,113,126,145,155,174,180,199,201,
            35, 37, 52, 58, 73, 83, 98,112,127,144,156,173,181,198,202,219,
            36, 53, 57, 74, 82, 99,111,128,143,157,172,182,197,203,218,220,
            54, 56, 75, 81,100,110,129,142,158,171,183,196,204,217,221,234,
            55, 76, 80,101,109,130,141,159,170,184,195,205,216,222,233,235,
            77, 79,102,108,131,140,160,169,185,194,206,215,223,232,236,245,
            78,103,107,132,139,161,168,186,193,207,214,224,231,237,244,246,
            104,106,133,138,162,167,187,192,208,213,225,230,238,243,247,252,
            105,134,137,163,166,188,191,209,212,226,229,239,242,248,251,253,
            135,136,164,165,189,190,210,211,227,228,240,241,249,250,254,255
        ]).long() 

    def transform_all(self, x):
        k = random.randrange(0,3)
        if k == 0:
            return self.transform_gb(x)
        elif k == 1:
            return self.transform_gn(x)
        elif k == 2:
            return self.transform_jpeg(x)

    def transform_gb(self, x):
        g_sigma = random.uniform(0.5,0.7)
        transform = T.GaussianBlur(kernel_size=(5, 5), sigma=(g_sigma, g_sigma))
        return transform(x)

    def transform_jpeg(self, input_img):
        # x = x.squeeze()
        B, C, H, W = input_img.shape
        for i in range(B):
            x = T.ToPILImage()(input_img[i,:,:,:])
            x = T.Lambda(self.randomJPEGcompression)(x)
            x = T.ToTensor()(x)
            x = x.cuda()
            input_img[i,:,:,:] = x
        return input_img

    def transform_scb(self, x):
        depth = 3
        x = (x * (2**depth - 1)).int()
        x = (x / (2**depth - 1)).float()
        return x

    def transform_gn(self, x):
        sigma = random.uniform(0.04,0.1)
        x = AddGaussianNoise(0,sigma)(x)
        return x

    def randomJPEGcompression(self, image):
        qf = 10 * random.randrange(1, 10)
        outputIoStream = BytesIO()
        image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
        outputIoStream.seek(0)
        return Image.open(outputIoStream)

    def forward(self,x):
        x = (x * 255).round()

        x_ycbcr = to_ycbcr(x, data_range = 255)
        H, W = x.shape[2], x.shape[3]
        K1 = H//16
        K2 = W//16

        trigger_weight = self.Conv5(x)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv6(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv7(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv8(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv_1x1_3(trigger_weight)
        trigger_weight = trigger_weight.repeat_interleave(16,2).repeat_interleave(16,3)

        x_blocks = blockify(x,16).transpose(1,2)
        N = x_blocks.shape[0]
        L = x_blocks.shape[1]
        x_blocks = x_blocks.contiguous().view(N * L, x_blocks.shape[2], 16, 16)
        x_blocks = self.Conv1(x_blocks)
        x_blocks = self.Maxpool(x_blocks)
        x_blocks = self.Conv2(x_blocks)
        x_blocks = self.Maxpool(x_blocks) # NL * C * 4 * 4
        x_blocks = x_blocks.view(N, L, x_blocks.shape[1], x_blocks.shape[2], x_blocks.shape[3]).transpose(1,2) # N * C * L * 4 * 4
        feature_patch, _ = torch.max(x_blocks, dim = 2)
        feature_patch = feature_patch.view(feature_patch.shape[0], 16 * feature_patch.shape[1], 1, 1) # N * 16C * 1 * 1

        feature_patch = self.Conv_1x1_1(feature_patch)
        feature_patch = feature_patch.view(feature_patch.shape[0], self.img_ch, self.trigger_channel)

        x_ycbcr_blocks = blockify(x_ycbcr,16) - 128
        dct_blocks = dct_2d(x_ycbcr_blocks)
        dct = deblockify(dct_blocks,(H,W))
        trigger_patch = torch.zeros_like(x[:,:,:16,:16], dtype = feature_patch.dtype)
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 256)
        trigger_patch[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]] = feature_patch.detach()
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 16, 16).repeat(1,1,K1,K2)
        dct_poisoned = dct + trigger_patch * trigger_weight.detach()
        dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)
        x_poisoned = x_poisoned.div(255)
        x_poisoned = self.transform_jpeg(x_poisoned)

        x_poisoned_ycbcr = to_ycbcr(x_poisoned, data_range = 255)
        x_poisoned_ycbcr_blocks = blockify(x_poisoned_ycbcr,16) - 128
        dct_x_poisoned_ycbcr_blocks = dct_2d(x_poisoned_ycbcr_blocks)
        dct_x_poisoned_ycbcr = deblockify(dct_x_poisoned_ycbcr_blocks,(H,W))
        delta_dct = (dct_x_poisoned_ycbcr - dct_poisoned.detach())/(trigger_patch * trigger_weight.detach())
        delta_dct_blocks = blockify(delta_dct,16).abs() # N * C * L * 16 * 16
        delta_dct_blocks = delta_dct_blocks.sum(2) # N * C * 16 * 16
        delta_dct_blocks = delta_dct_blocks.view(delta_dct_blocks.shape[0],delta_dct_blocks.shape[1],256)
        feature_patch_importance = delta_dct_blocks[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]]

        # topk_index = torch.topk(feature_patch_importance, self.trigger_channel//4 * 3, dim = 2, largest = True)[1]
        topk_index = torch.topk(feature_patch_importance, self.trigger_channel//8 * 7, dim = 2, largest = True)[1]
        feature_patch.scatter_(2, topk_index, 0)

        trigger_patch = torch.zeros_like(x[:,:,:16,:16], dtype = feature_patch.dtype)
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 256)
        trigger_patch[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]] = feature_patch
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 16, 16).repeat(1,1,K1,K2)

        dct_poisoned = dct + trigger_patch * trigger_weight
        dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)

        x_poisoned = x_poisoned.div(255)
        
        return x_poisoned



class Trigger_Net_v26(nn.Module):
    def __init__(self, img_ch=3, multiplier = 1, trigger_channel = 64):
        super(Trigger_Net_v26,self).__init__()
        
        self.img_ch = img_ch
        self.trigger_channel = trigger_channel
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv2 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))

        self.Conv_1x1_1 = nn.Conv2d(int(16 * 128 * multiplier), img_ch * trigger_channel, kernel_size=1,stride=1,padding=0)
        self.Conv_1x1_2 = nn.Conv2d(int(16 * 128 * multiplier), img_ch * trigger_channel, kernel_size=1,stride=1,padding=0)

        self.Conv5 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv6 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        self.Conv7 = conv_block(ch_in=int(128 * multiplier),ch_out=int(256 * multiplier))
        self.Conv8 = conv_block(ch_in=int(256 * multiplier),ch_out=int(512 * multiplier))
        self.Conv_1x1_3 = nn.Conv2d(int(512 * multiplier),img_ch,kernel_size=1,stride=1,padding=0)

        self.zigzag_indices = torch.tensor([
             0,  1,  5,  6, 14, 15, 27, 28, 44, 45, 65, 66, 90, 91,119,120,
             2,  4,  7, 13, 16, 26, 29, 43, 46, 64, 67, 89, 92,118,121,150,
             3,  8, 12, 17, 25, 30, 42, 47, 63, 68, 88, 93,117,122,149,151,
             9, 11, 18, 24, 31, 41, 48, 62, 69, 87, 94,116,123,148,152,177,
            10, 19, 23, 32, 40, 49, 61, 70, 86, 95,115,124,147,153,176,178,
            20, 22, 33, 39, 50, 60, 71, 85, 96,114,125,146,154,175,179,200,
            21, 34, 38, 51, 59, 72, 84, 97,113,126,145,155,174,180,199,201,
            35, 37, 52, 58, 73, 83, 98,112,127,144,156,173,181,198,202,219,
            36, 53, 57, 74, 82, 99,111,128,143,157,172,182,197,203,218,220,
            54, 56, 75, 81,100,110,129,142,158,171,183,196,204,217,221,234,
            55, 76, 80,101,109,130,141,159,170,184,195,205,216,222,233,235,
            77, 79,102,108,131,140,160,169,185,194,206,215,223,232,236,245,
            78,103,107,132,139,161,168,186,193,207,214,224,231,237,244,246,
            104,106,133,138,162,167,187,192,208,213,225,230,238,243,247,252,
            105,134,137,163,166,188,191,209,212,226,229,239,242,248,251,253,
            135,136,164,165,189,190,210,211,227,228,240,241,249,250,254,255
        ]).long() 

    def transform_all(self, x):
        k = random.randrange(0,3)
        if k == 0:
            return self.transform_gb(x)
        elif k == 1:
            return self.transform_gn(x)
        elif k == 2:
            return self.transform_jpeg(x)

    def transform_gb(self, x):
        g_sigma = random.uniform(0.5,0.7)
        transform = T.GaussianBlur(kernel_size=(5, 5), sigma=(g_sigma, g_sigma))
        return transform(x)

    def transform_jpeg(self, input_img):
        # x = x.squeeze()
        B, C, H, W = input_img.shape
        for i in range(B):
            x = T.ToPILImage()(input_img[i,:,:,:])
            x = T.Lambda(self.randomJPEGcompression)(x)
            x = T.ToTensor()(x)
            x = x.cuda()
            input_img[i,:,:,:] = x
        return input_img

    def transform_scb(self, x):
        depth = 3
        x = (x * (2**depth - 1)).int()
        x = (x / (2**depth - 1)).float()
        return x

    def transform_gn(self, x):
        sigma = random.uniform(0.04,0.1)
        x = AddGaussianNoise(0,sigma)(x)
        return x

    def randomJPEGcompression(self, image):
        qf = 10 * random.randrange(1, 10)
        outputIoStream = BytesIO()
        image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
        outputIoStream.seek(0)
        return Image.open(outputIoStream)

    def forward(self,x):
        x = (x * 255).round()

        x_ycbcr = to_ycbcr(x, data_range = 255)
        H, W = x.shape[2], x.shape[3]
        K1 = H//16
        K2 = W//16

        trigger_weight = self.Conv5(x)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv6(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv7(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv8(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv_1x1_3(trigger_weight)
        trigger_weight = trigger_weight.repeat_interleave(16,2).repeat_interleave(16,3)

        x_blocks = blockify(x,16).transpose(1,2)
        N = x_blocks.shape[0]
        L = x_blocks.shape[1]
        x_blocks = x_blocks.contiguous().view(N * L, x_blocks.shape[2], 16, 16)
        x_blocks = self.Conv1(x_blocks)
        x_blocks = self.Maxpool(x_blocks)
        x_blocks = self.Conv2(x_blocks)
        x_blocks = self.Maxpool(x_blocks) # NL * C * 4 * 4
        x_blocks = x_blocks.view(N, L, x_blocks.shape[1], x_blocks.shape[2], x_blocks.shape[3]).transpose(1,2) # N * C * L * 4 * 4
        feature_patch, _ = torch.max(x_blocks, dim = 2)
        feature_patch = feature_patch.view(feature_patch.shape[0], 16 * feature_patch.shape[1], 1, 1) # N * 16C * 1 * 1

        feature_patch = self.Conv_1x1_1(feature_patch)
        feature_patch = feature_patch.view(feature_patch.shape[0], self.img_ch, self.trigger_channel)

        x_ycbcr_blocks = blockify(x_ycbcr,16) - 128
        dct_blocks = dct_2d(x_ycbcr_blocks)
        dct = deblockify(dct_blocks,(H,W))
        trigger_patch = torch.zeros_like(x[:,:,:16,:16], dtype = feature_patch.dtype)
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 256)
        trigger_patch[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]] = feature_patch.detach()
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 16, 16).repeat(1,1,K1,K2)
        dct_poisoned = dct + trigger_patch * trigger_weight.detach()
        dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)
        x_poisoned = x_poisoned.div(255)
        x_poisoned = self.transform_jpeg(x_poisoned)

        x_poisoned_ycbcr = to_ycbcr(x_poisoned, data_range = 255)
        x_poisoned_ycbcr_blocks = blockify(x_poisoned_ycbcr,16) - 128
        dct_x_poisoned_ycbcr_blocks = dct_2d(x_poisoned_ycbcr_blocks)
        dct_x_poisoned_ycbcr = deblockify(dct_x_poisoned_ycbcr_blocks,(H,W))
        delta_dct = (dct_x_poisoned_ycbcr - dct_poisoned.detach())/(trigger_patch * trigger_weight.detach())
        delta_dct_blocks = blockify(delta_dct,16).abs() # N * C * L * 16 * 16
        delta_dct_blocks = delta_dct_blocks.sum(2) # N * C * 16 * 16
        delta_dct_blocks = delta_dct_blocks.view(delta_dct_blocks.shape[0],delta_dct_blocks.shape[1],256)
        feature_patch_importance = delta_dct_blocks[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]]

        # topk_index = torch.topk(feature_patch_importance, self.trigger_channel//4 * 3, dim = 2, largest = True)[1]
        topk_index = torch.topk(feature_patch_importance, self.trigger_channel//2, dim = 2, largest = True)[1]
        feature_patch.scatter_(2, topk_index, 0)

        trigger_patch = torch.zeros_like(x[:,:,:16,:16], dtype = feature_patch.dtype)
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 256)
        trigger_patch[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]] = feature_patch
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 16, 16).repeat(1,1,K1,K2)

        dct_poisoned = dct + trigger_patch * trigger_weight
        dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)

        x_poisoned = x_poisoned.div(255)
        
        return x_poisoned


class Trigger_Net_v27(nn.Module):
    def __init__(self, img_ch=3, multiplier = 1, trigger_channel = 64):
        super(Trigger_Net_v27,self).__init__()
        
        self.img_ch = img_ch
        self.trigger_channel = trigger_channel
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv2 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))

        self.Conv_1x1_1 = nn.Conv2d(int(16 * 128 * multiplier), img_ch * trigger_channel, kernel_size=1,stride=1,padding=0)
        self.Conv_1x1_2 = nn.Conv2d(int(16 * 128 * multiplier), img_ch * trigger_channel, kernel_size=1,stride=1,padding=0)

        self.Conv5 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv6 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        self.Conv7 = conv_block(ch_in=int(128 * multiplier),ch_out=int(256 * multiplier))
        self.Conv8 = conv_block(ch_in=int(256 * multiplier),ch_out=int(512 * multiplier))
        self.Conv_1x1_3 = nn.Conv2d(int(512 * multiplier),img_ch,kernel_size=1,stride=1,padding=0)

        self.zigzag_indices = torch.tensor([
             0,  1,  5,  6, 14, 15, 27, 28, 44, 45, 65, 66, 90, 91,119,120,
             2,  4,  7, 13, 16, 26, 29, 43, 46, 64, 67, 89, 92,118,121,150,
             3,  8, 12, 17, 25, 30, 42, 47, 63, 68, 88, 93,117,122,149,151,
             9, 11, 18, 24, 31, 41, 48, 62, 69, 87, 94,116,123,148,152,177,
            10, 19, 23, 32, 40, 49, 61, 70, 86, 95,115,124,147,153,176,178,
            20, 22, 33, 39, 50, 60, 71, 85, 96,114,125,146,154,175,179,200,
            21, 34, 38, 51, 59, 72, 84, 97,113,126,145,155,174,180,199,201,
            35, 37, 52, 58, 73, 83, 98,112,127,144,156,173,181,198,202,219,
            36, 53, 57, 74, 82, 99,111,128,143,157,172,182,197,203,218,220,
            54, 56, 75, 81,100,110,129,142,158,171,183,196,204,217,221,234,
            55, 76, 80,101,109,130,141,159,170,184,195,205,216,222,233,235,
            77, 79,102,108,131,140,160,169,185,194,206,215,223,232,236,245,
            78,103,107,132,139,161,168,186,193,207,214,224,231,237,244,246,
            104,106,133,138,162,167,187,192,208,213,225,230,238,243,247,252,
            105,134,137,163,166,188,191,209,212,226,229,239,242,248,251,253,
            135,136,164,165,189,190,210,211,227,228,240,241,249,250,254,255
        ]).long() 

    def transform_all(self, x):
        k = random.randrange(0,3)
        if k == 0:
            return self.transform_gb(x)
        elif k == 1:
            return self.transform_gn(x)
        elif k == 2:
            return self.transform_jpeg(x)

    def transform_gb(self, x):
        g_sigma = random.uniform(0.5,0.7)
        transform = T.GaussianBlur(kernel_size=(5, 5), sigma=(g_sigma, g_sigma))
        return transform(x)

    def transform_jpeg(self, input_img):
        # x = x.squeeze()
        B, C, H, W = input_img.shape
        for i in range(B):
            x = T.ToPILImage()(input_img[i,:,:,:])
            x = T.Lambda(self.randomJPEGcompression)(x)
            x = T.ToTensor()(x)
            x = x.cuda()
            input_img[i,:,:,:] = x
        return input_img

    def transform_scb(self, x):
        depth = 3
        x = (x * (2**depth - 1)).int()
        x = (x / (2**depth - 1)).float()
        return x

    def transform_gn(self, x):
        sigma = random.uniform(0.04,0.1)
        x = AddGaussianNoise(0,sigma)(x)
        return x

    def randomJPEGcompression(self, image):
        # qf = 10 * random.randrange(1, 10)
        qf = 50
        outputIoStream = BytesIO()
        image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
        outputIoStream.seek(0)
        return Image.open(outputIoStream)

    def forward(self,x):
        x = (x * 255).round()

        x_ycbcr = to_ycbcr(x, data_range = 255)
        H, W = x.shape[2], x.shape[3]
        K1 = H//16
        K2 = W//16

        trigger_weight = self.Conv5(x)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv6(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv7(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv8(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv_1x1_3(trigger_weight)
        trigger_weight = trigger_weight.repeat_interleave(16,2).repeat_interleave(16,3)

        x_blocks = blockify(x,16).transpose(1,2)
        N = x_blocks.shape[0]
        L = x_blocks.shape[1]
        x_blocks = x_blocks.contiguous().view(N * L, x_blocks.shape[2], 16, 16)
        x_blocks = self.Conv1(x_blocks)
        x_blocks = self.Maxpool(x_blocks)
        x_blocks = self.Conv2(x_blocks)
        x_blocks = self.Maxpool(x_blocks) # NL * C * 4 * 4
        x_blocks = x_blocks.view(N, L, x_blocks.shape[1], x_blocks.shape[2], x_blocks.shape[3]).transpose(1,2) # N * C * L * 4 * 4
        feature_patch, _ = torch.max(x_blocks, dim = 2)
        feature_patch = feature_patch.view(feature_patch.shape[0], 16 * feature_patch.shape[1], 1, 1) # N * 16C * 1 * 1

        feature_patch = self.Conv_1x1_1(feature_patch)
        feature_patch = feature_patch.view(feature_patch.shape[0], self.img_ch, self.trigger_channel)

        x_ycbcr_blocks = blockify(x_ycbcr,16) - 128
        dct_blocks = dct_2d(x_ycbcr_blocks)
        dct = deblockify(dct_blocks,(H,W))
        trigger_patch = torch.zeros_like(x[:,:,:16,:16], dtype = feature_patch.dtype)
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 256)
        trigger_patch[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]] = feature_patch.detach()
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 16, 16).repeat(1,1,K1,K2)
        dct_poisoned = dct + trigger_patch * trigger_weight.detach()
        dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)
        x_poisoned = x_poisoned.div(255)
        x_poisoned = self.transform_jpeg(x_poisoned)

        x_poisoned_ycbcr = to_ycbcr(x_poisoned, data_range = 255)
        x_poisoned_ycbcr_blocks = blockify(x_poisoned_ycbcr,16) - 128
        dct_x_poisoned_ycbcr_blocks = dct_2d(x_poisoned_ycbcr_blocks)
        dct_x_poisoned_ycbcr = deblockify(dct_x_poisoned_ycbcr_blocks,(H,W))
        delta_dct = (dct_x_poisoned_ycbcr - dct_poisoned.detach())/(trigger_patch * trigger_weight.detach())
        delta_dct_blocks = blockify(delta_dct,16).abs() # N * C * L * 16 * 16
        delta_dct_blocks = delta_dct_blocks.sum(2) # N * C * 16 * 16
        delta_dct_blocks = delta_dct_blocks.view(delta_dct_blocks.shape[0],delta_dct_blocks.shape[1],256)
        feature_patch_importance = delta_dct_blocks[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]]

        topk_index = torch.topk(feature_patch_importance, self.trigger_channel//4 * 3, dim = 2, largest = True)[1]
        # topk_index = torch.topk(feature_patch_importance, self.trigger_channel//2, dim = 2, largest = True)[1]
        feature_patch.scatter_(2, topk_index, 0)

        trigger_patch = torch.zeros_like(x[:,:,:16,:16], dtype = feature_patch.dtype)
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 256)
        trigger_patch[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]] = feature_patch
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 16, 16).repeat(1,1,K1,K2)

        dct_poisoned = dct + trigger_patch * trigger_weight
        dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)

        x_poisoned = x_poisoned.div(255)
        
        return x_poisoned


class Trigger_Net_v28(nn.Module):
    def __init__(self, img_ch=3, multiplier = 1, trigger_channel = 64):
        super(Trigger_Net_v28,self).__init__()
        
        self.img_ch = img_ch
        self.trigger_channel = trigger_channel
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv2 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))

        self.Conv_1x1_1 = nn.Conv2d(int(16 * 128 * multiplier), img_ch * trigger_channel, kernel_size=1,stride=1,padding=0)
        self.Conv_1x1_2 = nn.Conv2d(int(16 * 128 * multiplier), img_ch * trigger_channel, kernel_size=1,stride=1,padding=0)

        self.Conv5 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv6 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        self.Conv7 = conv_block(ch_in=int(128 * multiplier),ch_out=int(256 * multiplier))
        self.Conv8 = conv_block(ch_in=int(256 * multiplier),ch_out=int(512 * multiplier))
        self.Conv_1x1_3 = nn.Conv2d(int(512 * multiplier),img_ch,kernel_size=1,stride=1,padding=0)

        self.zigzag_indices = torch.tensor([
             0,  1,  5,  6, 14, 15, 27, 28, 44, 45, 65, 66, 90, 91,119,120,
             2,  4,  7, 13, 16, 26, 29, 43, 46, 64, 67, 89, 92,118,121,150,
             3,  8, 12, 17, 25, 30, 42, 47, 63, 68, 88, 93,117,122,149,151,
             9, 11, 18, 24, 31, 41, 48, 62, 69, 87, 94,116,123,148,152,177,
            10, 19, 23, 32, 40, 49, 61, 70, 86, 95,115,124,147,153,176,178,
            20, 22, 33, 39, 50, 60, 71, 85, 96,114,125,146,154,175,179,200,
            21, 34, 38, 51, 59, 72, 84, 97,113,126,145,155,174,180,199,201,
            35, 37, 52, 58, 73, 83, 98,112,127,144,156,173,181,198,202,219,
            36, 53, 57, 74, 82, 99,111,128,143,157,172,182,197,203,218,220,
            54, 56, 75, 81,100,110,129,142,158,171,183,196,204,217,221,234,
            55, 76, 80,101,109,130,141,159,170,184,195,205,216,222,233,235,
            77, 79,102,108,131,140,160,169,185,194,206,215,223,232,236,245,
            78,103,107,132,139,161,168,186,193,207,214,224,231,237,244,246,
            104,106,133,138,162,167,187,192,208,213,225,230,238,243,247,252,
            105,134,137,163,166,188,191,209,212,226,229,239,242,248,251,253,
            135,136,164,165,189,190,210,211,227,228,240,241,249,250,254,255
        ]).long() 

    def transform_all(self, x):
        k = random.randrange(0,3)
        if k == 0:
            return self.transform_gb(x)
        elif k == 1:
            return self.transform_gn(x)
        elif k == 2:
            return self.transform_jpeg(x)

    def transform_gb(self, x):
        g_sigma = random.uniform(0.4,1.0)
        transform = T.GaussianBlur(kernel_size=(5, 5), sigma=(g_sigma, g_sigma))
        return transform(x)

    def transform_jpeg(self, input_img):
        # x = x.squeeze()
        B, C, H, W = input_img.shape
        for i in range(B):
            x = T.ToPILImage()(input_img[i,:,:,:])
            x = T.Lambda(self.randomJPEGcompression)(x)
            x = T.ToTensor()(x)
            x = x.cuda()
            input_img[i,:,:,:] = x
        return input_img

    def transform_scb(self, x):
        depth = 3
        x = (x * (2**depth - 1)).int()
        x = (x / (2**depth - 1)).float()
        return x

    def transform_gn(self, x):
        sigma = random.uniform(0.04,0.1)
        x = AddGaussianNoise(0,sigma)(x)
        return x

    def randomJPEGcompression(self, image):
        qf = 10 * random.randrange(1, 10)
        outputIoStream = BytesIO()
        image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
        outputIoStream.seek(0)
        return Image.open(outputIoStream)

    def forward(self,x):
        x = (x * 255).round()

        x_ycbcr = to_ycbcr(x, data_range = 255)
        H, W = x.shape[2], x.shape[3]
        K1 = H//16
        K2 = W//16

        trigger_weight = self.Conv5(x)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv6(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv7(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv8(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv_1x1_3(trigger_weight)
        trigger_weight = trigger_weight.repeat_interleave(16,2).repeat_interleave(16,3)

        x_blocks = blockify(x,16).transpose(1,2)
        N = x_blocks.shape[0]
        L = x_blocks.shape[1]
        x_blocks = x_blocks.contiguous().view(N * L, x_blocks.shape[2], 16, 16)
        x_blocks = self.Conv1(x_blocks)
        x_blocks = self.Maxpool(x_blocks)
        x_blocks = self.Conv2(x_blocks)
        x_blocks = self.Maxpool(x_blocks) # NL * C * 4 * 4
        x_blocks = x_blocks.view(N, L, x_blocks.shape[1], x_blocks.shape[2], x_blocks.shape[3]).transpose(1,2) # N * C * L * 4 * 4
        feature_patch, _ = torch.max(x_blocks, dim = 2)
        feature_patch = feature_patch.view(feature_patch.shape[0], 16 * feature_patch.shape[1], 1, 1) # N * 16C * 1 * 1

        feature_patch = self.Conv_1x1_1(feature_patch)
        feature_patch = feature_patch.view(feature_patch.shape[0], self.img_ch, self.trigger_channel)

        x_ycbcr_blocks = blockify(x_ycbcr,16) - 128
        dct_blocks = dct_2d(x_ycbcr_blocks)
        dct = deblockify(dct_blocks,(H,W))
        trigger_patch = torch.zeros_like(x[:,:,:16,:16], dtype = feature_patch.dtype)
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 256)
        trigger_patch[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]] = feature_patch.detach()
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 16, 16).repeat(1,1,K1,K2)
        dct_poisoned = dct + trigger_patch * trigger_weight.detach()
        dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)
        x_poisoned = x_poisoned.div(255)
        x_poisoned = self.transform_gb(x_poisoned)

        x_poisoned_ycbcr = to_ycbcr(x_poisoned, data_range = 255)
        x_poisoned_ycbcr_blocks = blockify(x_poisoned_ycbcr,16) - 128
        dct_x_poisoned_ycbcr_blocks = dct_2d(x_poisoned_ycbcr_blocks)
        dct_x_poisoned_ycbcr = deblockify(dct_x_poisoned_ycbcr_blocks,(H,W))
        delta_dct = (dct_x_poisoned_ycbcr - dct_poisoned.detach())/(trigger_patch * trigger_weight.detach())
        delta_dct_blocks = blockify(delta_dct,16).abs() # N * C * L * 16 * 16
        delta_dct_blocks = delta_dct_blocks.sum(2) # N * C * 16 * 16
        delta_dct_blocks = delta_dct_blocks.view(delta_dct_blocks.shape[0],delta_dct_blocks.shape[1],256)
        feature_patch_importance = delta_dct_blocks[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]]

        topk_index = torch.topk(feature_patch_importance, self.trigger_channel//4 * 3, dim = 2, largest = True)[1]
        feature_patch.scatter_(2, topk_index, 0)


        trigger_patch = torch.zeros_like(x[:,:,:16,:16], dtype = feature_patch.dtype)
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 256)
        trigger_patch[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]] = feature_patch
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 16, 16).repeat(1,1,K1,K2)

        dct_poisoned = dct + trigger_patch * trigger_weight
        dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)

        x_poisoned = x_poisoned.div(255)
        
        return x_poisoned


class Trigger_Net_v29(nn.Module):
    def __init__(self, img_ch=3, multiplier = 1, trigger_channel = 64):
        super(Trigger_Net_v29,self).__init__()
        
        self.img_ch = img_ch
        self.trigger_channel = trigger_channel
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv2 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))

        self.Conv_1x1_1 = nn.Conv2d(int(16 * 128 * multiplier), img_ch * trigger_channel, kernel_size=1,stride=1,padding=0)
        self.Conv_1x1_2 = nn.Conv2d(int(16 * 128 * multiplier), img_ch * trigger_channel, kernel_size=1,stride=1,padding=0)

        self.Conv5 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv6 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        self.Conv7 = conv_block(ch_in=int(128 * multiplier),ch_out=int(256 * multiplier))
        self.Conv8 = conv_block(ch_in=int(256 * multiplier),ch_out=int(512 * multiplier))
        self.Conv_1x1_3 = nn.Conv2d(int(512 * multiplier),img_ch,kernel_size=1,stride=1,padding=0)

        self.zigzag_indices = torch.tensor([
             0,  1,  5,  6, 14, 15, 27, 28, 44, 45, 65, 66, 90, 91,119,120,
             2,  4,  7, 13, 16, 26, 29, 43, 46, 64, 67, 89, 92,118,121,150,
             3,  8, 12, 17, 25, 30, 42, 47, 63, 68, 88, 93,117,122,149,151,
             9, 11, 18, 24, 31, 41, 48, 62, 69, 87, 94,116,123,148,152,177,
            10, 19, 23, 32, 40, 49, 61, 70, 86, 95,115,124,147,153,176,178,
            20, 22, 33, 39, 50, 60, 71, 85, 96,114,125,146,154,175,179,200,
            21, 34, 38, 51, 59, 72, 84, 97,113,126,145,155,174,180,199,201,
            35, 37, 52, 58, 73, 83, 98,112,127,144,156,173,181,198,202,219,
            36, 53, 57, 74, 82, 99,111,128,143,157,172,182,197,203,218,220,
            54, 56, 75, 81,100,110,129,142,158,171,183,196,204,217,221,234,
            55, 76, 80,101,109,130,141,159,170,184,195,205,216,222,233,235,
            77, 79,102,108,131,140,160,169,185,194,206,215,223,232,236,245,
            78,103,107,132,139,161,168,186,193,207,214,224,231,237,244,246,
            104,106,133,138,162,167,187,192,208,213,225,230,238,243,247,252,
            105,134,137,163,166,188,191,209,212,226,229,239,242,248,251,253,
            135,136,164,165,189,190,210,211,227,228,240,241,249,250,254,255
        ]).long() 

    def transform_all(self, x):
        k = random.randrange(0,3)
        if k == 0:
            return self.transform_gb(x)
        elif k == 1:
            return self.transform_gn(x)
        elif k == 2:
            return self.transform_jpeg(x)

    def transform_gb(self, x):
        g_sigma = random.uniform(0.4,1.0)
        transform = T.GaussianBlur(kernel_size=(5, 5), sigma=(g_sigma, g_sigma))
        return transform(x)

    def transform_jpeg(self, input_img):
        # x = x.squeeze()
        B, C, H, W = input_img.shape
        for i in range(B):
            x = T.ToPILImage()(input_img[i,:,:,:])
            x = T.Lambda(self.randomJPEGcompression)(x)
            x = T.ToTensor()(x)
            x = x.cuda()
            input_img[i,:,:,:] = x
        return input_img

    def transform_scb(self, x):
        depth = 3
        x = (x * (2**depth - 1)).int()
        x = (x / (2**depth - 1)).float()
        return x

    def transform_gn(self, x):
        sigma = random.uniform(0.04,0.1)
        x = AddGaussianNoise(0,sigma)(x)
        return x

    def randomJPEGcompression(self, image):
        qf = 10 * random.randrange(1, 10)
        outputIoStream = BytesIO()
        image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
        outputIoStream.seek(0)
        return Image.open(outputIoStream)

    def forward(self,x):
        x = (x * 255).round()

        x_ycbcr = to_ycbcr(x, data_range = 255)
        H, W = x.shape[2], x.shape[3]
        K1 = H//16
        K2 = W//16

        trigger_weight = self.Conv5(x)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv6(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv7(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv8(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv_1x1_3(trigger_weight)
        trigger_weight = trigger_weight.repeat_interleave(16,2).repeat_interleave(16,3)

        x_blocks = blockify(x,16).transpose(1,2)
        N = x_blocks.shape[0]
        L = x_blocks.shape[1]
        x_blocks = x_blocks.contiguous().view(N * L, x_blocks.shape[2], 16, 16)
        x_blocks = self.Conv1(x_blocks)
        x_blocks = self.Maxpool(x_blocks)
        x_blocks = self.Conv2(x_blocks)
        x_blocks = self.Maxpool(x_blocks) # NL * C * 4 * 4
        x_blocks = x_blocks.view(N, L, x_blocks.shape[1], x_blocks.shape[2], x_blocks.shape[3]).transpose(1,2) # N * C * L * 4 * 4
        feature_patch, _ = torch.max(x_blocks, dim = 2)
        feature_patch = feature_patch.view(feature_patch.shape[0], 16 * feature_patch.shape[1], 1, 1) # N * 16C * 1 * 1

        feature_patch = self.Conv_1x1_1(feature_patch)
        feature_patch = feature_patch.view(feature_patch.shape[0], self.img_ch, self.trigger_channel)

        x_ycbcr_blocks = blockify(x_ycbcr,16) - 128
        dct_blocks = dct_2d(x_ycbcr_blocks)
        dct = deblockify(dct_blocks,(H,W))
        trigger_patch = torch.zeros_like(x[:,:,:16,:16], dtype = feature_patch.dtype)
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 256)
        trigger_patch[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]] = feature_patch.detach()
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 16, 16).repeat(1,1,K1,K2)
        dct_poisoned = dct + trigger_patch * trigger_weight.detach()
        dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)
        x_poisoned_npp = x_poisoned.div(255)

        x_poisoned = self.transform_gb(x_poisoned_npp)

        x_poisoned_ycbcr = to_ycbcr(x_poisoned, data_range = 255)
        x_poisoned_ycbcr_blocks = blockify(x_poisoned_ycbcr,16) - 128
        dct_x_poisoned_ycbcr_blocks = dct_2d(x_poisoned_ycbcr_blocks)
        dct_x_poisoned_ycbcr = deblockify(dct_x_poisoned_ycbcr_blocks,(H,W))
        delta_dct = (dct_x_poisoned_ycbcr - dct_poisoned.detach())/(trigger_patch * trigger_weight.detach())
        delta_dct_blocks = blockify(delta_dct,16).abs() # N * C * L * 16 * 16
        delta_dct_blocks = delta_dct_blocks.sum(2) # N * C * 16 * 16
        delta_dct_blocks = delta_dct_blocks.view(delta_dct_blocks.shape[0],delta_dct_blocks.shape[1],256)
        feature_patch_importance_gb = delta_dct_blocks[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]]

        x_poisoned = self.transform_gn(x_poisoned_npp)

        x_poisoned_ycbcr = to_ycbcr(x_poisoned, data_range = 255)
        x_poisoned_ycbcr_blocks = blockify(x_poisoned_ycbcr,16) - 128
        dct_x_poisoned_ycbcr_blocks = dct_2d(x_poisoned_ycbcr_blocks)
        dct_x_poisoned_ycbcr = deblockify(dct_x_poisoned_ycbcr_blocks,(H,W))
        delta_dct = (dct_x_poisoned_ycbcr - dct_poisoned.detach())/(trigger_patch * trigger_weight.detach())
        delta_dct_blocks = blockify(delta_dct,16).abs() # N * C * L * 16 * 16
        delta_dct_blocks = delta_dct_blocks.sum(2) # N * C * 16 * 16
        delta_dct_blocks = delta_dct_blocks.view(delta_dct_blocks.shape[0],delta_dct_blocks.shape[1],256)
        feature_patch_importance_gn = delta_dct_blocks[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]]

        feature_patch_importance = feature_patch_importance_gb * feature_patch_importance_gn
        topk_index = torch.topk(feature_patch_importance, self.trigger_channel//4 * 3, dim = 2, largest = True)[1]
        feature_patch.scatter_(2, topk_index, 0)


        trigger_patch = torch.zeros_like(x[:,:,:16,:16], dtype = feature_patch.dtype)
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 256)
        trigger_patch[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]] = feature_patch
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 16, 16).repeat(1,1,K1,K2)

        dct_poisoned = dct + trigger_patch * trigger_weight
        dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)

        x_poisoned = x_poisoned.div(255)
        
        return x_poisoned


class Trigger_Net_v30(nn.Module):
    def __init__(self, img_ch=3, multiplier = 1, trigger_channel = 64):
        super(Trigger_Net_v30,self).__init__()
        
        self.img_ch = img_ch
        self.trigger_channel = trigger_channel
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv2 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))

        self.Conv_1x1_1 = nn.Conv2d(int(16 * 128 * multiplier), img_ch * trigger_channel, kernel_size=1,stride=1,padding=0)
        self.Conv_1x1_2 = nn.Conv2d(int(16 * 128 * multiplier), img_ch * trigger_channel, kernel_size=1,stride=1,padding=0)

        self.Conv5 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv6 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        self.Conv7 = conv_block(ch_in=int(128 * multiplier),ch_out=int(256 * multiplier))
        self.Conv8 = conv_block(ch_in=int(256 * multiplier),ch_out=int(512 * multiplier))
        self.Conv_1x1_3 = nn.Conv2d(int(512 * multiplier),img_ch,kernel_size=1,stride=1,padding=0)

        self.zigzag_indices = torch.tensor([
             0,  1,  5,  6, 14, 15, 27, 28, 44, 45, 65, 66, 90, 91,119,120,
             2,  4,  7, 13, 16, 26, 29, 43, 46, 64, 67, 89, 92,118,121,150,
             3,  8, 12, 17, 25, 30, 42, 47, 63, 68, 88, 93,117,122,149,151,
             9, 11, 18, 24, 31, 41, 48, 62, 69, 87, 94,116,123,148,152,177,
            10, 19, 23, 32, 40, 49, 61, 70, 86, 95,115,124,147,153,176,178,
            20, 22, 33, 39, 50, 60, 71, 85, 96,114,125,146,154,175,179,200,
            21, 34, 38, 51, 59, 72, 84, 97,113,126,145,155,174,180,199,201,
            35, 37, 52, 58, 73, 83, 98,112,127,144,156,173,181,198,202,219,
            36, 53, 57, 74, 82, 99,111,128,143,157,172,182,197,203,218,220,
            54, 56, 75, 81,100,110,129,142,158,171,183,196,204,217,221,234,
            55, 76, 80,101,109,130,141,159,170,184,195,205,216,222,233,235,
            77, 79,102,108,131,140,160,169,185,194,206,215,223,232,236,245,
            78,103,107,132,139,161,168,186,193,207,214,224,231,237,244,246,
            104,106,133,138,162,167,187,192,208,213,225,230,238,243,247,252,
            105,134,137,163,166,188,191,209,212,226,229,239,242,248,251,253,
            135,136,164,165,189,190,210,211,227,228,240,241,249,250,254,255
        ]).long() 

    def transform_all(self, x):
        k = random.randrange(0,3)
        if k == 0:
            return self.transform_gb(x)
        elif k == 1:
            return self.transform_gn(x)
        elif k == 2:
            return self.transform_jpeg(x)

    def transform_gb(self, x):
        g_sigma = random.uniform(0.4,1.0)
        transform = T.GaussianBlur(kernel_size=(5, 5), sigma=(g_sigma, g_sigma))
        return transform(x)

    def transform_jpeg(self, input_img):
        # x = x.squeeze()
        B, C, H, W = input_img.shape
        for i in range(B):
            x = T.ToPILImage()(input_img[i,:,:,:])
            x = T.Lambda(self.randomJPEGcompression)(x)
            x = T.ToTensor()(x)
            x = x.cuda()
            input_img[i,:,:,:] = x
        return input_img

    def transform_scb(self, x):
        depth = 3
        x = (x * (2**depth - 1)).int()
        x = (x / (2**depth - 1)).float()
        return x

    def transform_gn(self, x):
        sigma = random.uniform(0.04,0.1)
        x = AddGaussianNoise(0,sigma)(x)
        return x

    def randomJPEGcompression(self, image):
        qf = 10 * random.randrange(1, 10)
        outputIoStream = BytesIO()
        image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
        outputIoStream.seek(0)
        return Image.open(outputIoStream)

    def forward(self,x):
        x = (x * 255).round()

        x_ycbcr = to_ycbcr(x, data_range = 255)
        H, W = x.shape[2], x.shape[3]
        K1 = H//16
        K2 = W//16

        trigger_weight = self.Conv5(x)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv6(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv7(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv8(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv_1x1_3(trigger_weight)
        trigger_weight = trigger_weight.repeat_interleave(16,2).repeat_interleave(16,3)

        x_blocks = blockify(x,16).transpose(1,2)
        N = x_blocks.shape[0]
        L = x_blocks.shape[1]
        x_blocks = x_blocks.contiguous().view(N * L, x_blocks.shape[2], 16, 16)
        x_blocks = self.Conv1(x_blocks)
        x_blocks = self.Maxpool(x_blocks)
        x_blocks = self.Conv2(x_blocks)
        x_blocks = self.Maxpool(x_blocks) # NL * C * 4 * 4
        x_blocks = x_blocks.view(N, L, x_blocks.shape[1], x_blocks.shape[2], x_blocks.shape[3]).transpose(1,2) # N * C * L * 4 * 4
        feature_patch, _ = torch.max(x_blocks, dim = 2)
        feature_patch = feature_patch.view(feature_patch.shape[0], 16 * feature_patch.shape[1], 1, 1) # N * 16C * 1 * 1

        feature_patch = self.Conv_1x1_1(feature_patch)
        feature_patch = feature_patch.view(feature_patch.shape[0], self.img_ch, self.trigger_channel)

        x_ycbcr_blocks = blockify(x_ycbcr,16) - 128
        dct_blocks = dct_2d(x_ycbcr_blocks)
        dct = deblockify(dct_blocks,(H,W))
        trigger_patch = torch.zeros_like(x[:,:,:16,:16], dtype = feature_patch.dtype)
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 256)
        trigger_patch[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]] = feature_patch.detach()
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 16, 16).repeat(1,1,K1,K2)
        dct_poisoned = dct + trigger_patch * trigger_weight.detach()
        dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)
        x_poisoned_npp = x_poisoned.div(255)

        x_poisoned = self.transform_gb(x_poisoned_npp)

        x_poisoned_ycbcr = to_ycbcr(x_poisoned, data_range = 255)
        x_poisoned_ycbcr_blocks = blockify(x_poisoned_ycbcr,16) - 128
        dct_x_poisoned_ycbcr_blocks = dct_2d(x_poisoned_ycbcr_blocks)
        dct_x_poisoned_ycbcr = deblockify(dct_x_poisoned_ycbcr_blocks,(H,W))
        delta_dct = (dct_x_poisoned_ycbcr - dct_poisoned.detach())/(trigger_patch * trigger_weight.detach())
        delta_dct_blocks = blockify(delta_dct,16).abs() # N * C * L * 16 * 16
        delta_dct_blocks = delta_dct_blocks.sum(2) # N * C * 16 * 16
        delta_dct_blocks = delta_dct_blocks.view(delta_dct_blocks.shape[0],delta_dct_blocks.shape[1],256)
        feature_patch_importance_gb = delta_dct_blocks[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]]

        x_poisoned = self.transform_gn(x_poisoned_npp)

        x_poisoned_ycbcr = to_ycbcr(x_poisoned, data_range = 255)
        x_poisoned_ycbcr_blocks = blockify(x_poisoned_ycbcr,16) - 128
        dct_x_poisoned_ycbcr_blocks = dct_2d(x_poisoned_ycbcr_blocks)
        dct_x_poisoned_ycbcr = deblockify(dct_x_poisoned_ycbcr_blocks,(H,W))
        delta_dct = (dct_x_poisoned_ycbcr - dct_poisoned.detach())/(trigger_patch * trigger_weight.detach())
        delta_dct_blocks = blockify(delta_dct,16).abs() # N * C * L * 16 * 16
        delta_dct_blocks = delta_dct_blocks.sum(2) # N * C * 16 * 16
        delta_dct_blocks = delta_dct_blocks.view(delta_dct_blocks.shape[0],delta_dct_blocks.shape[1],256)
        feature_patch_importance_gn = delta_dct_blocks[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]]

        feature_patch_importance = feature_patch_importance_gb * feature_patch_importance_gn
        topk_index = torch.topk(feature_patch_importance, self.trigger_channel//4 * 3, dim = 2, largest = True)[1]
        feature_patch.scatter_(2, topk_index, 0)

        topk_index = torch.topk(feature_patch_importance, self.trigger_channel//4, dim = 2, largest = False)[1]
        src = torch.zeros_like(topk_index).float()
        for k in range(self.trigger_channel//4):
            # multiple_coefficient = (self.trigger_channel//4 - k) / (self.trigger_channel//4) * 2 + 1
            # multiple_coefficient = (self.trigger_channel//4 - k) / (self.trigger_channel//4) * 1 + 0.5
            # multiple_coefficient = ((self.trigger_channel//4 - k) / (self.trigger_channel//4) * 2) ** 0.5
            multiple_coefficient = ((self.trigger_channel//4 - k) / (self.trigger_channel//4) * 1 + 0.5)**0.5
            # multiple_coefficient = ((self.trigger_channel//4 - k) / (self.trigger_channel//4) * 1.5 + 0.25)**0.5
            # multiple_coefficient = 0.8 * ((self.trigger_channel//4 - k) / (self.trigger_channel//4) * 1.5 + 0.25)**0.5
            # multiple_coefficient = ((self.trigger_channel//4 - 1 - k) / (self.trigger_channel//4 - 1) * 1 + 0.5)**0.5
            # multiple_coefficient = 1
            src[:,:,k] = multiple_coefficient
        feature_patch.scatter_(2, topk_index, src, reduce = 'multiply')

        trigger_patch = torch.zeros_like(x[:,:,:16,:16], dtype = feature_patch.dtype)
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 256)
        trigger_patch[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]] = feature_patch
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 16, 16).repeat(1,1,K1,K2)

        dct_poisoned = dct + trigger_patch * trigger_weight
        dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)

        x_poisoned = x_poisoned.div(255)
        
        return x_poisoned



class Trigger_Net_v31(nn.Module):
    def __init__(self, img_ch=3, multiplier = 1, trigger_channel = 64):
        super(Trigger_Net_v31,self).__init__()
        
        self.img_ch = img_ch
        self.trigger_channel = trigger_channel
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        # self.Maxpool2 = nn.MaxPool2d(kernel_size=4,stride=4)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv2 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        # self.Conv3 = conv_block(ch_in=int(128 * multiplier),ch_out=int(256 * multiplier))
        # self.Conv4 = conv_block(ch_in=int(256 * multiplier),ch_out=int(512 * multiplier))
        self.Conv_1x1_1 = nn.Conv2d(int(16 * 128 * multiplier), img_ch * trigger_channel, kernel_size=1,stride=1,padding=0)
        self.Conv_1x1_2 = nn.Conv2d(int(16 * 128 * multiplier), img_ch * trigger_channel, kernel_size=1,stride=1,padding=0)

        self.Conv5 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv6 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        self.Conv7 = conv_block(ch_in=int(128 * multiplier),ch_out=int(256 * multiplier))
        self.Conv8 = conv_block(ch_in=int(256 * multiplier),ch_out=int(512 * multiplier))
        self.Conv_1x1_3 = nn.Conv2d(int(512 * multiplier),img_ch,kernel_size=1,stride=1,padding=0)

        self.zigzag_indices = torch.tensor([
             0,  1,  5,  6, 14, 15, 27, 28, 44, 45, 65, 66, 90, 91,119,120,
             2,  4,  7, 13, 16, 26, 29, 43, 46, 64, 67, 89, 92,118,121,150,
             3,  8, 12, 17, 25, 30, 42, 47, 63, 68, 88, 93,117,122,149,151,
             9, 11, 18, 24, 31, 41, 48, 62, 69, 87, 94,116,123,148,152,177,
            10, 19, 23, 32, 40, 49, 61, 70, 86, 95,115,124,147,153,176,178,
            20, 22, 33, 39, 50, 60, 71, 85, 96,114,125,146,154,175,179,200,
            21, 34, 38, 51, 59, 72, 84, 97,113,126,145,155,174,180,199,201,
            35, 37, 52, 58, 73, 83, 98,112,127,144,156,173,181,198,202,219,
            36, 53, 57, 74, 82, 99,111,128,143,157,172,182,197,203,218,220,
            54, 56, 75, 81,100,110,129,142,158,171,183,196,204,217,221,234,
            55, 76, 80,101,109,130,141,159,170,184,195,205,216,222,233,235,
            77, 79,102,108,131,140,160,169,185,194,206,215,223,232,236,245,
            78,103,107,132,139,161,168,186,193,207,214,224,231,237,244,246,
            104,106,133,138,162,167,187,192,208,213,225,230,238,243,247,252,
            105,134,137,163,166,188,191,209,212,226,229,239,242,248,251,253,
            135,136,164,165,189,190,210,211,227,228,240,241,249,250,254,255
        ]).long() 

    def transform_all(self, x):
        k = random.randrange(0,3)
        if k == 0:
            return self.transform_gb(x)
        elif k == 1:
            return self.transform_gn(x)
        elif k == 2:
            return self.transform_jpeg(x)

    def transform_gb(self, x):
        g_sigma = random.uniform(0.4,1.0)
        transform = T.GaussianBlur(kernel_size=(5, 5), sigma=(g_sigma, g_sigma))
        return transform(x)

    def transform_jpeg(self, input_img):
        # x = x.squeeze()
        B, C, H, W = input_img.shape
        for i in range(B):
            x = T.ToPILImage()(input_img[i,:,:,:])
            x = T.Lambda(self.randomJPEGcompression)(x)
            x = T.ToTensor()(x)
            x = x.cuda()
            input_img[i,:,:,:] = x
        return input_img

    def transform_scb(self, x):
        depth = 3
        x = (x * (2**depth - 1)).int()
        x = (x / (2**depth - 1)).float()
        return x

    def transform_gn(self, x):
        sigma = random.uniform(0.04,0.1)
        x = AddGaussianNoise(0,sigma)(x)
        return x

    def randomJPEGcompression(self, image):
        qf = 10 * random.randrange(1, 10)
        outputIoStream = BytesIO()
        image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
        outputIoStream.seek(0)
        return Image.open(outputIoStream)

    def forward(self,x):
        x = (x * 255).round()

        x_ycbcr = to_ycbcr(x, data_range = 255)
        H, W = x.shape[2], x.shape[3]
        K1 = H//16
        K2 = W//16

        x_blocks = blockify(x,16).transpose(1,2)
        N = x_blocks.shape[0]
        L = x_blocks.shape[1]
        x_blocks = x_blocks.contiguous().view(N * L, x_blocks.shape[2], 16, 16)
        x_blocks = self.Conv1(x_blocks)
        x_blocks = self.Maxpool(x_blocks)
        x_blocks = self.Conv2(x_blocks)
        x_blocks = self.Maxpool(x_blocks) # NL * C * 4 * 4
        x_blocks = x_blocks.view(N, L, x_blocks.shape[1], x_blocks.shape[2], x_blocks.shape[3]).transpose(1,2) # N * C * L * 4 * 4
        feature_patch, _ = torch.max(x_blocks, dim = 2)
        feature_patch = feature_patch.view(feature_patch.shape[0], 16 * feature_patch.shape[1], 1, 1) # N * 16C * 1 * 1

        feature_patch_importance = self.Conv_1x1_2(feature_patch)
        feature_patch_importance = feature_patch_importance.view(feature_patch_importance.shape[0], self.img_ch, self.trigger_channel)
        topk_index = torch.topk(feature_patch_importance, self.trigger_channel//4 * 3, dim = 2, largest = False)[1]
        feature_patch = self.Conv_1x1_1(feature_patch)
        feature_patch = feature_patch.view(feature_patch.shape[0], self.img_ch, self.trigger_channel)
        feature_patch.scatter_(2, topk_index, 0)

        trigger_patch = torch.zeros_like(x[:,:,:16,:16], dtype = feature_patch.dtype)
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 256)
        trigger_patch[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]] = feature_patch
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 16, 16).repeat(1,1,K1,K2)

        trigger_weight = self.Conv5(x)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv6(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv7(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv8(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv_1x1_3(trigger_weight)
        trigger_weight = trigger_weight.repeat_interleave(16,2).repeat_interleave(16,3)

        x_ycbcr_blocks = blockify(x_ycbcr,16) - 128
        dct_blocks = dct_2d(x_ycbcr_blocks)
        dct = deblockify(dct_blocks,(H,W))
        dct_poisoned = dct + trigger_patch.detach() * trigger_weight.detach()
        dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)
        x_poisoned_npp = x_poisoned.div(255)

        x_poisoned = self.transform_gb(x_poisoned_npp)

        x_poisoned_ycbcr = to_ycbcr(x_poisoned, data_range = 255)
        x_poisoned_ycbcr_blocks = blockify(x_poisoned_ycbcr,16) - 128
        dct_x_poisoned_ycbcr_blocks = dct_2d(x_poisoned_ycbcr_blocks)
        dct_x_poisoned_ycbcr = deblockify(dct_x_poisoned_ycbcr_blocks,(H,W))
        delta_dct = (dct_x_poisoned_ycbcr - dct_poisoned.detach())/(trigger_patch.detach() * trigger_weight.detach())
        delta_dct_blocks = blockify(delta_dct,16).abs() # N * C * L * 16 * 16
        delta_dct_blocks = delta_dct_blocks.sum(2) # N * C * 16 * 16
        delta_dct_blocks = delta_dct_blocks.view(delta_dct_blocks.shape[0],delta_dct_blocks.shape[1],256)
        feature_patch_importance_gb = delta_dct_blocks[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]]
        feature_patch_importance_gb.scatter_(2, topk_index, 0)

        x_poisoned = self.transform_gn(x_poisoned_npp)

        x_poisoned_ycbcr = to_ycbcr(x_poisoned, data_range = 255)
        x_poisoned_ycbcr_blocks = blockify(x_poisoned_ycbcr,16) - 128
        dct_x_poisoned_ycbcr_blocks = dct_2d(x_poisoned_ycbcr_blocks)
        dct_x_poisoned_ycbcr = deblockify(dct_x_poisoned_ycbcr_blocks,(H,W))
        delta_dct = (dct_x_poisoned_ycbcr - dct_poisoned.detach())/(trigger_patch.detach() * trigger_weight.detach())
        delta_dct_blocks = blockify(delta_dct,16).abs() # N * C * L * 16 * 16
        delta_dct_blocks = delta_dct_blocks.sum(2) # N * C * 16 * 16
        delta_dct_blocks = delta_dct_blocks.view(delta_dct_blocks.shape[0],delta_dct_blocks.shape[1],256)
        feature_patch_importance_gn = delta_dct_blocks[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]]
        feature_patch_importance_gn.scatter_(2, topk_index, 0)

        feature_patch_importance_pp = feature_patch_importance_gb * feature_patch_importance_gn

        print (feature_patch_importance_pp[0,0,:])
        topk_index_pp = torch.topk(feature_patch_importance_pp, self.trigger_channel//4, dim = 2, largest = True)[1]
        src = torch.zeros_like(topk_index_pp).float()
        for k in range(self.trigger_channel//4):
            # multiple_coefficient = (self.trigger_channel//4 - k) / (self.trigger_channel//4) * 2 + 1
            multiple_coefficient = (k+1) / (self.trigger_channel//4) * 1 + 0.5
            # multiple_coefficient = ((self.trigger_channel//4 - k) / (self.trigger_channel//4) * 2) ** 0.5
            # multiple_coefficient = ((self.trigger_channel//4 - k) / (self.trigger_channel//4) * 1 + 0.5)**0.5
            # multiple_coefficient = (k / (self.trigger_channel//4) * 1.5 + 0.25)**0.5
            src[:,:,k] = multiple_coefficient
        feature_patch.scatter_(2, topk_index_pp, src, reduce = 'multiply')

        dct_poisoned = dct + trigger_patch * trigger_weight
        dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)

        x_poisoned = x_poisoned.div(255)
        
        return x_poisoned


class Trigger_Net_v32(nn.Module):
    def __init__(self, img_ch=3, multiplier = 1, trigger_channel = 64):
        super(Trigger_Net_v32,self).__init__()
        
        self.img_ch = img_ch
        self.trigger_channel = trigger_channel
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv2 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))

        self.Conv_1x1_1 = nn.Conv2d(int(16 * 128 * multiplier), img_ch * trigger_channel, kernel_size=1,stride=1,padding=0)
        self.Conv_1x1_2 = nn.Conv2d(int(16 * 128 * multiplier), img_ch * trigger_channel, kernel_size=1,stride=1,padding=0)

        self.Conv5 = conv_block(ch_in=img_ch,ch_out=int(64 * multiplier))
        self.Conv6 = conv_block(ch_in=int(64 * multiplier),ch_out=int(128 * multiplier))
        self.Conv7 = conv_block(ch_in=int(128 * multiplier),ch_out=int(256 * multiplier))
        self.Conv8 = conv_block(ch_in=int(256 * multiplier),ch_out=int(512 * multiplier))
        self.Conv_1x1_3 = nn.Conv2d(int(512 * multiplier),img_ch,kernel_size=1,stride=1,padding=0)

        self.zigzag_indices = torch.tensor([
             0,  1,  5,  6, 14, 15, 27, 28, 44, 45, 65, 66, 90, 91,119,120,
             2,  4,  7, 13, 16, 26, 29, 43, 46, 64, 67, 89, 92,118,121,150,
             3,  8, 12, 17, 25, 30, 42, 47, 63, 68, 88, 93,117,122,149,151,
             9, 11, 18, 24, 31, 41, 48, 62, 69, 87, 94,116,123,148,152,177,
            10, 19, 23, 32, 40, 49, 61, 70, 86, 95,115,124,147,153,176,178,
            20, 22, 33, 39, 50, 60, 71, 85, 96,114,125,146,154,175,179,200,
            21, 34, 38, 51, 59, 72, 84, 97,113,126,145,155,174,180,199,201,
            35, 37, 52, 58, 73, 83, 98,112,127,144,156,173,181,198,202,219,
            36, 53, 57, 74, 82, 99,111,128,143,157,172,182,197,203,218,220,
            54, 56, 75, 81,100,110,129,142,158,171,183,196,204,217,221,234,
            55, 76, 80,101,109,130,141,159,170,184,195,205,216,222,233,235,
            77, 79,102,108,131,140,160,169,185,194,206,215,223,232,236,245,
            78,103,107,132,139,161,168,186,193,207,214,224,231,237,244,246,
            104,106,133,138,162,167,187,192,208,213,225,230,238,243,247,252,
            105,134,137,163,166,188,191,209,212,226,229,239,242,248,251,253,
            135,136,164,165,189,190,210,211,227,228,240,241,249,250,254,255
        ]).long() 

    def transform_all(self, x):
        k = random.randrange(0,3)
        if k == 0:
            return self.transform_gb(x)
        elif k == 1:
            return self.transform_gn(x)
        elif k == 2:
            return self.transform_jpeg(x)

    def transform_gb(self, x):
        g_sigma = random.uniform(0.4,1.0)
        transform = T.GaussianBlur(kernel_size=(5, 5), sigma=(g_sigma, g_sigma))
        return transform(x)

    def transform_jpeg(self, input_img):
        # x = x.squeeze()
        B, C, H, W = input_img.shape
        for i in range(B):
            x = T.ToPILImage()(input_img[i,:,:,:])
            x = T.Lambda(self.randomJPEGcompression)(x)
            x = T.ToTensor()(x)
            x = x.cuda()
            input_img[i,:,:,:] = x
        return input_img

    def transform_scb(self, x):
        depth = 3
        x = (x * (2**depth - 1)).int()
        x = (x / (2**depth - 1)).float()
        return x

    def transform_gn(self, x):
        sigma = random.uniform(0.04,0.1)
        x = AddGaussianNoise(0,sigma)(x)
        return x

    def randomJPEGcompression(self, image):
        qf = 10 * random.randrange(1, 10)
        outputIoStream = BytesIO()
        image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
        outputIoStream.seek(0)
        return Image.open(outputIoStream)

    def forward(self,x):
        x = (x * 255).round()

        x_ycbcr = to_ycbcr(x, data_range = 255)
        H, W = x.shape[2], x.shape[3]
        K1 = H//16
        K2 = W//16

        trigger_weight = self.Conv5(x)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv6(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv7(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv8(trigger_weight)
        trigger_weight = self.Maxpool(trigger_weight)
        trigger_weight = self.Conv_1x1_3(trigger_weight)
        trigger_weight = trigger_weight.repeat_interleave(16,2).repeat_interleave(16,3)

        x_blocks = blockify(x,16).transpose(1,2)
        N = x_blocks.shape[0]
        L = x_blocks.shape[1]
        x_blocks = x_blocks.contiguous().view(N * L, x_blocks.shape[2], 16, 16)
        x_blocks = self.Conv1(x_blocks)
        x_blocks = self.Maxpool(x_blocks)
        x_blocks = self.Conv2(x_blocks)
        x_blocks = self.Maxpool(x_blocks) # NL * C * 4 * 4
        x_blocks = x_blocks.view(N, L, x_blocks.shape[1], x_blocks.shape[2], x_blocks.shape[3]).transpose(1,2) # N * C * L * 4 * 4
        feature_patch, _ = torch.max(x_blocks, dim = 2)
        feature_patch = feature_patch.view(feature_patch.shape[0], 16 * feature_patch.shape[1], 1, 1) # N * 16C * 1 * 1

        feature_patch = self.Conv_1x1_1(feature_patch)
        feature_patch = feature_patch.view(feature_patch.shape[0], self.img_ch, self.trigger_channel)

        x_ycbcr_blocks = blockify(x_ycbcr,16) - 128
        dct_blocks = dct_2d(x_ycbcr_blocks)
        dct = deblockify(dct_blocks,(H,W))
        trigger_patch = torch.zeros_like(x[:,:,:16,:16], dtype = feature_patch.dtype)
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 256)
        trigger_patch[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]] = feature_patch.detach()
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 16, 16).repeat(1,1,K1,K2)
        dct_poisoned = dct + trigger_patch * trigger_weight.detach()
        dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)
        x_poisoned_npp = x_poisoned.div(255)

        x_poisoned = self.transform_gb(x_poisoned_npp)

        x_poisoned_ycbcr = to_ycbcr(x_poisoned, data_range = 255)
        x_poisoned_ycbcr_blocks = blockify(x_poisoned_ycbcr,16) - 128
        dct_x_poisoned_ycbcr_blocks = dct_2d(x_poisoned_ycbcr_blocks)
        dct_x_poisoned_ycbcr = deblockify(dct_x_poisoned_ycbcr_blocks,(H,W))
        delta_dct = (dct_x_poisoned_ycbcr - dct_poisoned.detach())/(trigger_patch * trigger_weight.detach())
        delta_dct_blocks = blockify(delta_dct,16).abs() # N * C * L * 16 * 16
        delta_dct_blocks = delta_dct_blocks.sum(2) # N * C * 16 * 16
        delta_dct_blocks = delta_dct_blocks.view(delta_dct_blocks.shape[0],delta_dct_blocks.shape[1],256)
        feature_patch_importance_gb = delta_dct_blocks[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]]

        x_poisoned = self.transform_gn(x_poisoned_npp)

        x_poisoned_ycbcr = to_ycbcr(x_poisoned, data_range = 255)
        x_poisoned_ycbcr_blocks = blockify(x_poisoned_ycbcr,16) - 128
        dct_x_poisoned_ycbcr_blocks = dct_2d(x_poisoned_ycbcr_blocks)
        dct_x_poisoned_ycbcr = deblockify(dct_x_poisoned_ycbcr_blocks,(H,W))
        delta_dct = (dct_x_poisoned_ycbcr - dct_poisoned.detach())/(trigger_patch * trigger_weight.detach())
        delta_dct_blocks = blockify(delta_dct,16).abs() # N * C * L * 16 * 16
        delta_dct_blocks = delta_dct_blocks.sum(2) # N * C * 16 * 16
        delta_dct_blocks = delta_dct_blocks.view(delta_dct_blocks.shape[0],delta_dct_blocks.shape[1],256)
        feature_patch_importance_gn = delta_dct_blocks[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]]

        x_poisoned = self.transform_jpeg(x_poisoned_npp.clone())

        x_poisoned_ycbcr = to_ycbcr(x_poisoned, data_range = 255)
        x_poisoned_ycbcr_blocks = blockify(x_poisoned_ycbcr,16) - 128
        dct_x_poisoned_ycbcr_blocks = dct_2d(x_poisoned_ycbcr_blocks)
        dct_x_poisoned_ycbcr = deblockify(dct_x_poisoned_ycbcr_blocks,(H,W))
        delta_dct = (dct_x_poisoned_ycbcr - dct_poisoned.detach())/(trigger_patch * trigger_weight.detach())
        delta_dct_blocks = blockify(delta_dct,16).abs() # N * C * L * 16 * 16
        delta_dct_blocks = delta_dct_blocks.sum(2) # N * C * 16 * 16
        delta_dct_blocks = delta_dct_blocks.view(delta_dct_blocks.shape[0],delta_dct_blocks.shape[1],256)
        feature_patch_importance_jpeg = delta_dct_blocks[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]]

        feature_patch_importance = feature_patch_importance_gb * feature_patch_importance_gn * feature_patch_importance_jpeg
        topk_index = torch.topk(feature_patch_importance, self.trigger_channel//4 * 3, dim = 2, largest = True)[1]
        feature_patch.scatter_(2, topk_index, 0)

        topk_index = torch.topk(feature_patch_importance, self.trigger_channel//4, dim = 2, largest = False)[1]
        src = torch.zeros_like(topk_index).float()
        for k in range(self.trigger_channel//4):
            # multiple_coefficient = (self.trigger_channel//4 - k) / (self.trigger_channel//4) * 2 + 1
            # multiple_coefficient = (self.trigger_channel//4 - k) / (self.trigger_channel//4) * 1 + 0.5
            # multiple_coefficient = ((self.trigger_channel//4 - k) / (self.trigger_channel//4) * 2) ** 0.5
            multiple_coefficient = ((self.trigger_channel//4 - k) / (self.trigger_channel//4) * 1 + 0.5)**0.5
            # multiple_coefficient = ((self.trigger_channel//4 - k) / (self.trigger_channel//4) * 1.5 + 0.25)**0.5
            # multiple_coefficient = 0.8 * ((self.trigger_channel//4 - k) / (self.trigger_channel//4) * 1.5 + 0.25)**0.5
            # multiple_coefficient = ((self.trigger_channel//4 - 1 - k) / (self.trigger_channel//4 - 1) * 1 + 0.5)**0.5
            # multiple_coefficient = 1
            src[:,:,k] = multiple_coefficient
        feature_patch.scatter_(2, topk_index, src, reduce = 'multiply')

        trigger_patch = torch.zeros_like(x[:,:,:16,:16], dtype = feature_patch.dtype)
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 256)
        trigger_patch[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]] = feature_patch
        trigger_patch = trigger_patch.view(trigger_patch.shape[0], trigger_patch.shape[1], 16, 16).repeat(1,1,K1,K2)

        dct_poisoned = dct + trigger_patch * trigger_weight
        dct_poisoned_blocks = blockify(dct_poisoned,16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)

        x_poisoned = x_poisoned.div(255)
        
        # return x_poisoned
        return x_poisoned, trigger_weight

def randomJPEGcompression(image):
    qf = 10 * random.randrange(1, 10)
    outputIoStream = BytesIO()
    image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
    outputIoStream.seek(0)
    return Image.open(outputIoStream)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), device = tensor.device) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def transform_all(x):
    k = random.randrange(0,3)
    if k == 0:
        return transform_gb(x)
    elif k == 1:
        return transform_gn(x)
    elif k == 2:
        return transform_jpeg(x)

def transform_gb(x):
    g_sigma = random.uniform(0.5,0.7)
    transform = T.GaussianBlur(kernel_size=(5, 5), sigma=(g_sigma, g_sigma))
    return transform(x)

def transform_jpeg(input_img):
    # x = x.squeeze()
    B, C, H, W = input_img.shape
    for i in range(B):
        x = T.ToPILImage()(input_img[i,:,:,:])
        x = T.Lambda(randomJPEGcompression)(x)
        x = T.ToTensor()(x)
        x = x.cuda()
        input_img[i,:,:,:] = x
    return input_img

def transform_scb(x):
    depth = 3
    x = (x * (2**depth - 1)).int()
    x = (x / (2**depth - 1)).float()
    return x

def transform_gn(x):
    sigma = random.uniform(0.02,0.1)
    x = AddGaussianNoise(0,sigma)(x)
    return x


def pad_to(x, stride):
    h, w = x.shape[-2:]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pads = (lw, uw, lh, uh)

    # zero-padding by default.
    # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = F.pad(x, pads, "constant", 0)

    return out, pads

def unpad(x, pad):
    if pad[2]+pad[3] > 0:
        x = x[:,:,pad[2]:-pad[3],:]
    if pad[0]+pad[1] > 0:
        x = x[:,:,:,pad[0]:-pad[1]]
    return x



class Add_trigger_net_v1(nn.Module):
    def __init__(self, trigger_channel = 64):
        super(Add_trigger_net_v1,self).__init__()
        self.trigger_channel = trigger_channel
        self.zigzag_indices = torch.tensor([
             0,  1,  5,  6, 14, 15, 27, 28, 44, 45, 65, 66, 90, 91,119,120,
             2,  4,  7, 13, 16, 26, 29, 43, 46, 64, 67, 89, 92,118,121,150,
             3,  8, 12, 17, 25, 30, 42, 47, 63, 68, 88, 93,117,122,149,151,
             9, 11, 18, 24, 31, 41, 48, 62, 69, 87, 94,116,123,148,152,177,
            10, 19, 23, 32, 40, 49, 61, 70, 86, 95,115,124,147,153,176,178,
            20, 22, 33, 39, 50, 60, 71, 85, 96,114,125,146,154,175,179,200,
            21, 34, 38, 51, 59, 72, 84, 97,113,126,145,155,174,180,199,201,
            35, 37, 52, 58, 73, 83, 98,112,127,144,156,173,181,198,202,219,
            36, 53, 57, 74, 82, 99,111,128,143,157,172,182,197,203,218,220,
            54, 56, 75, 81,100,110,129,142,158,171,183,196,204,217,221,234,
            55, 76, 80,101,109,130,141,159,170,184,195,205,216,222,233,235,
            77, 79,102,108,131,140,160,169,185,194,206,215,223,232,236,245,
            78,103,107,132,139,161,168,186,193,207,214,224,231,237,244,246,
            104,106,133,138,162,167,187,192,208,213,225,230,238,243,247,252,
            105,134,137,163,166,188,191,209,212,226,229,239,242,248,251,253,
            135,136,164,165,189,190,210,211,227,228,240,241,249,250,254,255
        ]).long() 
        
    def forward(self,x, strength):
        if len(x.shape) == 3:
            mark = 1
            x = x.unsqueeze(0)
        else:
            mark = 0

        x, pads = pad_to(x, 16)
        x = (x * 255).round()

        x_ycbcr = to_ycbcr(x, data_range = 255)
        H, W = x.shape[2], x.shape[3]
        K1 = H//16
        K2 = W//16

        x_ycbcr_blocks = blockify(x_ycbcr,16) - 128
        dct_blocks = dct_2d(x_ycbcr_blocks)
        dct_poisoned_blocks = dct_blocks.contiguous().view(dct_blocks.shape[0], dct_blocks.shape[1], dct_blocks.shape[2], 256)
        dct_poisoned_blocks[..., self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]] += strength
        dct_poisoned_blocks = dct_poisoned_blocks.view(dct_blocks.shape[0], dct_blocks.shape[1], dct_blocks.shape[2], 16, 16)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)

        x_poisoned = x_poisoned.div(255)

        x_poisoned = unpad(x_poisoned, pads)

        x_poisoned = x_poisoned.clamp(0, 1)
        if mark:
            x_poisoned = x_poisoned.squeeze(0)
        return x_poisoned


class Add_trigger_net_v2(nn.Module):
    def __init__(self, trigger_channel = 64):
        super(Add_trigger_net_v2,self).__init__()
        self.trigger_channel = trigger_channel
        self.zigzag_indices = torch.tensor([
             0,  1,  5,  6, 14, 15, 27, 28, 44, 45, 65, 66, 90, 91,119,120,
             2,  4,  7, 13, 16, 26, 29, 43, 46, 64, 67, 89, 92,118,121,150,
             3,  8, 12, 17, 25, 30, 42, 47, 63, 68, 88, 93,117,122,149,151,
             9, 11, 18, 24, 31, 41, 48, 62, 69, 87, 94,116,123,148,152,177,
            10, 19, 23, 32, 40, 49, 61, 70, 86, 95,115,124,147,153,176,178,
            20, 22, 33, 39, 50, 60, 71, 85, 96,114,125,146,154,175,179,200,
            21, 34, 38, 51, 59, 72, 84, 97,113,126,145,155,174,180,199,201,
            35, 37, 52, 58, 73, 83, 98,112,127,144,156,173,181,198,202,219,
            36, 53, 57, 74, 82, 99,111,128,143,157,172,182,197,203,218,220,
            54, 56, 75, 81,100,110,129,142,158,171,183,196,204,217,221,234,
            55, 76, 80,101,109,130,141,159,170,184,195,205,216,222,233,235,
            77, 79,102,108,131,140,160,169,185,194,206,215,223,232,236,245,
            78,103,107,132,139,161,168,186,193,207,214,224,231,237,244,246,
            104,106,133,138,162,167,187,192,208,213,225,230,238,243,247,252,
            105,134,137,163,166,188,191,209,212,226,229,239,242,248,251,253,
            135,136,164,165,189,190,210,211,227,228,240,241,249,250,254,255
        ]).long() 
        
    def forward(self,x, strength):
        if len(x.shape) == 3:
            mark = 1
            x = x.unsqueeze(0)
        else:
            mark = 0

        x, pads = pad_to(x, 16)
        x = (x * 255).round()

        x_ycbcr = to_ycbcr(x, data_range = 255)
        H, W = x.shape[2], x.shape[3]
        K1 = H//16
        K2 = W//16

        x_ycbcr_blocks = blockify(x_ycbcr,16) - 128
        dct_blocks = dct_2d(x_ycbcr_blocks) # N*C*L*16*16
        dct_poisoned_blocks = dct_blocks.permute(0,2,1,3,4).contiguous().view(dct_blocks.shape[0], dct_blocks.shape[2], 3 , 16 * 16)
        dct_poisoned_blocks[..., :, self.zigzag_indices[128 - self.trigger_channel//2: 128 - self.trigger_channel//2 + self.trigger_channel]] += strength
        dct_poisoned_blocks = dct_poisoned_blocks.view(dct_blocks.shape[0], dct_blocks.shape[2], 3, 16, 16)
        dct_poisoned_blocks = dct_poisoned_blocks.permute(0,2,1,3,4)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)

        x_poisoned = x_poisoned.div(255)

        x_poisoned = unpad(x_poisoned, pads)

        x_poisoned = x_poisoned.clamp(0, 1)
        if mark:
            x_poisoned = x_poisoned.squeeze(0)
        return x_poisoned


class Add_trigger_net_v3(nn.Module):
    def __init__(self, trigger_channel = 64):
        super(Add_trigger_net_v3,self).__init__()
        self.trigger_channel = trigger_channel
        self.zigzag_indices = torch.tensor([
             0,  1,  5,  6, 14, 15, 27, 28, 44, 45, 65, 66, 90, 91,119,120,
             2,  4,  7, 13, 16, 26, 29, 43, 46, 64, 67, 89, 92,118,121,150,
             3,  8, 12, 17, 25, 30, 42, 47, 63, 68, 88, 93,117,122,149,151,
             9, 11, 18, 24, 31, 41, 48, 62, 69, 87, 94,116,123,148,152,177,
            10, 19, 23, 32, 40, 49, 61, 70, 86, 95,115,124,147,153,176,178,
            20, 22, 33, 39, 50, 60, 71, 85, 96,114,125,146,154,175,179,200,
            21, 34, 38, 51, 59, 72, 84, 97,113,126,145,155,174,180,199,201,
            35, 37, 52, 58, 73, 83, 98,112,127,144,156,173,181,198,202,219,
            36, 53, 57, 74, 82, 99,111,128,143,157,172,182,197,203,218,220,
            54, 56, 75, 81,100,110,129,142,158,171,183,196,204,217,221,234,
            55, 76, 80,101,109,130,141,159,170,184,195,205,216,222,233,235,
            77, 79,102,108,131,140,160,169,185,194,206,215,223,232,236,245,
            78,103,107,132,139,161,168,186,193,207,214,224,231,237,244,246,
            104,106,133,138,162,167,187,192,208,213,225,230,238,243,247,252,
            105,134,137,163,166,188,191,209,212,226,229,239,242,248,251,253,
            135,136,164,165,189,190,210,211,227,228,240,241,249,250,254,255
        ]).long() 
        
    def forward(self,x, strength):
        if len(x.shape) == 3:
            mark = 1
            x = x.unsqueeze(0)
        else:
            mark = 0

        x, pads = pad_to(x, 16)
        x = (x * 255).round()

        x_ycbcr = to_ycbcr(x, data_range = 255)
        H, W = x.shape[2], x.shape[3]
        K1 = H//16
        K2 = W//16

        x_ycbcr_blocks = blockify(x_ycbcr,16) - 128
        dct_blocks = dct_2d(x_ycbcr_blocks) # N*C*L*16*16
        dct_poisoned_blocks = dct_blocks.permute(0,2,1,3,4).contiguous().view(dct_blocks.shape[0], dct_blocks.shape[2], 3 , 16 * 16)
        dct_poisoned_blocks[..., :, self.zigzag_indices[256 - self.trigger_channel: 256]] += strength
        dct_poisoned_blocks = dct_poisoned_blocks.view(dct_blocks.shape[0], dct_blocks.shape[2], 3, 16, 16)
        dct_poisoned_blocks = dct_poisoned_blocks.permute(0,2,1,3,4)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)

        x_poisoned = x_poisoned.div(255)

        x_poisoned = unpad(x_poisoned, pads)

        x_poisoned = x_poisoned.clamp(0, 1)
        if mark:
            x_poisoned = x_poisoned.squeeze(0)
        return x_poisoned


class Add_trigger_net_v4(nn.Module):
    def __init__(self, trigger_channel = 64):
        super(Add_trigger_net_v4,self).__init__()
        self.trigger_channel = trigger_channel
        self.zigzag_indices = torch.tensor([
             0,  1,  5,  6, 14, 15, 27, 28, 44, 45, 65, 66, 90, 91,119,120,
             2,  4,  7, 13, 16, 26, 29, 43, 46, 64, 67, 89, 92,118,121,150,
             3,  8, 12, 17, 25, 30, 42, 47, 63, 68, 88, 93,117,122,149,151,
             9, 11, 18, 24, 31, 41, 48, 62, 69, 87, 94,116,123,148,152,177,
            10, 19, 23, 32, 40, 49, 61, 70, 86, 95,115,124,147,153,176,178,
            20, 22, 33, 39, 50, 60, 71, 85, 96,114,125,146,154,175,179,200,
            21, 34, 38, 51, 59, 72, 84, 97,113,126,145,155,174,180,199,201,
            35, 37, 52, 58, 73, 83, 98,112,127,144,156,173,181,198,202,219,
            36, 53, 57, 74, 82, 99,111,128,143,157,172,182,197,203,218,220,
            54, 56, 75, 81,100,110,129,142,158,171,183,196,204,217,221,234,
            55, 76, 80,101,109,130,141,159,170,184,195,205,216,222,233,235,
            77, 79,102,108,131,140,160,169,185,194,206,215,223,232,236,245,
            78,103,107,132,139,161,168,186,193,207,214,224,231,237,244,246,
            104,106,133,138,162,167,187,192,208,213,225,230,238,243,247,252,
            105,134,137,163,166,188,191,209,212,226,229,239,242,248,251,253,
            135,136,164,165,189,190,210,211,227,228,240,241,249,250,254,255
        ]).long() 
        
    def forward(self, x, strength):
        if len(x.shape) == 3:
            mark = 1
            x = x.unsqueeze(0)
        else:
            mark = 0

        x, pads = pad_to(x, 16)
        x = (x * 255).round()

        x_ycbcr = to_ycbcr(x, data_range = 255)
        H, W = x.shape[2], x.shape[3]
        K1 = H//16
        K2 = W//16

        x_ycbcr_blocks = blockify(x_ycbcr,16) - 128
        dct_blocks = dct_2d(x_ycbcr_blocks) # N*C*L*16*16
        dct_poisoned_blocks = dct_blocks.permute(0,2,1,3,4).contiguous().view(dct_blocks.shape[0], dct_blocks.shape[2], 3 , 16 * 16)
        dct_poisoned_blocks[..., :, self.zigzag_indices[128 - self.trigger_channel//4: 128 - self.trigger_channel//4 + self.trigger_channel//2]] += strength
        dct_poisoned_blocks[..., :, self.zigzag_indices[256 - self.trigger_channel//2: 256]] += strength
        dct_poisoned_blocks = dct_poisoned_blocks.view(dct_blocks.shape[0], dct_blocks.shape[2], 3, 16, 16)
        dct_poisoned_blocks = dct_poisoned_blocks.permute(0,2,1,3,4)
        x_ycbcr_poisoned_blocks = idct_2d(dct_poisoned_blocks)
        x_ycbcr_poisoned = deblockify(x_ycbcr_poisoned_blocks,(H,W)) + 128
        
        x_poisoned = to_rgb(x_ycbcr_poisoned, data_range = 255)

        x_poisoned = x_poisoned.div(255)

        x_poisoned = unpad(x_poisoned, pads)

        x_poisoned = x_poisoned.clamp(0, 1)
        if mark:
            x_poisoned = x_poisoned.squeeze(0)
        return x_poisoned