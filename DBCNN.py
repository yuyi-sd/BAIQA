import os
import torch
import torchvision
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np


# import torch
# import torch.nn as nn
import timm
from timm.models.vision_transformer import Block
# from models.swin import SwinTransformer
# import torch.nn.functional as F
from einops import rearrange
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from torchvision import models
from transformers import Transformer
from posencode import PositionEmbeddingSine


class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=1, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2 )//2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:,None]*a[None,:])
        g = g/torch.sum(g)
        self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))
    def forward(self, input):
        input = input**2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out+1e-12).sqrt()
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.device = device
        # self.cfg = cfg
        self.L2pooling_l1 = L2pooling(channels=256)
        self.L2pooling_l2 = L2pooling(channels=512)
        self.L2pooling_l3 = L2pooling(channels=1024)
        self.L2pooling_l4 = L2pooling(channels=2048)            
        # if cfg.network =='resnet50':
        #     from resnet_modify  import resnet50 as resnet_modifyresnet
        #     dim_modelt = 3840
        #     modelpretrain = models.resnet50(pretrained=True)
        # elif cfg.network =='resnet34':
        #     from resnet_modify  import resnet34 as resnet_modifyresnet
        #     modelpretrain = models.resnet34(pretrained=True)
        #     dim_modelt = 960
        #     self.L2pooling_l1 = L2pooling(channels=64)
        #     self.L2pooling_l2 = L2pooling(channels=128)
        #     self.L2pooling_l3 = L2pooling(channels=256)
        #     self.L2pooling_l4 = L2pooling(channels=512)
        # elif cfg.network == 'resnet18':
        #     from resnet_modify  import resnet18 as resnet_modifyresnet
        #     modelpretrain = models.resnet18(pretrained=True)
        #     dim_modelt = 960
        #     self.L2pooling_l1 = L2pooling(channels=64)
        #     self.L2pooling_l2 = L2pooling(channels=128)
        #     self.L2pooling_l3 = L2pooling(channels=256)
        #     self.L2pooling_l4 = L2pooling(channels=512)

        from resnet_modify  import resnet18 as resnet_modifyresnet
        modelpretrain = models.resnet18(pretrained=True)
        dim_modelt = 960
        self.L2pooling_l1 = L2pooling(channels=64)
        self.L2pooling_l2 = L2pooling(channels=128)
        self.L2pooling_l3 = L2pooling(channels=256)
        self.L2pooling_l4 = L2pooling(channels=512)

        # torch.save(modelpretrain.state_dict(), 'modelpretrain')
        self.model = resnet_modifyresnet()
        # self.model.load_state_dict(torch.load('modelpretrain'), strict=True)
        self.model.load_state_dict(modelpretrain.state_dict(), strict=True)
        self.dim_modelt = dim_modelt
        # os.remove("modelpretrain")
    
        nheadt=16
        num_encoder_layerst=2
        dim_feedforwardt=64
        ddropout=0.5
        normalize =True
        self.transformer = Transformer(d_model=dim_modelt,nhead=nheadt,
                                       num_encoder_layers=num_encoder_layerst,
                                       dim_feedforward=dim_feedforwardt,
                                       normalize_before=normalize,
                                       dropout = ddropout)
        self.position_embedding = PositionEmbeddingSine(dim_modelt // 2, normalize=True)
        self.fc2 = nn.Linear(dim_modelt, self.model.fc.in_features) 
        self.fc = nn.Linear(self.model.fc.in_features*2, 1) 
        self.ReLU = nn.ReLU()
        self.avg7 = nn.AvgPool2d((7, 7))
        self.avg8 = nn.AvgPool2d((8, 8))
        self.avg4 = nn.AvgPool2d((4, 4))
        self.avg2 = nn.AvgPool2d((2, 2))
        self.drop2d = nn.Dropout(p=0.1)
        self.consistency = nn.L1Loss()

    def forward(self, x):
        self.pos_enc_1 = self.position_embedding(torch.ones(1, self.dim_modelt, 7, 7).cuda())
        self.pos_enc = self.pos_enc_1.repeat(x.shape[0],1,1,1).contiguous()

        out,layer1,layer2,layer3,layer4 = self.model(x) 

        layer1_t = self.avg8(self.drop2d(self.L2pooling_l1(F.normalize(layer1,dim=1, p=2))))
        layer2_t = self.avg4(self.drop2d(self.L2pooling_l2(F.normalize(layer2,dim=1, p=2))))
        layer3_t = self.avg2(self.drop2d(self.L2pooling_l3(F.normalize(layer3,dim=1, p=2))))
        layer4_t =           self.drop2d(self.L2pooling_l4(F.normalize(layer4,dim=1, p=2)))
        layers = torch.cat((layer1_t,layer2_t,layer3_t,layer4_t),dim=1)

        out_t_c = self.transformer(layers,self.pos_enc)
        out_t_o = torch.flatten(self.avg7(out_t_c),start_dim=1)
        out_t_o = self.fc2(out_t_o)
        layer4_o = self.avg7(layer4)
        layer4_o = torch.flatten(layer4_o,start_dim=1)
        predictionQA = self.fc(torch.flatten(torch.cat((out_t_o,layer4_o),dim=1),start_dim=1))
        
        # =============================================================================
        # =============================================================================


        fout,flayer1,flayer2,flayer3,flayer4 = self.model(torch.flip(x, [3])) 
        flayer1_t = self.avg8( self.L2pooling_l1(F.normalize(flayer1,dim=1, p=2)))
        flayer2_t = self.avg4( self.L2pooling_l2(F.normalize(flayer2,dim=1, p=2)))
        flayer3_t = self.avg2( self.L2pooling_l3(F.normalize(flayer3,dim=1, p=2)))
        flayer4_t =            self.L2pooling_l4(F.normalize(flayer4,dim=1, p=2))
        flayers = torch.cat((flayer1_t,flayer2_t,flayer3_t,flayer4_t),dim=1)
        fout_t_c = self.transformer(flayers,self.pos_enc)
        fout_t_o = torch.flatten(self.avg7(fout_t_c),start_dim=1)
        fout_t_o = (self.fc2(fout_t_o))
        flayer4_o = self.avg7(flayer4)
        flayer4_o = torch.flatten(flayer4_o,start_dim=1)
        fpredictionQA =  (self.fc(torch.flatten(torch.cat((fout_t_o,flayer4_o),dim=1),start_dim=1)))

        
        consistloss1 = self.consistency(out_t_c,fout_t_c.detach())
        consistloss2 = self.consistency(layer4,flayer4.detach())
        consistloss = 1*(consistloss1+consistloss2)
                
        # return predictionQA, consistloss
        return predictionQA


def SPSP(x, P=1, method='avg'):
    batch_size = x.size(0)
    map_size = x.size()[-2:]
    pool_features = []
    for p in range(1, P+1):
        pool_size = [np.int(d / p) for d in map_size]
        if method == 'maxmin':
            M = F.max_pool2d(x, pool_size)
            m = -F.max_pool2d(-x, pool_size)
            pool_features.append(torch.cat((M, m), 1).view(batch_size, -1))  # max & min pooling
        elif method == 'max':
            M = F.max_pool2d(x, pool_size)
            pool_features.append(M.view(batch_size, -1))  # max pooling
        elif method == 'min':
            m = -F.max_pool2d(-x, pool_size)
            pool_features.append(m.view(batch_size, -1))  # min pooling
        elif method == 'avg':
            a = F.avg_pool2d(x, pool_size)
            pool_features.append(a.view(batch_size, -1))  # average pooling
        else:
            m1  = F.avg_pool2d(x, pool_size)
            rm2 = torch.sqrt(F.relu(F.avg_pool2d(torch.pow(x, 2), pool_size) - torch.pow(m1, 2)))
            if method == 'std':
                pool_features.append(rm2.view(batch_size, -1))  # std pooling
            else:
                pool_features.append(torch.cat((m1, rm2), 1).view(batch_size, -1))  # statistical pooling: mean & std
    return torch.cat(pool_features, dim=1)

class LinearityIQA(nn.Module):
    def __init__(self, arch='resnext101_32x8d', pool='avg', use_bn_end=False, P6=1, P7=1):
        super().__init__()
        self.pool = pool
        self.use_bn_end = use_bn_end
        if pool in ['max', 'min', 'avg', 'std']:
            c = 1
        else:
            c = 2
        self.P6 = P6  #
        self.P7 = P7  #
        features = list(models.__dict__[arch](pretrained=True).children())[:-2]
        if arch == 'alexnet':
            in_features = [256, 256]
            self.id1 = 9
            self.id2 = 12
            features = features[0]
        elif arch == 'vgg16':
            in_features = [512, 512]
            self.id1 = 23
            self.id2 = 30
            features = features[0]
        elif 'res' in arch:
            self.id1 = 6
            self.id2 = 7
            if arch == 'resnet18' or arch == 'resnet34':
                in_features = [256, 512]
            else:
                in_features = [1024, 2048]
        else: 
            print('The arch is not implemented!')
        self.features = nn.Sequential(*features)
        self.dr6 = nn.Sequential(nn.Linear(in_features[0] * c * sum([p * p for p in range(1, self.P6+1)]), 1024),
                                 nn.BatchNorm1d(1024),
                                 nn.Linear(1024, 256),
                                 nn.BatchNorm1d(256),
                                 nn.Linear(256, 64),
                                 nn.BatchNorm1d(64), nn.ReLU())
        self.dr7 = nn.Sequential(nn.Linear(in_features[1] * c * sum([p * p for p in range(1, self.P7+1)]), 1024),
                                 nn.BatchNorm1d(1024),
                                 nn.Linear(1024, 256),
                                 nn.BatchNorm1d(256),
                                 nn.Linear(256, 64),
                                 nn.BatchNorm1d(64), nn.ReLU())

        if self.use_bn_end:
            self.regr6 = nn.Sequential(nn.Linear(64, 1), nn.BatchNorm1d(1))
            self.regr7 = nn.Sequential(nn.Linear(64, 1), nn.BatchNorm1d(1))
            self.regression = nn.Sequential(nn.Linear(64 * 2, 1), nn.BatchNorm1d(1))
        else:
            self.regr6 = nn.Linear(64, 1)
            self.regr7 = nn.Linear(64, 1)
            self.regression = nn.Linear(64 * 2, 1)

    def extract_features(self, x):
        f, pq = [], []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == self.id1:
                x6 = SPSP(x, P=self.P6, method=self.pool)
                x6 = self.dr6(x6)
                f.append(x6)
                pq.append(self.regr6(x6))
            if ii == self.id2:
                x7 = SPSP(x, P=self.P7, method=self.pool)
                x7 = self.dr7(x7)
                f.append(x7)
                pq.append(self.regr7(x7))

        f = torch.cat(f, dim=1)

        return f, pq

    def forward(self, x):
        f, pq = self.extract_features(x)
        s = self.regression(f)
        pq.append(s)

        return pq

def norm_loss_with_normalization(y_pred, y, alpha=[1, 1], p=2, q=2, detach=False, exponent=True):
    """norm_loss_with_normalization: norm-in-norm"""
    # y = y.squeeze()
    N = y_pred.size(0)
    if N > 1:  
        m_hat = torch.mean(y_pred.detach()) if detach else torch.mean(y_pred)
        y_pred = y_pred - m_hat  # very important!!
        normalization = torch.norm(y_pred.detach(), p=q) if detach else torch.norm(y_pred, p=q)  # Actually, z-score normalization is related to q = 2.
        # print('bhat = {}'.format(normalization.item()))
        y_pred = y_pred / (eps + normalization)  # very important!
        y = y - torch.mean(y)
        y = y / (eps + torch.norm(y, p=q))
        scale = np.power(2, max(1,1./q)) * np.power(N, max(0,1./p-1./q)) # p, q>0
        loss0, loss1 = 0, 0
        if alpha[0] > 0:
            err = y_pred - y
            if p < 1:  # avoid gradient explosion when 0<=p<1; and avoid vanishing gradient problem when p < 0
                err += eps 
            loss0 = torch.norm(err, p=p) / scale  # Actually, p=q=2 is related to PLCC
            loss0 = torch.pow(loss0, p) if exponent else loss0 #
        if alpha[1] > 0:
            rho =  torch.cosine_similarity(y_pred.t(), y.t())  #  
            err = rho * y_pred - y
            if p < 1:  # avoid gradient explosion when 0<=p<1; and avoid vanishing gradient problem when p < 0
                err += eps 
            loss1 = torch.norm(err, p=p) / scale  # Actually, p=q=2 is related to LSR
            loss1 = torch.pow(loss1, p) if exponent else loss1 #  #  
        return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
    else:
        return F.l1_loss(y_pred, y_pred.detach())  # 0 for batch with single sample.


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 dim_mlp=1024., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.dim_mlp = dim_mlp
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = self.dim_mlp
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size=7,
                 dim_mlp=1024, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 dim_mlp=dim_mlp,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.input_resolution[0], w=self.input_resolution[1])
        x = F.relu(self.conv(x))
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class SwinTransformer(nn.Module):
    def __init__(self, patches_resolution, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 embed_dim=256, drop=0.1, drop_rate=0., drop_path_rate=0.1, dropout=0., window_size=7,
                 dim_mlp=1024, qkv_bias=True, qk_scale=None, attn_drop_rate=0., norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False, scale=0.8, **kwargs):
        super().__init__()
        self.scale = scale
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.dropout = nn.Dropout(p=drop)
        self.num_features = embed_dim
        self.num_layers = len(depths)
        self.patches_resolution = (patches_resolution[0], patches_resolution[1])
        self.downsample = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, stride=2, padding=1)
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=self.embed_dim,
                input_resolution=patches_resolution,
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size,
                dim_mlp=dim_mlp,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=dropout,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=downsample,
                use_checkpoint=use_checkpoint
            )
            self.layers.append(layer)

    def forward(self, x):
        x = self.dropout(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for layer in self.layers:
            _x = x
            x = layer(x)
            x = self.scale * x + _x
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.patches_resolution[0], w=self.patches_resolution[1])
        return x


class TABlock(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        _x = x
        B, C, N = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x
        return x


class SaveOutput:
    def __init__(self):
        self.outputs = []
    
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
    
    def clear(self):
        self.outputs = []


class MANIQA(nn.Module):
    def __init__(self, embed_dim=768, num_outputs=1, patch_size=8, drop=0.1, 
                    depths=[2, 2], window_size=4, dim_mlp=768, num_heads=[4, 4],
                    img_size=224, num_tab=2, scale=0.8, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_size = img_size // patch_size
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        
        self.vit = timm.create_model('vit_base_patch8_224', pretrained=True)
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

        self.tablock1 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2)
            self.tablock1.append(tab)

        self.conv1 = nn.Conv2d(embed_dim * 4, embed_dim, 1, 1, 0)
        self.swintransformer1 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )

        self.tablock2 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2)
            self.tablock2.append(tab)

        self.conv2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.swintransformer2 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim // 2,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )
        
        self.fc_score = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.ReLU()
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.Sigmoid()
        )
    
    def extract_feature(self, save_output):
        x6 = save_output.outputs[6][:, 1:]
        x7 = save_output.outputs[7][:, 1:]
        x8 = save_output.outputs[8][:, 1:]
        x9 = save_output.outputs[9][:, 1:]
        x = torch.cat((x6, x7, x8, x9), dim=2)
        return x

    def forward(self, x):
        _x = self.vit(x)
        x = self.extract_feature(self.save_output)
        self.save_output.outputs.clear()

        # stage 1
        x = rearrange(x, 'b (h w) c -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock1:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv1(x)
        x = self.swintransformer1(x)

        # stage2
        x = rearrange(x, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock2:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv2(x)
        x = self.swintransformer2(x)

        x = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        score = torch.tensor([]).cuda()
        for i in range(x.shape[0]):
            f = self.fc_score(x[i])
            w = self.fc_weight(x[i])
            _s = torch.sum(f * w) / torch.sum(w)
            score = torch.cat((score, _s.unsqueeze(0)), 0)
        return score
        

def weight_init(net): 
    for m in net.modules():    
        if isinstance(m, nn.Conv2d):         
            nn.init.kaiming_normal_(m.weight.data,nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data,nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        


class SCNN(nn.Module):

    def __init__(self):
        """Declare all needed layers."""
        super(SCNN, self).__init__()

        # Linear classifier.

        self.num_class = 39
#        self.features = nn.Sequential(nn.Conv2d(3,48,3,1,1),nn.ReLU(inplace=True),
#                                      nn.Conv2d(48,48,3,2,1),nn.ReLU(inplace=True),
#                                      nn.Conv2d(48,64,3,1,1),nn.ReLU(inplace=True),
#                                      nn.Conv2d(64,64,3,2,1),nn.ReLU(inplace=True),
#                                      nn.Conv2d(64,64,3,1,1),nn.ReLU(inplace=True),
#                                      nn.Conv2d(64,64,3,2,1),nn.ReLU(inplace=True),
#                                      nn.Conv2d(64,128,3,1,1),nn.ReLU(inplace=True),
#                                      nn.Conv2d(128,128,3,1,1),nn.ReLU(inplace=True),
#                                      nn.Conv2d(128,128,3,2,1),nn.ReLU(inplace=True))
        self.features = nn.Sequential(nn.Conv2d(3,48,3,1,1),nn.BatchNorm2d(48),nn.ReLU(inplace=True),
                                      nn.Conv2d(48,48,3,2,1),nn.BatchNorm2d(48),nn.ReLU(inplace=True),
                                      nn.Conv2d(48,64,3,1,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                      nn.Conv2d(64,64,3,2,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                      nn.Conv2d(64,64,3,1,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                      nn.Conv2d(64,64,3,2,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                      nn.Conv2d(64,128,3,1,1),nn.BatchNorm2d(128),nn.ReLU(inplace=True),
                                      nn.Conv2d(128,128,3,1,1),nn.BatchNorm2d(128),nn.ReLU(inplace=True),
                                      nn.Conv2d(128,128,3,2,1),nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        weight_init(self.features)
        self.pooling = nn.AvgPool2d(14,1)
        self.projection = nn.Sequential(nn.Conv2d(128,256,1,1,0), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                                        nn.Conv2d(256,256,1,1,0), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        weight_init(self.projection)    
        self.classifier = nn.Linear(256,self.num_class)
        weight_init(self.classifier)

    def forward(self, X):
#        return X
        N = X.size()[0]
        assert X.size() == (N, 3, 224, 224)
        X = self.features(X)
        assert X.size() == (N, 128, 14, 14)
        X = self.pooling(X)
        assert X.size() == (N, 128, 1, 1)
        X = self.projection(X)
        X = X.view(X.size(0), -1)          
        X = self.classifier(X)
        assert X.size() == (N, self.num_class)
        return X


class DBCNN(torch.nn.Module):

    def __init__(self, scnn_root = './pretrained_scnn/scnn.pkl', fc_only = True):
        """Declare all needed layers."""
        nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.features1 = torchvision.models.vgg16(pretrained=True).features
        self.features1 = nn.Sequential(*list(self.features1.children())
                                            [:-1])
        scnn = SCNN()
        scnn = torch.nn.DataParallel(scnn).cuda()
              
        scnn.load_state_dict(torch.load(scnn_root))
        self.features2 = scnn.module.features
        
        # Linear classifier.
        self.fc = torch.nn.Linear(512*128, 1)
        
        if fc_only == True:
            # Freeze all previous layers.
            for param in self.features1.parameters():
                param.requires_grad = False
            for param in self.features2.parameters():
                param.requires_grad = False
            # Initialize the fc layers.
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)

        

    def forward(self, X):
        """Forward pass of the network.
        """
        N = X.size()[0]
        X1 = self.features1(X)
        H = X1.size()[2]
        W = X1.size()[3]
        assert X1.size()[1] == 512
        X2 = self.features2(X)
        H2 = X2.size()[2]
        W2 = X2.size()[3]
        assert X2.size()[1] == 128        
        
        if (H != H2) | (W != W2):
            X2 = F.upsample_bilinear(X2,(H,W))

        X1 = X1.view(N, 512, H*W)
        X2 = X2.view(N, 128, H*W)  
        X = torch.bmm(X1, torch.transpose(X2, 1, 2)) / (H*W)  # Bilinear
        assert X.size() == (N, 512, 128)
        X = X.view(N, 512*128)
        X = torch.sqrt(X + 1e-8)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)
        assert X.size() == (N, 1)
        return X