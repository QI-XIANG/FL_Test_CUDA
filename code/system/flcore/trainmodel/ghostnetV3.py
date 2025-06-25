# 2020.06.09-Changed for building GhostNet
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch and https://github.com/rwightman/pytorch-image-models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Optional, List, Tuple
from timm.models.registry import register_model

#__all__ = ['ghost_net']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x    

    
class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x
    
    
def gcd(a,b):
    if a<b:
        a,b=b,a
    while(a%b != 0):
        c = a%b
        a=b
        b=c
    return b
    
def MyNorm(dim):
    return nn.GroupNorm(1, dim)  

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True,mode=None,args=None):
        super(GhostModule, self).__init__()
        #self.args=args
        self.mode = mode
        self.gate_loc = 'before'
        
        self.inter_mode = 'nearest'
        self.scale = 1.0
        
        self.infer_mode = False
        self.num_conv_branches = 3
        self.dconv_scale = True
        self.gate_fn = nn.Sigmoid()

        # if args.gate_fn=='hard_sigmoid':
        #     self.gate_fn=hard_sigmoid
        # elif args.gate_fn=='sigmoid': 
        #     self.gate_fn=nn.Sigmoid()
        # elif args.gate_fn=='relu': 
        #     self.gate_fn=nn.ReLU()
        # elif args.gate_fn=='clip': 
        #     self.gate_fn=myclip 
        # elif args.gate_fn=='tanh': 
        #     self.gate_fn=nn.Tanh()

        if self.mode in ['ori']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio) 
            new_channels = init_channels*(ratio-1)
            if self.infer_mode:  
                self.primary_conv = nn.Sequential(  
                    nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                    nn.BatchNorm2d(init_channels),
                    nn.ReLU(inplace=True) if relu else nn.Sequential(),
                )
                self.cheap_operation = nn.Sequential(
                    nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                    nn.BatchNorm2d(new_channels),
                    nn.ReLU(inplace=True) if relu else nn.Sequential(),
                )
            else:
                self.primary_rpr_skip = nn.BatchNorm2d(inp) \
                    if inp == init_channels and stride == 1 else None
                primary_rpr_conv = list()
                for _ in range(self.num_conv_branches):
                    primary_rpr_conv.append(self._conv_bn(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False))
                self.primary_rpr_conv = nn.ModuleList(primary_rpr_conv)
                # Re-parameterizable scale branch
                self.primary_rpr_scale = None
                if kernel_size > 1:
                    self.primary_rpr_scale = self._conv_bn(inp, init_channels, 1, 1, 0, bias=False)
                self.primary_activation = nn.ReLU(inplace=True) if relu else None


                self.cheap_rpr_skip = nn.BatchNorm2d(init_channels) \
                    if init_channels == new_channels else None
                cheap_rpr_conv = list()
                for _ in range(self.num_conv_branches):
                    cheap_rpr_conv.append(self._conv_bn(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False))
                self.cheap_rpr_conv = nn.ModuleList(cheap_rpr_conv)
                # Re-parameterizable scale branch
                self.cheap_rpr_scale = None
                if dw_size > 1:
                    self.cheap_rpr_scale = self._conv_bn(init_channels, new_channels, 1, 1, 0, groups=init_channels, bias=False)
                self.cheap_activation = nn.ReLU(inplace=True) if relu else None
                self.in_channels = init_channels
                self.groups = init_channels
                self.kernel_size = dw_size
     
        elif self.mode in ['ori_shortcut_mul_conv15']: 
            self.oup = oup
            init_channels = math.ceil(oup / ratio) 
            new_channels = init_channels*(ratio-1)
            self.short_conv = nn.Sequential( 
                nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(1,5), stride=1, padding=(0,2), groups=oup,bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5,1), stride=1, padding=(2,0), groups=oup,bias=False),
                nn.BatchNorm2d(oup),
            )
            if self.infer_mode:
                self.primary_conv = nn.Sequential(  
                    nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                    nn.BatchNorm2d(init_channels),
                    nn.ReLU(inplace=True) if relu else nn.Sequential(),
                )
                self.cheap_operation = nn.Sequential(
                    nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                    nn.BatchNorm2d(new_channels),
                    nn.ReLU(inplace=True) if relu else nn.Sequential(),
                ) 
            else:
                self.primary_rpr_skip = nn.BatchNorm2d(inp) \
                    if inp == init_channels and stride == 1 else None
                primary_rpr_conv = list()
                for _ in range(self.num_conv_branches):
                    primary_rpr_conv.append(self._conv_bn(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False))
                self.primary_rpr_conv = nn.ModuleList(primary_rpr_conv)
                # Re-parameterizable scale branch
                self.primary_rpr_scale = None
                if kernel_size > 1:
                    self.primary_rpr_scale = self._conv_bn(inp, init_channels, 1, 1, 0, bias=False)
                self.primary_activation = nn.ReLU(inplace=True) if relu else None


                self.cheap_rpr_skip = nn.BatchNorm2d(init_channels) \
                    if init_channels == new_channels else None
                cheap_rpr_conv = list()
                for _ in range(self.num_conv_branches):
                    cheap_rpr_conv.append(self._conv_bn(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False))
                self.cheap_rpr_conv = nn.ModuleList(cheap_rpr_conv)
                # Re-parameterizable scale branch
                self.cheap_rpr_scale = None
                if dw_size > 1:
                    self.cheap_rpr_scale = self._conv_bn(init_channels, new_channels, 1, 1, 0, groups=init_channels, bias=False)
                self.cheap_activation = nn.ReLU(inplace=True) if relu else None
                self.in_channels = init_channels
                self.groups = init_channels
                self.kernel_size = dw_size

      
    def forward(self, x):
        if self.mode in ['ori']:
            if self.infer_mode:
                x1 = self.primary_conv(x)
                x2 = self.cheap_operation(x1)
            else:
                identity_out = 0
                if self.primary_rpr_skip is not None:
                    identity_out = self.primary_rpr_skip(x)
                scale_out = 0
                if self.primary_rpr_scale is not None and self.dconv_scale:
                    scale_out = self.primary_rpr_scale(x)
                x1 = scale_out + identity_out
                for ix in range(self.num_conv_branches):
                    x1 += self.primary_rpr_conv[ix](x)
                if self.primary_activation is not None:
                    x1 = self.primary_activation(x1)

                cheap_identity_out = 0
                if self.cheap_rpr_skip is not None:
                    cheap_identity_out = self.cheap_rpr_skip(x1)
                cheap_scale_out = 0
                if self.cheap_rpr_scale is not None and self.dconv_scale:
                    cheap_scale_out = self.cheap_rpr_scale(x1)
                x2 = cheap_scale_out + cheap_identity_out
                for ix in range(self.num_conv_branches):
                    x2 += self.cheap_rpr_conv[ix](x1)
                if self.cheap_activation is not None:
                    x2 = self.cheap_activation(x2)

            out = torch.cat([x1,x2], dim=1)
            return out

        elif self.mode in ['ori_shortcut_mul_conv15']:  
            res=self.short_conv(F.avg_pool2d(x,kernel_size=2,stride=2))
            
            if self.infer_mode:
                x1 = self.primary_conv(x)
                x2 = self.cheap_operation(x1)
            else:
                identity_out = 0
                if self.primary_rpr_skip is not None:
                    identity_out = self.primary_rpr_skip(x)
                scale_out = 0
                if self.primary_rpr_scale is not None and self.dconv_scale:
                    scale_out = self.primary_rpr_scale(x)
                x1 = scale_out + identity_out
                for ix in range(self.num_conv_branches):
                    x1 += self.primary_rpr_conv[ix](x)
                if self.primary_activation is not None:
                    x1 = self.primary_activation(x1)

                cheap_identity_out = 0
                if self.cheap_rpr_skip is not None:
                    cheap_identity_out = self.cheap_rpr_skip(x1)
                cheap_scale_out = 0
                if self.cheap_rpr_scale is not None and self.dconv_scale:
                    cheap_scale_out = self.cheap_rpr_scale(x1)
                x2 = cheap_scale_out + cheap_identity_out
                for ix in range(self.num_conv_branches):
                    x2 += self.cheap_rpr_conv[ix](x1)
                if self.cheap_activation is not None:
                    x2 = self.cheap_activation(x2)

            out = torch.cat([x1,x2], dim=1)

            if self.gate_loc=='before':
                return out[:,:self.oup,:,:]*F.interpolate(self.gate_fn(res/self.scale),size=out.shape[-2:],mode=self.inter_mode) # 'nearest'
#                 return out*F.interpolate(self.gate_fn(res/self.scale),size=out.shape[-1].item(),mode=self.inter_mode) # 'nearest'
            else:
                return out[:,:self.oup,:,:]*self.gate_fn(F.interpolate(res,size=out.shape[-2:],mode=self.inter_mode))  
#                 return out*self.gate_fn(F.interpolate(res,size=out.shape[-1],mode=self.inter_mode))  


    def reparameterize(self):
        """ Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.infer_mode:
            return
        primary_kernel, primary_bias = self._get_kernel_bias_primary()
        self.primary_conv = nn.Conv2d(in_channels=self.primary_rpr_conv[0].conv.in_channels,
                                      out_channels=self.primary_rpr_conv[0].conv.out_channels,
                                      kernel_size=self.primary_rpr_conv[0].conv.kernel_size,
                                      stride=self.primary_rpr_conv[0].conv.stride,
                                      padding=self.primary_rpr_conv[0].conv.padding,
                                      dilation=self.primary_rpr_conv[0].conv.dilation,
                                      groups=self.primary_rpr_conv[0].conv.groups,
                                      bias=True)
        self.primary_conv.weight.data = primary_kernel
        self.primary_conv.bias.data = primary_bias
        self.primary_conv = nn.Sequential(
            self.primary_conv, 
            self.primary_activation if self.primary_activation is not None else nn.Sequential()
        )

        cheap_kernel, cheap_bias = self._get_kernel_bias_cheap()
        self.cheap_operation = nn.Conv2d(in_channels=self.cheap_rpr_conv[0].conv.in_channels,
                                      out_channels=self.cheap_rpr_conv[0].conv.out_channels,
                                      kernel_size=self.cheap_rpr_conv[0].conv.kernel_size,
                                      stride=self.cheap_rpr_conv[0].conv.stride,
                                      padding=self.cheap_rpr_conv[0].conv.padding,
                                      dilation=self.cheap_rpr_conv[0].conv.dilation,
                                      groups=self.cheap_rpr_conv[0].conv.groups,
                                      bias=True)
        self.cheap_operation.weight.data = cheap_kernel
        self.cheap_operation.bias.data = cheap_bias

        self.cheap_operation = nn.Sequential(
            self.cheap_operation, 
            self.cheap_activation if self.cheap_activation is not None else nn.Sequential()
        )

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        if hasattr(self, 'primary_rpr_conv'):
            self.__delattr__('primary_rpr_conv')
        if hasattr(self, 'primary_rpr_scale'):
            self.__delattr__('primary_rpr_scale')
        if hasattr(self, 'primary_rpr_skip'):
            self.__delattr__('primary_rpr_skip')

        if hasattr(self, 'cheap_rpr_conv'):
            self.__delattr__('cheap_rpr_conv')
        if hasattr(self, 'cheap_rpr_scale'):
            self.__delattr__('cheap_rpr_scale')
        if hasattr(self, 'cheap_rpr_skip'):
            self.__delattr__('cheap_rpr_skip')

        self.infer_mode = True

    def _get_kernel_bias_primary(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.primary_rpr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.primary_rpr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.primary_rpr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.primary_rpr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.primary_rpr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final
    
    def _get_kernel_bias_cheap(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.cheap_rpr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.cheap_rpr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.cheap_rpr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.cheap_rpr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.cheap_rpr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                 self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
        """ Helper method to construct conv-batchnorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              groups=groups,
                                              bias=bias))
        mod_list.add_module('bn', nn.BatchNorm2d(out_channels))
        return mod_list


class GhostBottleneck(nn.Module): 
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.,layer_id=None,args=None):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        self.num_conv_branches = 3
        self.infer_mode = False
        self.dconv_scale = True

        # Point-wise expansion
        if layer_id<=1:
            self.ghost1 = GhostModule(in_chs, mid_chs, relu=True,mode='ori',args=args)
        else:
            self.ghost1 = GhostModule(in_chs, mid_chs, relu=True,mode='ori_shortcut_mul_conv15',args=args) ####这里是扩张 mid_chs远大于in_chs

        # Depth-wise convolution
        if self.stride > 1:
            if self.infer_mode:
                self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                 padding=(dw_kernel_size-1)//2,
                                 groups=mid_chs, bias=False)
                self.bn_dw = nn.BatchNorm2d(mid_chs)
            else:
                self.dw_rpr_skip = nn.BatchNorm2d(mid_chs) if stride == 1 else None
                dw_rpr_conv = list()
                for _ in range(self.num_conv_branches):
                    dw_rpr_conv.append(self._conv_bn(mid_chs, mid_chs, dw_kernel_size, stride, (dw_kernel_size-1)//2, groups=mid_chs, bias=False))
                self.dw_rpr_conv = nn.ModuleList(dw_rpr_conv)
                # Re-parameterizable scale branch
                self.dw_rpr_scale = None
                if dw_kernel_size > 1:
                    self.dw_rpr_scale = self._conv_bn(mid_chs, mid_chs, 1, 2, 0, groups=mid_chs, bias=False)
                self.kernel_size = dw_kernel_size
                self.in_channels = mid_chs

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        if layer_id<=1:
            self.ghost2 = GhostModule(mid_chs, out_chs, relu=False,mode='ori',args=args)
        else:
            self.ghost2 = GhostModule(mid_chs, out_chs, relu=False,mode='ori',args=args)
        
        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            if self.infer_mode:
                x = self.conv_dw(x)
                x = self.bn_dw(x)
            else:
                dw_identity_out = 0
                if self.dw_rpr_skip is not None:
                    dw_identity_out = self.dw_rpr_skip(x)
                dw_scale_out = 0
                if self.dw_rpr_scale is not None and self.dconv_scale:
                    dw_scale_out = self.dw_rpr_scale(x)
                x1 = dw_scale_out + dw_identity_out
                for ix in range(self.num_conv_branches):
                    x1 += self.dw_rpr_conv[ix](x)
                x = x1

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)
        
        x += self.shortcut(residual)
        return x

    def _conv_bn(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
        """ Helper method to construct conv-batchnorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              groups=groups,
                                              bias=bias))
        mod_list.add_module('bn', nn.BatchNorm2d(out_channels))
        return mod_list


    def reparameterize(self):
        """ Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.infer_mode or self.stride == 1:
            return
        dw_kernel, dw_bias = self._get_kernel_bias_dw()
        self.conv_dw = nn.Conv2d(in_channels=self.dw_rpr_conv[0].conv.in_channels,
                                      out_channels=self.dw_rpr_conv[0].conv.out_channels,
                                      kernel_size=self.dw_rpr_conv[0].conv.kernel_size,
                                      stride=self.dw_rpr_conv[0].conv.stride,
                                      padding=self.dw_rpr_conv[0].conv.padding,
                                      dilation=self.dw_rpr_conv[0].conv.dilation,
                                      groups=self.dw_rpr_conv[0].conv.groups,
                                      bias=True)
        self.conv_dw.weight.data = dw_kernel
        self.conv_dw.bias.data = dw_bias
        self.bn_dw = nn.Identity()

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        if hasattr(self, 'dw_rpr_conv'):
            self.__delattr__('dw_rpr_conv')
        if hasattr(self, 'dw_rpr_scale'):
            self.__delattr__('dw_rpr_scale')
        if hasattr(self, 'dw_rpr_skip'):
            self.__delattr__('dw_rpr_skip')

        self.infer_mode = True

    def _get_kernel_bias_dw(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.dw_rpr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.dw_rpr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.dw_rpr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.dw_rpr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.dw_rpr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final


    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                 self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width=1.0, dropout=0.2, block=GhostBottleneck, args=None):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        #block = block
        layer_id=0
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                if block==GhostBottleneck:
                    layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                  se_ratio=se_ratio,layer_id=layer_id,args=args))
                input_channel = output_channel
                layer_id+=1
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel
        
        self.blocks = nn.Sequential(*stages)        

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        # if self.dropout > 0.:
            # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        x = x.squeeze()
        return x

    def reparameterize(self):
        for _, module in self.named_modules():
            if isinstance(module, GhostModule):
                module.reparameterize()
            if isinstance(module, GhostBottleneck):
                module.reparameterize()

@register_model
def ghostnetv3(**kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k, t, c, SE, s 
        # stage1
        [[3,  16,  16, 0, 1]],
        # stage2
        [[3,  48,  24, 0, 2]],
        [[3,  72,  24, 0, 1]],
        # stage3
        [[5,  72,  40, 0.25, 2]],
        [[5, 120,  40, 0.25, 1]],
        # stage4
        [[3, 240,  80, 0, 2]],
        [[3, 200,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
        ],
        # stage5
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
        ]
    ]
    return GhostNet(cfgs, num_classes=1000, width=kwargs['width'], dropout=0.2)

if __name__=='__main__':
    model = ghostnetv3(width=1.0)
    model.eval()
    print(model)
    input1 = torch.randn(32,3,320,256)
    input2 = torch.randn(32,3,256,320)
    input3 = torch.randn(32,3,224,224)

    with torch.inference_mode():
        y11 = model(input1)
        y12 = model(input2)
        y13 = model(input3)
    model.reparameterize()
    print(model)
    with torch.inference_mode():
        y21 = model(input1)
        y22 = model(input2)
        y23 = model(input3)
    print(torch.allclose(y11, y21), torch.norm(y11 - y21))
    print(torch.allclose(y12, y22), torch.norm(y12 - y22))
    print(torch.allclose(y13, y23), torch.norm(y13 - y23))

#-------------------------------------------------------------------------------------------------------------------

class FedProtoCINIC10_GhostNetV3(nn.Module):
    """
    Optimized Neural Network Model for Federated Learning on CINIC-10,
    using a pre-trained **GhostNet** backbone as a **frozen feature extractor**
    and a lightweight, trainable classification head.

    This model leverages a custom GhostNet implementation, adapting it for
    small image datasets and the unique requirements of federated learning.

    It's designed to:
    1.  **Reduce Computational Overhead:** By freezing the majority of parameters (the backbone),
        client-side training only involves updating a small classification head. This significantly
        lowers FLOPs, memory usage, and communication costs.
    2.  **Leverage Pre-training (if available):** While this specific GhostNet code doesn't
        directly include pre-trained weights loading, in a real scenario, you would load them
        to benefit from rich features learned on large datasets like ImageNet.
    3.  **Improve Performance on Small Images:** Adapts the initial convolutional layer for 32x32
        inputs to preserve more spatial information.
    4.  **Mitigate Non-IID Issues:** A smaller trainable part is less prone to overfitting
        small, non-IID client datasets, leading to more stable aggregation.

    為 CINIC-10 設計的優化神經網路模型，用於聯邦學習。
    此模型使用預訓練的 **GhostNet** 主幹作為**凍結的特徵提取器**，
    並帶有一個輕量級、可訓練的分類頭。

    它的設計目標是：
    1.  **降低計算開銷：** 通過凍結大部分參數（主幹），
        客戶端訓練僅涉及更新一個小型分類頭。這顯著降低了 FLOPs、內存使用量和通信成本。
    2.  **利用預訓練（如果可用）：** 雖然此 GhostNet 代碼不直接包含預訓練權重加載，
        但在實際情況中，您會加載它們以受益於在大型數據集（如 ImageNet）上學習到的豐富特徵。
    3.  **提高小圖像性能：** 調整初始卷積層以適應 32x32 輸入，以保留更多空間信息。
    4.  **緩解非獨立同分佈問題：** 較小的可訓練部分不易在小型、非獨立同分佈的客戶端數據集上過擬合，
        從而實現更穩定的聚合。
    """
    def __init__(self, num_classes=10):
        """
        Initializes the FedProtoCINIC10_GhostNetV3 with a GhostNet backbone.

        Parameters:
            num_classes (int): The number of output classes. For CINIC-10, this is 10.
        """
        super(FedProtoCINIC10_GhostNetV3, self).__init__()

        # --- Instantiate the GhostNet backbone ---
        # The provided GhostNet code has a 'ghostnetv3' function.
        # You can choose a 'width' multiplier, e.g., 1.0 for the standard model.
        # If you have specific pre-trained weights (e.g., a .pth file), you would load them here.
        # Example (if you have a state_dict):
        # backbone = GhostNet(cfgs=...) # initialize without kwargs that might conflict with loading
        # state_dict = torch.load('path/to/your/ghostnet_weights.pth', map_location='cpu')
        # backbone.load_state_dict(state_dict)

        # For this example, we'll use the 'ghostnetv3' function with default width=1.0.
        # Note: The `ghostnetv3` function in your provided code takes `width` as a keyword argument.
        # If you have pre-trained weights, ensure the `width` matches the pre-trained model.
        
        # We need to define cfgs as it's passed to GhostNet, and it's defined in the original `ghostnetv3` function
        # from the file you provided.
        cfgs = [
            # k, t, c, SE, s
            # stage1
            [[3,  16,  16, 0, 1]],
            # stage2
            [[3,  48,  24, 0, 2]],
            [[3,  72,  24, 0, 1]],
            # stage3
            [[5,  72,  40, 0.25, 2]],
            [[5, 120,  40, 0.25, 1]],
            # stage4
            [[3, 240,  80, 0, 2]],
            [[3, 200,  80, 0, 1],
             [3, 184,  80, 0, 1],
             [3, 184,  80, 0, 1],
             [3, 480, 112, 0.25, 1],
             [3, 672, 112, 0.25, 1]
            ],
            # stage5
            [[5, 672, 160, 0.25, 2]],
            [[5, 960, 160, 0, 1],
             [5, 960, 160, 0.25, 1],
             [5, 960, 160, 0, 1],
             [5, 960, 160, 0.25, 1]
            ]
        ]
        
        # Instantiate GhostNet. If you have pre-trained weights, load them *after* instantiation.
        backbone = GhostNet(cfgs=cfgs, num_classes=1000, width=1.0, dropout=0.2)
        print("GhostNet backbone initialized.")

        # --- Adaptation for 32x32 input ---
        # The `conv_stem` of GhostNet typically has a stride of 2.
        # For small 32x32 images, we need to reduce this to stride=1 to avoid aggressive downsampling.
        # This modification re-initializes the weights of this specific convolutional layer.
        
        original_conv_stem = backbone.conv_stem
        
        # Create a new Conv2d layer with stride=1
        new_conv_stem = nn.Conv2d(
            in_channels=original_conv_stem.in_channels,
            out_channels=original_conv_stem.out_channels,
            kernel_size=original_conv_stem.kernel_size,
            stride=1,  # Key change: Reduce stride from 2 to 1 for small inputs
            padding=original_conv_stem.padding,
            bias=original_conv_stem.bias
        )
        # Replace the original conv_stem
        backbone.conv_stem = new_conv_stem
        
        # If there's a pooling layer immediately after conv_stem (like a MaxPool2d),
        # you might need to remove or modify it too for small images.
        # In the provided GhostNet code, it's followed by bn1 and act1, not a pooling layer.
        # So no explicit maxpool removal is needed here.

        print("GhostNet: Adapted initial conv_stem stride for 32x32 inputs (no resizing).")

        # --- Freeze the entire feature extractor backbone ---
        # This is CRITICAL for reducing client-side computational load.
        # Clients will only train the much smaller classification head.
        # The feature extractor comprises `conv_stem`, `bn1`, `act1`, and `blocks` from the GhostNet.
        # The `conv_head` is typically the last feature aggregation layer before the final classifier,
        # so we include it in the frozen feature extractor.

        # 凍結整個特徵提取器主幹。
        # 這對於降低客戶端計算負載至關重要。
        # 客戶端將只訓練小得多的分類頭。
        # 特徵提取器包括 GhostNet 的 `conv_stem`、`bn1`、`act1` 和 `blocks`。
        # `conv_head` 通常是最終分類器之前的最後一個特徵聚合層，
        # 因此我們將其包含在凍結的特徵提取器中。
        self.feature_extractor = nn.Sequential(
            backbone.conv_stem,
            backbone.bn1,
            backbone.act1,
            backbone.blocks,
            backbone.global_pool,
            backbone.conv_head,
            backbone.act2
        )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # Set requires_grad to False to freeze parameters
        print("GhostNet backbone (feature_extractor) frozen.")

        # --- Define the new, trainable classification head ---
        # The original GhostNet classifier is `backbone.classifier`.
        # We replace it with a new linear layer suitable for our `num_classes`.
        # This is the only part of the model whose parameters will be updated during client training.

        # 定義新的、可訓練的分類頭。
        # 原始 GhostNet 分類器是 `backbone.classifier`。
        # 我們將用一個適合我們 `num_classes` 的新線性層替換它。
        # 這是模型中唯一在客戶端訓練期間更新其參數的部分。

        # Get the input feature dimension for the new classifier from the original model's last linear layer.
        feature_dim = backbone.classifier.in_features
        self.classifier = nn.Linear(feature_dim, num_classes)
        print(f"New classifier initialized with {feature_dim} input features and {num_classes} output classes.")

    def forward(self, x):
        """
        Performs a forward pass through the network, producing classification logits.

        Parameters:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: The classification logits from the model.
        """
        # Pass input through the frozen feature extractor (inference mode for these layers)
        # Even if the model is in training mode, these frozen layers should behave like evaluation mode.
        # 即使模型在訓練模式，這些凍結層也應行為如評估模式。
        self.feature_extractor.eval()
        with torch.no_grad():  # Ensure no gradients are computed for the frozen backbone
            x = self.feature_extractor(x)

        # The global_pool, conv_head, and act2 are already part of the feature_extractor
        # and should output a flat feature vector suitable for the classifier.
        # Based on the GhostNet forward pass, after conv_head and act2, it's flattened.
        # x = x.view(x.size(0), -1) # This flattening is handled implicitly by the GhostNet's original forward up to conv_head/act2
        
        # Ensure the tensor is flattened before passing to the final linear layer
        # The output of self.feature_extractor (which includes global_pool, conv_head, act2)
        # should already be `(batch_size, output_channel)`.
        # If it's still (batch_size, channels, 1, 1), flatten it.
        if x.dim() == 4 and x.shape[2] == 1 and x.shape[3] == 1:
             x = x.view(x.size(0), -1)
        
        # Pass the flattened features through the trainable classification layer.
        # 將展平後的特徵通過可訓練的分類層。
        x = self.classifier(x)
        return x

    def get_features(self, x):
        """
        Extracts features (embeddings) from the input before the final classification layer.
        This method is highly useful in prototype-based federated learning (like FedProto)
        where class prototypes (centroids) are computed from these feature embeddings
        to mitigate issues like non-IID data and class imbalance.

        Parameters:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: The feature embeddings extracted from the input images.
        """
        # Ensure the feature extractor is in evaluation mode for consistent feature extraction
        # (e.g., BatchNorm layers use global mean/variance).
        # 確保特徵提取器處於評估模式以進行一致的特徵提取
        # (例如，BatchNorm 層使用全局均值/方差)。
        self.feature_extractor.eval()
        with torch.no_grad():  # No gradients needed for feature extraction
            x = self.feature_extractor(x)

        # The feature_extractor already includes global_pool and conv_head/act2,
        # which prepares the output for the classifier.
        # Ensure the tensor is flattened before returning as features.
        if x.dim() == 4 and x.shape[2] == 1 and x.shape[3] == 1:
            x = x.view(x.size(0), -1)
            
        return x