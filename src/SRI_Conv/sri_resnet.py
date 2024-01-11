import math
from itertools import repeat
from typing import Type, Any, Callable, Union, List, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .sri_conv import SRI_Conv2d

def _repeat(x, t):
    if not isinstance(x, list):
        return list(repeat(x, t))
    elif len(x) == 1:
        return list(repeat(x[0], t))
    else:
        return x

def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

def sri_convkxk(k: int, in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
               , kernel_shape: str = 'o', train_index_mat: bool = False, bias: bool = False, ri_k: int = None,
               force_circular: bool = False) -> SRI_Conv2d:
    """3x3 RI convolution with padding"""
    return SRI_Conv2d(in_planes, out_planes, kernel_size=k, stride=stride,
                     padding=int(k//2), groups=groups, bias=bias, dilation=dilation,
                     kernel_shape=kernel_shape, train_index_mat=train_index_mat,
                     ri_k=ri_k, force_circular=force_circular)



class SRIBasicBlock(nn.Module):
    """
    Avoid the issue due to RIconv with stride differ to 1
    """
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        kernel_shape: str = 'o',
        train_index_mat: bool = False,
        ri_conv_size: int = 3,
        ri_groups: int = 1,
        ri_k: int = None,
        force_circular: bool = False,
        **kwargs: Any,
    ) -> None:
        super(SRIBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.stride = stride
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        conv1 = sri_convkxk(ri_conv_size, inplanes, planes, 
                            kernel_shape=kernel_shape, 
                            train_index_mat=train_index_mat, 
                            groups=ri_groups, ri_k=ri_k, 
                            force_circular=force_circular)
        if self.stride != 1:
            self.conv1 = nn.Sequential(
                nn.AvgPool2d(kernel_size=self.stride, stride=self.stride),
                conv1,
            )
        else:
            self.conv1 = conv1
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = sri_convkxk(ri_conv_size, planes, planes, kernel_shape=kernel_shape, 
                                train_index_mat=train_index_mat, groups=ri_groups,
                                ri_k=ri_k)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SRIBottleneck(nn.Module):
    """
    Symmetric Residual Bottleneck Block
    """

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        kernel_shape: str = 'o',
        train_index_mat: bool = False,
        ri_conv_size: int = 3,
        ri_groups: int = 1,
        ri_index_mat_cin: bool = False,
        ri_index_mat_cout: bool = False,
        ri_k: int = None,
        force_circular: bool = False,
        **kwargs: Any,
    ) -> None:
        super(SRIBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        ri_groups = width if ri_groups != 1 else 1
        self.stride = stride
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # conv1 and conv3 is the same in all case
        conv2 = sri_convkxk(ri_conv_size, width, width, stride=1, kernel_shape=kernel_shape, 
                            train_index_mat=train_index_mat, groups=ri_groups,
                            ri_index_mat_cin=ri_index_mat_cin, ri_index_mat_cout=ri_index_mat_cout,
                            ri_k=ri_k, force_circular=force_circular)
        if self.stride != 1:
            self.conv2 = nn.Sequential(
                nn.AvgPool2d(kernel_size=self.stride, stride=self.stride),
                conv2
            )
        else:
            self.conv2 = conv2
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SRI_ResNet(nn.Module):
    """
    Here we only introduce new parameters for SRI_ResNet.
    
    Args:
        large_conv: if True, use 5x5 conv in the first layer, otherwise use 3x3 conv.
        kernel_shape: shape of the SRI convolutional kernel. default: 'o'
        train_index_mat: if True, train the index matrix. default: False
        ri_conv_size: size of the SRI convolutional kernel, can be a list of length 4. default: 3
        deepwise_ri: if True, use deepwise SRI convolution. default: False
        inplanes: number of base channels. default: 64
        layer_stride: per-layer stride for each stage in the ResNet. default: [1, 2, 2, 2]
        ri_k: number of bands in the index matrix. default: None
        force_circle: if True, force the SRI weight matrix to be circular. default: False
        skip_first_maxpool: if True, skip the first maxpool layer, set True when processing small images. default: False
    """

    def __init__(
        self,
        block: Type[Union[SRIBasicBlock, SRIBottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        large_conv: bool = False,
        kernel_shape: str = 'o',
        train_index_mat: bool = False,
        ri_conv_size: Union[int, list] = 3,
        deepwise_ri: bool = False,
        inplanes: int = 64,
        layer_stride: Type[Union[int, List[int]]] = [1, 2, 2, 2],
        ri_k: Union[int, list] = None,
        force_circular: bool = False,
        skip_first_maxpool: bool = False,
        **kwargs,
    ) -> None:
        super(SRI_ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.large_conv = large_conv
        self.kernel_shape = kernel_shape
        self.train_index_mat = train_index_mat
        self.ri_conv_size = _repeat(ri_conv_size, 4)
        self.deepwise_ri = deepwise_ri
        self.ri_k = _repeat(ri_k, 4)
        self.force_circular = force_circular
        if isinstance(layer_stride, float):
            layer_stride = [layer_stride for _ in range(4)]
        else:
            layer_stride = layer_stride

        self.inplanes = inplanes
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = SRI_Conv2d(3, self.inplanes, kernel_size=5, stride=1, 
                                padding=2, bias=False, kernel_shape=kernel_shape, 
                                train_index_mat=train_index_mat, 
                                force_circular=force_circular)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        if skip_first_maxpool:
            self.maxpool = nn.Identity()
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, inplanes, layers[0], stride=layer_stride[0],
                                       ri_k=self.ri_k[0], ri_conv_size=self.ri_conv_size[0])
        self.layer2 = self._make_layer(block, 2*inplanes, layers[1], stride=layer_stride[1],
                                       dilate=replace_stride_with_dilation[0],
                                       ri_k=self.ri_k[1], ri_conv_size=self.ri_conv_size[1])
        self.layer3 = self._make_layer(block, 4*inplanes, layers[2], stride=layer_stride[2],
                                       dilate=replace_stride_with_dilation[1],
                                       ri_k=self.ri_k[2], ri_conv_size=self.ri_conv_size[2])
        self.layer4 = self._make_layer(block, 8*inplanes, layers[3], stride=layer_stride[3],
                                       dilate=replace_stride_with_dilation[2],
                                       ri_k=self.ri_k[3], ri_conv_size=self.ri_conv_size[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(8 * inplanes * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SRI_Conv2d):
                _, fan_out = nn.init._calculate_fan_in_and_fan_out(
                    torch.zeros(m.weight_matrix_shape)
                )
                gain = nn.init.calculate_gain('relu', 0)
                std = gain / math.sqrt(fan_out)
                nn.init.normal_(m.weight, 0, std)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, SRIBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, SRIBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[SRIBasicBlock, SRIBottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, ri_conv_size: int = 3, ri_k: int = None) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(stride, stride),
                conv1x1(self.inplanes, planes * block.expansion),
                norm_layer(planes * block.expansion),
            )

        layers = []
        ri_groups = planes if self.deepwise_ri else 1
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, 
                            kernel_shape=self.kernel_shape, train_index_mat=self.train_index_mat,
                            ri_conv_size=ri_conv_size, ri_groups=ri_groups,
                            large_conv=self.large_conv, ri_k=ri_k, force_circular=self.force_circular))
        self.inplanes = planes * block.expansion
        ri_groups = planes if self.deepwise_ri else 1
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_shape=self.kernel_shape, 
                                train_index_mat=self.train_index_mat,
                                ri_conv_size=ri_conv_size, ri_groups=ri_groups,
                                large_conv=self.large_conv,
 ri_k=ri_k, force_circular=self.force_circular))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[SRIBasicBlock, SRIBottleneck]],
    layers: List[int],
    **kwargs: Any
) -> SRI_ResNet:
    model = SRI_ResNet(block, layers, **kwargs)
    return model


def sri_resnet18(**kwargs: Any) -> SRI_ResNet:
    r"""SRI_ResNet-18 model from
    'SRI-Conv: Symmetric Rotation-Invariant Convolutional Kernel'
    """
    return _resnet('sri_resnet18', SRIBasicBlock, [2, 2, 2, 2], **kwargs)

def sri_resnet50(**kwargs: Any) -> SRI_ResNet:
    r"""SRI_ResNet-50 model from
    'SRI-Conv: Symmetric Rotation-Invariant Convolutional Kernel'
    """
    return _resnet('sri_resnet50', SRIBottleneck, [3, 4, 6, 3], **kwargs)