import torch.nn as nn
from .sri_conv import SRI_Conv2d

def convert_to_sri_conv(model, kernel_shape='o', train_index_mat=False, ri_conv_size=5, ri_groups=1, ri_k=None, force_circular=False):
    '''
    Recursively replace Conv layers to SRI_Conv layers.
    If convolutional layer with stride >= 2 found, add AvgPool2d before it.
    '''
    state = {'counter': 0}

    def _replace_handler(module, state):
        for attr, target in module.named_children():
            if type(target) == nn.Conv2d:
                layer = SRI_Conv2d(
                    target.in_channels,
                    target.out_channels,
                    kernel_size=target.kernel_size,
                    stride=1,
                    padding=target.padding,
                    dilation=target.dilation,
                    groups=target.groups,
                    bias=target.bias is not None,
                    kernel_shape=kernel_shape,
                    train_index_mat=train_index_mat,
                    ri_conv_size=ri_conv_size,
                    ri_groups=ri_groups,
                    ri_k=ri_k,
                    force_circular=force_circular)
                if target.stride[0] > 1 or target.stride[1] > 1:
                    avg_pool = nn.AvgPool2d(target.stride, target.stride)
                    layer = nn.Sequential(avg_pool, layer)
                state['counter'] += 1
            _replace_handler(target, state)

    _replace_handler(model, state)