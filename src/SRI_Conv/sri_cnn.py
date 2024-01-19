from itertools import repeat
import torch
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
    

class SRI_Net(nn.Module):
    def __init__(self, in_channels, n_classes, base_channels=64, ri_shape='o', 
                 kernel_size=3, dropout_p=-1, train_index_mat=False, 
                 ri_k=None, **kwargs):
        super(SRI_Net, self).__init__()
        kernel_size = _repeat(kernel_size, 5)
        ri_k = _repeat(ri_k, 5)
        hidden_channels = [base_channels * k for k in [1, 2, 4, 4, 8]]

        self.layer1 = nn.Sequential(
            SRI_Conv2d(in_channels, hidden_channels[0], kernel_size=kernel_size[0], 
                      padding=kernel_size[0]//2, kernel_shape=ri_shape, 
                      train_index_mat=train_index_mat, ri_k=ri_k[0]),
            nn.BatchNorm2d(hidden_channels[0]),
            nn.ReLU()) # 28

        self.layer2 = nn.Sequential(
            SRI_Conv2d(hidden_channels[0], hidden_channels[1], kernel_size=kernel_size[1], 
                      padding=kernel_size[1]//2, kernel_shape=ri_shape,
                      train_index_mat=train_index_mat, ri_k=ri_k[1]),
            nn.BatchNorm2d(hidden_channels[1]),
            nn.ReLU(),
            SRI_Conv2d(hidden_channels[1], hidden_channels[1], kernel_size=kernel_size[1], 
                      padding=kernel_size[1]//2, kernel_shape=ri_shape,
                      train_index_mat=train_index_mat, ri_k=ri_k[1]),
            nn.BatchNorm2d(hidden_channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # 14

        self.layer3 = nn.Sequential(
            SRI_Conv2d(hidden_channels[1], hidden_channels[2], kernel_size=kernel_size[2], 
                      padding=kernel_size[2]//2, kernel_shape=ri_shape,
                      train_index_mat=train_index_mat, ri_k=ri_k[2]),
            nn.BatchNorm2d(hidden_channels[2]),
            nn.ReLU()) # 14
        
        self.layer4 = nn.Sequential(
            SRI_Conv2d(hidden_channels[2], hidden_channels[3], kernel_size=kernel_size[3], 
                      padding=max((kernel_size[3]//2)-1, 0), kernel_shape=ri_shape,
                      train_index_mat=train_index_mat, ri_k=ri_k[3]),
            nn.BatchNorm2d(hidden_channels[3]),
            nn.ReLU()) # 12

        self.layer5 = nn.Sequential(
            SRI_Conv2d(hidden_channels[3], hidden_channels[4], kernel_size=kernel_size[4], 
                      padding=max((kernel_size[4]//2)-1, 0), kernel_shape=ri_shape,
                      train_index_mat=train_index_mat, ri_k=ri_k[4]),
            nn.BatchNorm2d(hidden_channels[4]), 
            nn.ReLU(), # 10
            SRI_Conv2d(hidden_channels[4], hidden_channels[4], kernel_size=kernel_size[4], 
                      padding=kernel_size[4]//2, kernel_shape=ri_shape,
                      train_index_mat=train_index_mat, ri_k=ri_k[4]),
            nn.BatchNorm2d(hidden_channels[4]),
            nn.ReLU()) # 10
        
        self.dropout_p = dropout_p
        if dropout_p > 0:
            self.dropout3 = nn.Dropout(dropout_p)
            self.dropout4 = nn.Dropout(dropout_p)
            self.dropout5 = nn.Dropout(dropout_p)
        else:
            self.dropout3 = nn.Identity()
            self.dropout4 = nn.Identity()
            self.dropout5 = nn.Identity()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)

        self.fc = nn.Sequential(
            nn.Linear(hidden_channels[4], n_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.dropout3(x)
        x = self.layer4(x)
        x = self.dropout4(x)
        x = self.layer5(x)
        x = self.dropout5(x)

        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    