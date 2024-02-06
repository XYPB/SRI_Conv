# Official Pypi Implementation of "SRI-Conv: Symmetric Rotation-Invariant Convolutional Kernel"
*Yuexi Du, Nicha, Dvornek, John Onofrey*

*Yale University*

**This is the initial official release of SRI_Conv**

version: 1.2.0

### News

- [1.2.0] Gaussian Smooth Kernel is available now!
- [1.1.0] `SRI_Conv1d` and `SRI_Conv3d` are also available now.

### Installation

```bash
pip install SRI-Conv
```


### Requirement:
```bash
"scipy>=1.9.0",
"numpy>=1.22.0",
"torch>=1.8.0"
```

**Note**: Using lower version of torch and numpy should be fine given that we didn't use any new feature in the new torch version, but we do suggest you to follow the required dependencies. If you have to use the different version of torch/numpy, you may also try to install the package from source code at [project repo](https://github.com/XYPB/SRI_Conv).

### Usage
```python
>>> import torch
>>> from SRI_Conv import SRI_Conv2d, sri_resnet18
>>> x = torch.randn(2, 3, 32, 32)
>>> sri_conv = SRI_Conv2d(3, 16, 3)
>>> conv_out = sri_conv(x)
>>> sri_r18 = sri_resnet18()
>>> output = sri_r18(x)
# To reproduce the SRI-ResNet18 used in the paper, use:
>>> sri_r18 = sri_resnet18(ri_conv_size=[9, 9, 5, 5], skip_first_maxpool=True)
```
