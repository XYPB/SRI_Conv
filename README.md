# Official Pypi Implementation of "SRI-Conv: Symmetric Rotation-Invariant Convolutional Kernel"
*Yuexi Du, Nicha, Dvornek, John Onofrey*

*Yale University*

**This is the initial official release of SRI_Conv**

version: 1.0.2

### Installation

```bash
pip install SRI-Conv
```


### Requirement:
```bash
"scipy>=1.9.1",
"numpy>=1.23.1",
"torch>=1.13.0"
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
```
