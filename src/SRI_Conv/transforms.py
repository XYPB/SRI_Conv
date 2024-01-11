import torchvision.transforms.functional as F

class PadRotateWrapper(object):
    def __init__(self, rotate_trans, padding="zeros", img_size=32):
        super().__init__()
        self.rotate_trans = rotate_trans
        self.padding_mode = padding
        if padding != "zeros":
            self.to_pad = int(img_size // 2)
        else:
            self.to_pad = None
        self.img_size = img_size
        
    def __call__(self, x):
        # pad the image before rotate
        if self.to_pad != None:
            x = F.pad(x, (self.to_pad, self.to_pad, self.to_pad, self.to_pad), padding_mode=self.padding_mode)
        x = self.rotate_trans(x)
        # crop the image after rotate
        if self.to_pad != None:
            x = F.center_crop(x, [self.img_size, self.img_size])
        return x

class FixRotate(object):
    def __init__(self, degree, expand=False):
        super().__init__()
        self.degree = degree
        self.expand = expand

    def __call__(self, x):
        return F.rotate(x, self.degree, expand=self.expand)