# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 18:15:01 2019

@author: abobashe
"""
import torch
import torchvision.transforms as transforms

import numpy as np
from PIL import Image

imagenet_mean = [ 0.485, 0.456, 0.406 ]
imagenet_std  = [ 0.229, 0.224, 0.225 ]

joconde_mean_domains = [ 0.5966,  0.5625,  0.5044] 
joconde_std_domains = [ 0.2784,  0.2776,  0.2726] 

joconde_mean_animals = [ 0.5990,  0.5556,  0.5030]
joconde_std_animals = [ 0.2799,  0.2832,  0.2895]

class NormalizeMultiCrop(transforms.Normalize):
    """Normalize a tensor of images for multiple crops with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (N, C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor of images.
        """
        assert(tensor.dim() == 4)
        
        tlist = [transforms.functional.normalize(m, self.mean, self.std) for m in torch.unbind(tensor, dim=0) ]
        res = torch.stack(tlist, dim=0)
        
        return res

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    

class PadToSquare(object):
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (int, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric', 'wrap', 'circle']


        self.fill = fill
        self.padding_mode = padding_mode
        

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        w, h = img.size

        padding = 0
        
        if (self.padding_mode in ['constant', 'edge', 'reflect', 'symmetric', 'circle']):
            if (w > h):
                padding = (0 , (w-h) // 2)
            elif (w < h):
                padding = ((h-w) // 2, 0)
          
        if (self.padding_mode in ['wrap']):
            if (w > h):
                padding = (0, 0, 0, (w-h))
            elif (w < h):
                padding = (0, 0, (h-w), 0)     

        if (self.padding_mode in ['constant', 'edge', 'reflect', 'symmetric']):        
            return transforms.functional.pad(img, padding, self.fill, self.padding_mode)
        
        if (self.padding_mode in ['wrap', 'circle']):           
            return wrap(img, padding)

    def __repr__(self):
        return self.__class__.__name__ + ' fill={1}, padding_mode={2})'.\
            format(self.fill, self.padding_mode)

#TODO: move under the classs
def wrap(img, padding):
    r"""Wrap the given PIL Image.

    Args:
        img (PIL Image): Image to be padded.
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
    Returns:
        PIL Image: Padded image.
    """
    if not transforms.functional._is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if not isinstance(padding, (int, tuple)):
        raise TypeError('Got inappropriate padding arg')
   

    if isinstance(padding, tuple) and len(padding) not in [2, 4]:
        raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    if isinstance(padding, tuple) and len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    if isinstance(padding, tuple) and len(padding) == 4:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]

    if img.mode == 'P':
        palette = img.getpalette()
        img = np.asarray(img)
        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), 'wrap')
        img = Image.fromarray(img)
        img.putpalette(palette)
        return img

    img = np.asarray(img)
    # RGB image
    if len(img.shape) == 3:
        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'wrap')
    # Grayscale image
    if len(img.shape) == 2:
        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), 'wrap')

    return Image.fromarray(img)
           

