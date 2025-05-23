'''
Adapted from https://github.com/facebookresearch/barlowtwins
and mostly taken from https://github.com/MaxLikesMath/Barlow-Twins-Pytorch
'''

import random
from PIL import Image, ImageOps, ImageFilter

import numpy as np
import torch
import torchvision.transforms as transforms

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class TransformFirst:
    def __init__(self, transform=None):
        if transform == None:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                            saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),
                Solarization(p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __call__(self, x):
        return self.transform(x)

class TransformSecond:
    def __init__(self, transform=None):
        if transform == None:

            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                            saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.1),
                Solarization(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __call__(self, x):
        return self.transform(x)

class Transform:
    def __init__(self, transform=None, transform_prime=None):
        '''

        :param transform: Transforms to be applied to first input
        :param transform_prime: transforms to be applied to second
        '''
        if transform == None:
            self.transform = TransformFirst()
        else:
            self.transform = transform
        if transform_prime == None:
            self.transform_prime = TransformSecond()
        else:
            self.transform_prime = transform_prime

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return torch.cat((y1.unsqueeze(0), y2.unsqueeze(0)))
