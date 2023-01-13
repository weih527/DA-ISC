import albumentations as albu
import numpy as np
import cv2
from albumentations import *


def strong_aug(p=.5, cropsize=[512, 512]):
    return Compose([
        Flip(),
        Transpose(),
        Rotate(),
        OneOf([Resize(p=0.2, height=cropsize[0], width=cropsize[1]),
               RandomSizedCrop(((256, 512)), p=0.2, height=cropsize[0], width=cropsize[1], interpolation=2),
               ], p=0.2),
        RandomBrightnessContrast(),
        MotionBlur(p=0.2),
        ElasticTransform(p=0.3),
    ], p=p)


def create_transformer(transformations, images):
    target = {}
    for i, image in enumerate(images[1:]):
        target['image' + str(i)] = 'image'
    return albu.Compose(transformations, p=0.5, additional_targets=target)(image=images[0],
                                                                           mask=images[1]
                                                                           )


def aug_img_lab(img, lab, cropsize, p=0.5):
    images = [img, lab]
    transformed = create_transformer(strong_aug(p=p, cropsize=cropsize), images)
    return transformed['image'], transformed['mask']
