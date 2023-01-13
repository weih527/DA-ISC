'''
Description: 
Author: weihuang
Date: 2021-11-18 12:39:28
LastEditors: weihuang
LastEditTime: 2021-11-18 12:51:21
'''
import os
import os.path as osp
import numpy as np
from PIL import Image

in_path = '../data/Lucchi/training_groundtruth'
out_path = '../data/Lucchi/training_groundtruth_diff_10'
if not osp.exists(out_path):
    os.makedirs(out_path)

NUM = 165
stride = 10
for i in range(NUM-stride):
    img1 = np.asarray(Image.open(osp.join(in_path, str(i).zfill(3)+'.png')))
    img2 = np.asarray(Image.open(osp.join(in_path, str(i+stride).zfill(3)+'.png')))
    img1 = (img1 / 255).astype(np.bool)
    img2 = (img2 / 255).astype(np.bool)
    # diff = img1 ^ img2  # 异或
    diff = np.bitwise_xor(img1, img2)
    diff = (diff).astype(np.uint8) * 255
    Image.fromarray(diff).save(osp.join(out_path, str(i).zfill(3)+'.png'))
print('Done')
