'''
Description: 
Author: weihuang
Date: 2021-11-18 15:47:44
LastEditors: weihuang
LastEditTime: 2021-11-18 19:18:39
'''
import os
import sys
import torch
import random
import numpy as np
from PIL import Image
import os.path as osp
from random import randint
from torch.utils import data
from utils.pre_processing import normalization2, approximate_image, cropping
from dataset.data_aug import aug_img_lab


class sourceDataSet(data.Dataset):
    def __init__(self, root_img, root_label, list_path, crop_size=(512, 512), stride=1):
        self.root_img = root_img
        self.root_label = root_label
        self.list_path = list_path
        self.crop_size = crop_size
        self.stride = stride
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.length = len(self.img_ids)

    def __len__(self):
        # return int(sys.maxsize)
        return 400000

    def __getitem__(self, index):
        k = random.randint(0, len(self.img_ids)-1-self.stride)
        current_img = np.asarray(Image.open(osp.join(self.root_img, self.img_ids[k])), dtype=np.uint8)
        current_label = np.asarray(Image.open(osp.join(self.root_label, self.img_ids[k])), dtype=np.uint8)
        aux_img = np.asarray(Image.open(osp.join(self.root_img, self.img_ids[k+self.stride])), dtype=np.uint8)
        aux_label = np.asarray(Image.open(osp.join(self.root_label, self.img_ids[k+self.stride])), dtype=np.uint8)

        # data augmentation
        current_img = normalization2(current_img.astype(np.float32), max=1, min=0)
        aux_img = normalization2(aux_img.astype(np.float32), max=1, min=0)
        seed = np.random.randint(2147483647)
        random.seed(seed)
        current_img, current_label = aug_img_lab(current_img, current_label, self.crop_size)
        random.seed(seed)
        aux_img, aux_label = aug_img_lab(aux_img, aux_label, self.crop_size)
        current_label = approximate_image(current_label.copy())
        aux_label = approximate_image(aux_label.copy())

        # cropping image with the input size
        size = current_img.shape
        y_loc = randint(0, size[0] - self.crop_size[0])
        x_loc = randint(0, size[1] - self.crop_size[1])
        current_img = cropping(current_img, self.crop_size[0], self.crop_size[1], y_loc, x_loc)
        current_label = cropping(current_label, self.crop_size[0], self.crop_size[1], y_loc, x_loc)
        aux_img = cropping(aux_img, self.crop_size[0], self.crop_size[1], y_loc, x_loc)
        aux_label = cropping(aux_label, self.crop_size[0], self.crop_size[1], y_loc, x_loc)

        current_img = np.expand_dims(current_img, axis=0)  # add additional dimension
        current_img = torch.from_numpy(current_img.astype(np.float32)).float()
        aux_img = np.expand_dims(aux_img, axis=0)  # add additional dimension
        aux_img = torch.from_numpy(aux_img.astype(np.float32)).float()

        current_label = (current_label / 255).astype(np.bool)
        aux_label = (aux_label / 255).astype(np.bool)
        diff = np.bitwise_xor(current_label, aux_label)
        current_label = torch.from_numpy(current_label.astype(np.float32)).long()
        aux_label = torch.from_numpy(aux_label.astype(np.float32)).long()
        diff = torch.from_numpy(diff.astype(np.float32)).long()

        return current_img, current_label, aux_img, aux_label, diff


if __name__ == '__main__':
    # data_dir_img = '../data/VNC3/training/'
    # data_dir_label = '../data/VNC3/training_groundtruth/'
    # data_list = '../data/VNC3/train.txt'
    data_dir_img = '../data/Lucchi/training'
    data_dir_label = '../data/Lucchi/training_groundtruth'
    data_list = '../data/Lucchi/train.txt'
    input_size = (512, 512)
    stride = 10
    dst = sourceDataSet(data_dir_img,
                        data_dir_label,
                        data_list,
                        crop_size=input_size,
                        stride=stride)

    out_path = './data_temp'
    if not osp.exists(out_path):
        os.makedirs(out_path)
    for i, data in enumerate(dst):
        if i < 50:
            print(i)
            current_img, current_label, aux_img, aux_label, diff = data
            current_img = (current_img.numpy() * 255).astype(np.uint8)
            current_label = (current_label.numpy() * 255).astype(np.uint8)
            current_img = current_img.squeeze()
            aux_img = (aux_img.numpy() * 255).astype(np.uint8)
            aux_label = (aux_label.numpy() * 255).astype(np.uint8)
            aux_img = aux_img.squeeze()
            diff = (diff.numpy() * 255).astype(np.uint8)
            concat1 = np.concatenate([current_img, aux_img, diff], axis=1)
            concat2 = np.concatenate([current_label, aux_label, diff], axis=1)
            concat = np.concatenate([concat1, concat2], axis=0)
            Image.fromarray(concat).save(osp.join(out_path, str(i).zfill(4)+'.png'))
        else:
            break
    print('Done')
