'''
Description: 
Author: weihuang
Date: 2021-11-18 15:47:44
LastEditors: Please set LastEditors
LastEditTime: 2021-11-29 20:07:05
'''
import os
import sys
import h5py
import torch
import random
import numpy as np
from PIL import Image
import os.path as osp
from random import randint
from torch.utils import data
from scipy.ndimage.interpolation import rotate
from utils.pre_processing import normalization2, approximate_image, cropping
from dataset.data_aug import aug_img_lab


class sourceDataSet(data.Dataset):
    def __init__(self, root_img, root_label, list_path=None, crop_size=(512, 512), stride=1):
        print('Load %s' % root_img)
        f = h5py.File(root_img, 'r')
        self.raws = f['main'][:]
        f.close()
        f = h5py.File(root_label, 'r')
        self.labels = f['main'][:]
        f.close()
        print('Data shape:', self.raws.shape)
        self.crop_size = crop_size
        self.stride = stride
        self.length = self.raws.shape[0]
        self.padding = 100
        self.padded_size = [x+2*self.padding for x in self.crop_size]

    def __len__(self):
        # return int(sys.maxsize)
        return 400000

    def __getitem__(self, index):
        k = random.randint(0, self.length - 1 - self.stride)
        current_img = self.raws[k]
        current_label = self.labels[k]
        aux_img = self.raws[k+self.stride]
        aux_label = self.labels[k+self.stride]

        # cropping image with the input size
        size = current_img.shape
        y_loc = randint(0, size[0] - self.padded_size[0])
        x_loc = randint(0, size[1] - self.padded_size[1])
        current_img = cropping(current_img, self.padded_size[0], self.padded_size[1], y_loc, x_loc)
        current_label = cropping(current_label, self.padded_size[0], self.padded_size[1], y_loc, x_loc)
        aux_img = cropping(aux_img, self.padded_size[0], self.padded_size[1], y_loc, x_loc)
        aux_label = cropping(aux_label, self.padded_size[0], self.padded_size[1], y_loc, x_loc)

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

        # crop padding
        if current_img.shape[0] > self.crop_size[0]:
            current_img = current_img[self.padding:-self.padding, self.padding:-self.padding]
            current_label = current_label[self.padding:-self.padding, self.padding:-self.padding]
            aux_img = aux_img[self.padding:-self.padding, self.padding:-self.padding]
            aux_label = aux_label[self.padding:-self.padding, self.padding:-self.padding]

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


class sourceDataSet_chang(data.Dataset):
    def __init__(self, root_img, root_label, list_path=None, crop_size=(512, 512), stride=1):
        print('Load %s' % root_img)
        f = h5py.File(root_img, 'r')
        self.raws = f['main'][:]
        f.close()
        f = h5py.File(root_label, 'r')
        self.labels = f['main'][:]
        f.close()
        print('Data shape:', self.raws.shape)
        self.crop_size = crop_size
        self.stride = stride
        self.length = self.raws.shape[0]
        self.padding = 100
        self.padded_size = [x+2*self.padding for x in self.crop_size]
        self.rigid_aug = True
        self.elastic = True
        self.angle = (0,359)
        self.prob = 0.8

    def __len__(self):
        # return int(sys.maxsize)
        return 400000

    def __getitem__(self, index):
        k = random.randint(0, self.length - 1 - self.stride)
        current_img = self.raws[k]
        current_label = self.labels[k]
        aux_img = self.raws[k+self.stride]
        aux_label = self.labels[k+self.stride]

        # cropping image with the input size
        size = current_img.shape
        y_loc = randint(0, size[0] - self.padded_size[0])
        x_loc = randint(0, size[1] - self.padded_size[1])
        current_img = cropping(current_img, self.padded_size[0], self.padded_size[1], y_loc, x_loc)
        current_label = cropping(current_label, self.padded_size[0], self.padded_size[1], y_loc, x_loc)
        aux_img = cropping(aux_img, self.padded_size[0], self.padded_size[1], y_loc, x_loc)
        aux_label = cropping(aux_label, self.padded_size[0], self.padded_size[1], y_loc, x_loc)

        # data augmentation
        if self.elastic and random.uniform(0,1) < self.prob:
            do_elastic = True
        else:
            do_elastic = False
        # rigid augmentation
        if self.rigid_aug:
            if random.uniform(0,1) < 0.5:
                current_img = np.flip(current_img, axis=0)
                current_label = np.flip(current_label, axis=0)
                aux_img = np.flip(aux_img, axis=0)
                aux_label = np.flip(aux_label, axis=0)
            if random.uniform(0,1) < 0.5:
                current_img = np.flip(current_img, axis=1)
                current_label = np.flip(current_label, axis=1)
                aux_img = np.flip(aux_img, axis=1)
                aux_label = np.flip(aux_label, axis=1)
            
            k = random.choice([0,1,2,3])
            current_img = np.rot90(current_img, k)
            current_label = np.rot90(current_label, k)
            aux_img = np.rot90(aux_img, k)
            aux_label = np.rot90(aux_label, k)
        # elastic deformation
        if do_elastic:
            angle = random.randint(self.angle[0], self.angle[1])
            current_img = current_img.astype(np.float32)
            current_img = rotate(current_img, angle, axes=(0,1), reshape=False, order=3)
            current_label = rotate(current_label, angle, axes=(0,1), reshape=False, order=0)
            aux_img = aux_img.astype(np.float32)
            aux_img = rotate(aux_img, angle, axes=(0,1), reshape=False, order=3)
            aux_label = rotate(aux_label, angle, axes=(0,1), reshape=False, order=0)

        # crop padding
        if current_img.shape[0] > self.crop_size[0]:
            current_img = current_img[self.padding:-self.padding, self.padding:-self.padding]
            current_label = current_label[self.padding:-self.padding, self.padding:-self.padding]
            aux_img = aux_img[self.padding:-self.padding, self.padding:-self.padding]
            aux_label = aux_label[self.padding:-self.padding, self.padding:-self.padding]

        current_img = normalization2(current_img.astype(np.float32), max=1, min=0)
        aux_img = normalization2(aux_img.astype(np.float32), max=1, min=0)
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
    # data_dir_img = '../data/Mito/human/training.hdf'
    # data_dir_label = '../data/Mito/human/training_groundtruth.hdf'
    data_dir_img = '../data/Mito/rat/training.hdf'
    data_dir_label = '../data/Mito/rat/training_groundtruth.hdf'
    data_list = None
    input_size = (512, 512)
    stride = 1
    dst = sourceDataSet_chang(data_dir_img,
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
