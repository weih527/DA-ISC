import os
import cv2
import sys
import math
import torch
import random
import torchvision
from torch.utils import data
import numpy as np
import os.path as osp
from PIL import Image
from random import randint
import matplotlib.pyplot as plt
from utils.pre_processing import normalization2, approximate_image, cropping, multi_cropping
from dataset.data_aug import aug_img_lab
from utils.metrics import dice_coeff

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef

class targetDataSet(data.Dataset):
    def __init__(self, root_img, root_label, list_path, crop_size=(512, 512), stride=1):
        self.root_img = root_img
        self.root_label = root_label
        self.list_path = list_path
        self.crop_size = crop_size
        self.stride = stride
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.length = len(self.img_ids)

    def __len__(self):
        return int(sys.maxsize)

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


class targetDataSet_val(data.Dataset):
    def __init__(self, root_img, root_label, list_path, max_iters=None, crop_size=[512, 512]):
        self.root_img = root_img
        self.root_label = root_label
        self.list_path = list_path
        self.crop_size = crop_size
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for name in self.img_ids:
            img_file = osp.join(self.root_img, name)
            label_file = osp.join(self.root_label, name[:-4] + '.png')
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"])
        image_as_np = np.asarray(image, np.float32)

        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        label_as_img = np.asarray(label, np.float32)
        original_label = torch.from_numpy(np.asarray(label_as_img) / 255.0)

        img_shape = image_as_np.shape

        crop_n1 = math.ceil(img_shape[0] / self.crop_size[0])
        crop_n2 = math.ceil(img_shape[1] / self.crop_size[1])
        if crop_n1 == 1:
            crop_n1 = crop_n1
        else:
            crop_n1 = crop_n1 + 1
        if crop_n2 == 1:
            crop_n2 = crop_n2
        else:
            crop_n2 = crop_n2 + 1

        image_as_np = multi_cropping(image_as_np,
                                     crop_size=self.crop_size[0],
                                     crop_num1=crop_n1, crop_num2=crop_n2)

        processed_list = []

        for array in image_as_np:
            image_to_add = normalization2(array, max=1, min=0)
            processed_list.append(image_to_add)

        image_as_tensor = torch.Tensor(processed_list)

        label_as_np = multi_cropping(label_as_img,
                                     crop_size=self.crop_size[0],
                                     crop_num1=crop_n1, crop_num2=crop_n2)
        label_as_np = label_as_np / 255.0

        label_as_np = torch.from_numpy(label_as_np).long()
        return image_as_tensor, label_as_np, original_label


class targetDataSet_val_weih(data.Dataset):
    def __init__(self, root_img, root_label, list_path, max_iters=None, crop_size=[512, 512]):

        self.root_img = root_img
        self.root_label = root_label
        self.list_path = list_path
        self.crop_size = crop_size
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        if 'Lucchi' in self.root_img:
            self.padding_x = 176
            self.padding_y = 48
        else:
            raise NotImplementedError

        for name in self.img_ids:
            img_file = osp.join(self.root_img, name)
            label_file = osp.join(self.root_label, name[:-4] + '.png')
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"])
        image_as_np = np.asarray(image, np.float32)
        image_as_np = normalization2(image_as_np, max=1, min=0)
        img_shape = image_as_np.shape

        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        label_as_img = np.asarray(label, np.float32)
        label_as_img = label_as_img / 255.0
        original_label = torch.from_numpy(label_as_img.copy())

        # padding
        image_as_np = np.pad(image_as_np, ((self.padding_x,self.padding_x), (self.padding_y,self.padding_y)), mode='reflect')
        label_as_img = np.pad(label_as_img, ((self.padding_x,self.padding_x), (self.padding_y,self.padding_y)), mode='reflect')

        image_as_tensor = torch.Tensor(image_as_np)
        label_as_np = torch.from_numpy(label_as_img).long()
        return image_as_tensor, label_as_np, original_label


class targetDataSet_test(data.Dataset):
    def __init__(self, root_img, root_label, list_path, test_aug,crop_size=[512, 512],max_iters=None):

        self.root_img = root_img
        self.root_label = root_label
        self.list_path = list_path
        self.crop_size = crop_size
        self.test_aug = test_aug
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for name in self.img_ids:
            img_file = osp.join(self.root_img, name)
            label_file = osp.join(self.root_label, name[:-4] + '.png')
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        test_aug = self.test_aug

        image = Image.open(datafiles["img"])
        image_as_np = np.asarray(image, np.float32)

        label = Image.open(datafiles["label"])
        name = datafiles["name"]
        label_as_img = np.asarray(label, np.float32)

        if test_aug == 1:
            image_as_np = cv2.flip(image_as_np, 1)
            label_as_img = cv2.flip(label_as_img, 1)

        if test_aug == 2:
            image_as_np = cv2.flip(image_as_np, 0)
            label_as_img = cv2.flip(label_as_img, 0)

        if test_aug == 3:
            image_as_np = cv2.flip(image_as_np, -1)
            label_as_img = cv2.flip(label_as_img, -1)

        original_label = torch.from_numpy(np.asarray(label_as_img) / 255)

        img_shape = image_as_np.shape

        crop_n1 = math.ceil(img_shape[0] / self.crop_size[0])
        crop_n2 = math.ceil(img_shape[1] / self.crop_size[1])
        if crop_n1 == 1:
            crop_n1 = crop_n1
        else:
            crop_n1 = crop_n1 + 1
        if crop_n2 == 1:
            crop_n2 = crop_n2
        else:
            crop_n2 = crop_n2 + 1

        image_as_np = multi_cropping(image_as_np,
                                     crop_size=self.crop_size[0],
                                     crop_num1=crop_n1, crop_num2=crop_n2)

        processed_list = []

        for array in image_as_np:
            image_to_add = normalization2(array, max=1, min=0)
            processed_list.append(image_to_add)

        image_as_tensor = torch.Tensor(processed_list)

        label_as_np = multi_cropping(label_as_img,
                                     crop_size=self.crop_size[0],
                                     crop_num1=crop_n1, crop_num2=crop_n2)
        label_as_np = label_as_np / 255

        label_as_np = torch.from_numpy(label_as_np).long()
        return image_as_tensor, label_as_np, original_label


class targetDataSet_test_weih(data.Dataset):
    def __init__(self, root_img, root_label, list_path, test_aug=0, max_iters=None, crop_size=[512, 512]):

        self.root_img = root_img
        self.root_label = root_label
        self.list_path = list_path
        self.crop_size = crop_size
        self.test_aug = test_aug
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        if 'Lucchi' in self.root_img:
            self.padding_x = 176
            self.padding_y = 48
        else:
            raise NotImplementedError

        for name in self.img_ids:
            img_file = osp.join(self.root_img, name)
            label_file = osp.join(self.root_label, name[:-4] + '.png')
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"])
        image_as_np = np.asarray(image, np.float32)

        label = Image.open(datafiles["label"])
        name = datafiles["name"]
        label_as_img = np.asarray(label, np.float32)

        image_as_np = normalization2(image_as_np, max=1, min=0)
        img_shape = image_as_np.shape
        label_as_img = label_as_img / 255.0
        original_label = torch.from_numpy(label_as_img.copy())

        # padding
        image_as_np = np.pad(image_as_np, ((self.padding_x,self.padding_x), (self.padding_y,self.padding_y)), mode='reflect')
        label_as_img = np.pad(label_as_img, ((self.padding_x,self.padding_x), (self.padding_y,self.padding_y)), mode='reflect')

        # TTA
        if self.test_aug == 0:
            image_as_np_stack = image_as_np[np.newaxis,...]
            label_as_img_stack = label_as_img[np.newaxis,...]
        else:
            image_as_np1 = cv2.flip(image_as_np.copy(), 1)
            label_as_img1 = cv2.flip(label_as_img.copy(), 1)
            image_as_np2 = cv2.flip(image_as_np.copy(), 0)
            label_as_img2 = cv2.flip(label_as_img.copy(), 0)
            image_as_np3 = cv2.flip(image_as_np.copy(), -1)
            label_as_img3 = cv2.flip(label_as_img.copy(), -1)
            image_as_np_stack = np.stack([image_as_np, image_as_np1, image_as_np2, image_as_np3], axis=0)
            label_as_img_stack = np.stack([label_as_img, label_as_img1, label_as_img2, label_as_img3], axis=0)
        image_as_np_stack = torch.from_numpy(image_as_np_stack)
        label_as_img_stack = torch.from_numpy(label_as_img_stack).long()
        return image_as_np_stack, label_as_img_stack, original_label


class Evaluation(object):
    def __init__(self, root_label, list_path):
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.labels = []
        self.length = len(self.img_ids)
        for k in range(self.length):
            lb = np.asarray(Image.open(osp.join(root_label, self.img_ids[k])), dtype=np.uint8)
            lb = (lb / 255).astype(np.uint8)
            self.labels.append(lb)
        self.labels = np.asarray(self.labels, dtype=np.uint8)

    def __call__(self, preds, mode='dice'):
        if mode == 'dice':
            return self.metric_dice(preds)
        else:
            return self.metric_map(preds)

    def metric_dice(self, preds):
        assert preds.shape[0] == self.length, "Prediction ERROR!"
        dices = []
        jacs = []
        for k in range(self.length):
            dice, jac = dice_coeff(preds[k], self.labels[k])
            dices.append(dice)
            jacs.append(jac)
        dice_avg = sum(dices) / len(dices)
        jac_avg = sum(jacs) / len(jacs)
        return dice_avg, jac_avg

    def metric_map(self, preds):
        assert preds.shape[0] == self.length, "Prediction ERROR!"
        total_mAP = []
        total_F1 = []
        total_MCC = []
        total_IoU = []
        for i in range(self.length):
            pred_temp = preds[i]
            gt_temp = self.labels[i]
            
            serial_segs = gt_temp.reshape(-1)
            mAP = average_precision_score(serial_segs, pred_temp.reshape(-1))

            bin_segs = pred_temp
            bin_segs[bin_segs>=0.5] = 1
            bin_segs[bin_segs<0.5] = 0
            serial_bin_segs = bin_segs.reshape(-1)

            intersection = np.logical_and(serial_segs==1, serial_bin_segs==1)
            union = np.logical_or(serial_segs==1, serial_bin_segs==1)
            IoU = np.sum(intersection) / np.sum(union)

            F1 = f1_score(serial_segs, serial_bin_segs)
            MCC = matthews_corrcoef(serial_segs, serial_bin_segs)
            
            total_mAP.append(mAP)
            total_F1.append(F1)
            total_MCC.append(MCC)
            total_IoU.append(IoU)
        mean_mAP = sum(total_mAP) / len(total_mAP)
        mean_F1 = sum(total_F1) / len(total_F1)
        mean_MCC = sum(total_MCC) / len(total_MCC)
        mean_IoU = sum(total_IoU) / len(total_IoU)
        return mean_mAP, mean_F1, mean_MCC, mean_IoU

    def get_gt(self):
        return self.labels


class targetDataSet_val_twoimgs(data.Dataset):
    def __init__(self, root_img, root_label, list_path, crop_size=[512, 512], stride=1):

        self.root_img = root_img
        self.root_label = root_label
        self.list_path = list_path
        self.crop_size = crop_size
        self.stride = stride
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        self.iters = len(self.img_ids) - self.stride

        if 'Lucchi' in self.root_img:
            self.padding_x = 176
            self.padding_y = 48
        else:
            raise NotImplementedError

        for name in self.img_ids:
            img_file = osp.join(self.root_img, name)
            label_file = osp.join(self.root_label, name[:-4] + '.png')
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return self.iters

    def __getitem__(self, index):
        datafiles1 = self.files[index]
        datafiles2 = self.files[index+self.stride]

        current_img = Image.open(datafiles1["img"])
        current_img = np.asarray(current_img, np.float32)
        current_img = normalization2(current_img, max=1, min=0)
        
        current_label = Image.open(datafiles1["label"])
        current_label = np.asarray(current_label, np.float32)
        current_label = (current_label / 255).astype(np.bool)
        
        aux_img = Image.open(datafiles2["img"])
        aux_img = np.asarray(aux_img, np.float32)
        aux_img = normalization2(aux_img, max=1, min=0)
        
        aux_label = Image.open(datafiles2["label"])
        aux_label = np.asarray(aux_label, np.float32)
        aux_label = (aux_label / 255).astype(np.bool)
        
        diff = np.bitwise_xor(current_label, aux_label)
        current_label = torch.from_numpy(current_label.astype(np.float32)).long()
        aux_label = torch.from_numpy(aux_label.astype(np.float32)).long()
        diff = torch.from_numpy(diff.astype(np.float32)).long()

        # padding
        current_img = np.pad(current_img, ((self.padding_x,self.padding_x), (self.padding_y,self.padding_y)), mode='reflect')
        aux_img = np.pad(aux_img, ((self.padding_x,self.padding_x), (self.padding_y,self.padding_y)), mode='reflect')

        current_img = np.expand_dims(current_img, axis=0)
        current_img = torch.from_numpy(current_img.astype(np.float32)).float()
        aux_img = np.expand_dims(aux_img, axis=0)
        aux_img = torch.from_numpy(aux_img.astype(np.float32)).float()
        return current_img, current_label, aux_img, aux_label, diff


if __name__ == '__main__':
    data_dir_img = '../data/Lucchi/testing'
    data_dir_label = '../data/Lucchi/testing_groundtruth'
    data_list = '../data/Lucchi/testing.txt'
    input_size = (512, 512)
    dst = targetDataSet_test_weih(data_dir_img,
                        data_dir_label,
                        data_list,
                        max_iters=None,
                        crop_size=input_size)

    out_path = './data_temp'
    if not osp.exists(out_path):
        os.makedirs(out_path)
    for i, data in enumerate(dst):
        if i < 50:
            print(i)
            imgs, labels, original_label, _, _ = data
            imgs = (imgs.numpy() * 255).astype(np.uint8)
            labels = (labels.numpy() * 255).astype(np.uint8)
            imgs = imgs.squeeze()
            concat = np.concatenate([imgs, labels], axis=1)
            Image.fromarray(concat).save(osp.join(out_path, str(i).zfill(4)+'.png'))
        else:
            break
    print('Done')