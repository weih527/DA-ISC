'''
Descripttion: 
version: 0.0
Author: Wei Huang
Date: 2021-11-29 17:12:59
'''
import os
import h5py
import torch
from torch.utils import data
import numpy as np
import os.path as osp
from PIL import Image
from utils.pre_processing import normalization2, approximate_image, cropping, multi_cropping
from utils.metrics import dice_coeff

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef

class Evaluation(object):
    def __init__(self, root_label, crop_size=[512, 512]):
        f = h5py.File(root_label, 'r')
        self.labels = f['main'][:]
        f.close()
        self.labels = self.labels[:, :crop_size[0], :crop_size[1]]
        self.length = self.labels.shape[0]
        self.labels = (self.labels / 255).astype(np.uint8)
        self.labels = self.labels.reshape(-1)

    def __call__(self, preds, metric='F1'):
        assert preds.shape[0] == self.length, "Prediction ERROR!"
        if metric == 'F1':
            preds[preds>=0.5] = 1
            preds[preds<0.5] = 0
            F1 = f1_score(self.labels, preds.reshape(-1))
            return F1
        elif metric == 'mAP':
            mAP = average_precision_score(self.labels, preds.reshape(-1))
            return mAP
        else:
            raise NotImplementedError


class targetDataSet_val_twoimgs(data.Dataset):
    def __init__(self, root_img, root_label, list_path=None, crop_size=[512, 512], stride=1):
        print('Load %s' % root_img)
        f = h5py.File(root_img, 'r')
        self.raws = f['main'][:]
        f.close()
        # f = h5py.File(root_label, 'r')
        # self.labels = f['main'][:]
        # f.close()
        self.raws = self.raws[:, :crop_size[0], :crop_size[1]]
        # self.labels = self.labels[:, :crop_size[0], :crop_size[1]]
        print('Data shape:', self.raws.shape)
        self.stride = stride
        self.iters = self.raws.shape[0] - self.stride

    def __len__(self):
        return self.iters

    def __getitem__(self, index):
        current_img = self.raws[index]
        current_img = normalization2(current_img.astype(np.float32), max=1, min=0)
        aux_img = self.raws[index+self.stride]
        aux_img = normalization2(aux_img.astype(np.float32), max=1, min=0)
        
        # current_label = self.labels[index]
        # current_label = (current_label / 255).astype(np.bool)
        # aux_label = self.labels[index+self.stride]
        # aux_label = (aux_label / 255).astype(np.bool)
        
        # diff = np.bitwise_xor(current_label, aux_label)
        # current_label = torch.from_numpy(current_label.astype(np.float32)).long()
        # aux_label = torch.from_numpy(aux_label.astype(np.float32)).long()
        # diff = torch.from_numpy(diff.astype(np.float32)).long()

        current_img = np.expand_dims(current_img, axis=0)
        current_img = torch.from_numpy(current_img.astype(np.float32)).float()
        aux_img = np.expand_dims(aux_img, axis=0)
        aux_img = torch.from_numpy(aux_img.astype(np.float32)).float()
        # return current_img, current_label, aux_img, aux_label, diff
        return current_img, aux_img


class targetDataSet_test_twoimgs(data.Dataset):
    def __init__(self, root_img, root_label, list_path=None, crop_size=[1024, 1024], stride=1):
        print('Load %s' % root_img)
        f = h5py.File(root_img, 'r')
        raws = f['main'][:]
        f.close()
        print('raw shape:', raws.shape)
        raws_padded = np.pad(raws, ((0,0),(256,256),(256,256)), mode='reflect')
        self.raws_padded_shape = raws_padded.shape
        print('padded raw shape:', self.raws_padded_shape)
        self.stride = stride
        self.crop_size = crop_size
        self.stride_xy = crop_size[0] // 2  # 512
        self.num_xy = ((self.raws_padded_shape[1] - crop_size[0]) // self.stride_xy) + 1  # 7+1
        assert (self.raws_padded_shape[1] - crop_size[0]) % self.stride_xy == 0, "padded error!"
        self.num_per_image = self.num_xy * self.num_xy
        self.iter_image = self.raws_padded_shape[0] - self.stride
        self.iters = self.iter_image * self.num_per_image
        print('iters:', self.iters)

        # normalization2
        self.raws_padded_norm = np.zeros_like(raws_padded, dtype=np.float32)
        for k in range(self.raws_padded_shape[0]):
            self.raws_padded_norm[k] = normalization2(raws_padded[k].astype(np.float32), max=1, min=0)

        del raws_padded, raws

        self.reset_output()
        self.weight_vol = self.get_weight()

    def __len__(self):
        return self.iters

    def reset_output(self):
        self.out_results = np.zeros(self.raws_padded_shape, dtype=np.float32)
        self.weight_map = np.zeros(self.raws_padded_shape, dtype=np.float32)

    def __getitem__(self, index):
        pos_image = index // self.num_per_image
        pre_image = index % self.num_per_image
        pos_y = pre_image // self.num_xy
        pos_x = pre_image % self.num_xy

        # find position
        fromy = pos_y * self.stride_xy
        endy = fromy + self.crop_size[0]
        if endy > self.raws_padded_shape[1]:
            endy = self.raws_padded_shape[1]
            fromy = endy - self.crop_size[0]
        fromx = pos_x * self.stride_xy
        endx = fromx + self.crop_size[1]
        if endx > self.raws_padded_shape[2]:
            endx = self.raws_padded_shape[2]
            fromx = endx - self.crop_size[1]
        self.pos = [pos_image, fromy, fromx]

        current_img = self.raws_padded_norm[pos_image, fromx:endx, fromy:endy].copy()
        aux_img = self.raws_padded_norm[pos_image+self.stride, fromx:endx, fromy:endy].copy()

        current_img = current_img.astype(np.float32)
        current_img = np.expand_dims(current_img, axis=0)
        current_img = torch.from_numpy(np.ascontiguousarray(current_img))
        aux_img = aux_img.astype(np.float32)
        aux_img = np.expand_dims(aux_img, axis=0)
        aux_img = torch.from_numpy(np.ascontiguousarray(aux_img))
        return current_img, aux_img

    def get_weight(self, sigma=0.2, mu=0.0):
        yy, xx = np.meshgrid(np.linspace(-1, 1, self.crop_size[0], dtype=np.float32),
                            np.linspace(-1, 1, self.crop_size[1], dtype=np.float32), indexing='ij')
        dd = np.sqrt(yy * yy + xx * xx)
        weight = 1e-6 + np.exp(-((dd - mu) ** 2 / (2.0 * sigma ** 2)))
        return weight

    def add_vol(self, pred_vol, axu_vol):
        pos_image, fromy, fromx = self.pos
        self.out_results[pos_image, fromx:fromx+self.crop_size[0], \
                        fromy:fromy+self.crop_size[1]] += pred_vol * self.weight_vol
        self.out_results[pos_image+self.stride, fromx:fromx+self.crop_size[0], \
                        fromy:fromy+self.crop_size[1]] += axu_vol * self.weight_vol
        self.weight_map[pos_image, fromx:fromx+self.crop_size[0], \
                            fromy:fromy+self.crop_size[1]] += self.weight_vol
        self.weight_map[pos_image+self.stride, fromx:fromx+self.crop_size[0], \
                            fromy:fromy+self.crop_size[1]] += self.weight_vol
    
    def get_results(self):
        self.out_results = self.out_results / self.weight_map
        self.out_results = self.out_results[:, 256:-256, 256:-256]
        return self.out_results


class Evaluation(object):
    def __init__(self, root_label, list_path=None):
        print('Load %s' % root_label)
        f = h5py.File(root_label, 'r')
        self.labels = f['main'][:]
        f.close()
        if self.labels.max() > 1:
            self.labels = self.labels / 255.0
        self.labels = self.labels.astype(np.uint8)
        self.length = self.labels.shape[0]

    def __call__(self, preds, mode='dice'):
        if mode == 'dice':
            return self.metric_dice(preds)
        elif mode == 'map_2d':
            return self.metric_map_2d(preds)
        elif mode == 'map_3d':
            return self.metric_map_3d(preds)
        else:
            raise NotImplementedError

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

    def metric_map_2d(self, preds):
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

    def metric_map_3d(self, preds):
        serial_segs = self.labels.reshape(-1)
        mAP = average_precision_score(serial_segs, preds.reshape(-1))

        bin_segs = preds
        bin_segs[bin_segs>=0.5] = 1
        bin_segs[bin_segs<0.5] = 0
        serial_bin_segs = bin_segs.reshape(-1)

        intersection = np.logical_and(serial_segs==1, serial_bin_segs==1)
        union = np.logical_or(serial_segs==1, serial_bin_segs==1)
        IoU = np.sum(intersection) / np.sum(union)

        F1 = f1_score(serial_segs, serial_bin_segs)
        MCC = matthews_corrcoef(serial_segs, serial_bin_segs)
        return mAP, F1, MCC, IoU

    def get_gt(self):
        return self.labels