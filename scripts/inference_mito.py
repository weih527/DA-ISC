'''
Description: 
Author: weihuang
Date: 2021-11-16 21:17:31
LastEditors: Please set LastEditors
LastEditTime: 2023-01-13 21:51:31
'''

import os
import cv2
import yaml
import h5py
import time
import argparse
import numpy as np
from tqdm import tqdm
from attrdict import AttrDict
from collections import OrderedDict
from PIL import Image

import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F

from model.CoDetectionCNN import CoDetectionCNN
from dataset.target_dataset_mito import targetDataSet_test_twoimgs, Evaluation
from utils.show import show_test

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='mitor2h', help='config file')
    parser.add_argument('-mn', '--model_name', type=str, default='mitor2h')
    parser.add_argument('-mm', '--mode_map', type=str, default='map_2d')
    parser.add_argument('-sw', '--show', action='store_true', default=False)
    args = parser.parse_args()
    
    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.load(f, Loader=yaml.FullLoader))
    
    trained_model = args.model_name
    out_path = os.path.join('../inference', trained_model)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print('out_path: ' + out_path)
    seg_img_path = os.path.join(out_path, 'seg_img')
    if not os.path.exists(seg_img_path):
        os.makedirs(seg_img_path)
    
    device = torch.device('cuda:0')
    model = CoDetectionCNN(n_channels=cfg.MODEL.input_nc,
                           n_classes=cfg.MODEL.output_nc).to(device)
    
    ckpt_path = os.path.join('../models', trained_model, 'model.ckpt')
    checkpoint = torch.load(ckpt_path)
    new_state_dict = OrderedDict()
    state_dict = checkpoint['model_weights']
    for k, v in state_dict.items():
        # name = k[7:] # remove module.
        name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    val_data = targetDataSet_test_twoimgs(cfg.DATA.data_dir_val,
                                        cfg.DATA.data_dir_val_label,
                                        cfg.DATA.data_list_val,
                                        crop_size=(cfg.DATA.input_size_test, cfg.DATA.input_size_test),
                                        stride=cfg.DATA.target_stride)
    valid_provider = torch.utils.data.DataLoader(val_data, batch_size=1)

    target_evaluation = Evaluation(root_label=cfg.DATA.data_dir_val_label,
                                   list_path=cfg.DATA.data_list_val)
    print('Begin inference...')
    f_valid_txt = open(os.path.join(out_path, 'scores.txt'), 'w')
    print('the number of sub-volume:', len(val_data))
    t1 = time.time()
    pbar = tqdm(total=len(val_data))
    for k, data in enumerate(valid_provider, 0):
        cimg, aimg = data
        cimg = cimg.to(device)
        aimg = aimg.to(device)
        img_cat = torch.cat([cimg, aimg], dim=1)
        with torch.no_grad():
            cpred, apred = model(img_cat, diff=False)
        cpred = torch.nn.functional.softmax(cpred, dim=1)
        cpred = cpred[:, 1]
        apred = torch.nn.functional.softmax(apred, dim=1)
        apred = apred[:, 1]
        cpred = np.squeeze(cpred.data.cpu().numpy())
        apred = np.squeeze(apred.data.cpu().numpy())
        val_data.add_vol(cpred, apred)
        pbar.update(1)
    pbar.close()
    preds = val_data.get_results()
    t2 = time.time()
    print('Prediction time (s):', (t2 - t1))

    f_out = h5py.File(os.path.join(out_path, 'preds.hdf'), 'w')
    f_out.create_dataset('main', data=preds, dtype=np.float32, compression='gzip')
    f_out.close()

    if args.show:
        print('Show...')
        preds_int = preds.copy()
        preds_int[preds_int>=0.5] = 1
        preds_int[preds_int<0.5] = 0
        # show_test(preds_int, target_evaluation.get_gt(), cfg.DATA.data_dir_val, seg_img_path)
        for k in range(preds_int.shape[0]):
            temp = preds_int[k]
            temp = (temp * 255).astype(np.uint8)
            Image.fromarray(temp).save(os.path.join(seg_img_path, str(k).zfill(4)+'.png'))
        del preds_int

    # mAP, F1, MCC, and IoU
    print('Measure on mAP, F1, MCC, and IoU...')
    t3 = time.time()
    mAP, F1, MCC, IoU = target_evaluation(preds, mode=args.mode_map)
    t4 = time.time()
    print('mAP=%.4f, F1=%.4f, MCC=%.4f, IoU=%.4f' % (mAP, F1, MCC, IoU))
    print('Measurement time (s):', (t4 - t3))
    f_valid_txt.write('mAP=%.4f, F1=%.4f, MCC=%.4f, IoU=%.4f' % (mAP, F1, MCC, IoU))
    f_valid_txt.write('\n')
    f_valid_txt.close()

    print('Done')