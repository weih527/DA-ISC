'''
Description: 
Author: weihuang
Date: 2021-11-17 09:38:38
LastEditors: weihuang
LastEditTime: 2021-11-17 10:37:47
'''

import os
import cv2
import torch
import numpy as np
from torch.utils import data
from torch.autograd import Variable
from utils.metrics import dice_coeff
from utils.postprocessing import postpre
from dataset.target_dataset import targetDataSet_test
from utils.tools_self import save_array_as_nii_volume
from utils.show import save_prediction_image

def test_model(model, valloader, save_dir, i_iter):
    device = torch.device('cuda:0')
    total_dice = 0
    total_jac = 0
    count = 0
    pred_total = []
    original_msk_total = []
    for i_pic, (images_v, masks_v, original_msk, _, name)in enumerate(valloader):
        stacked_img = torch.Tensor([]).to(device)
        for index in range(images_v.size()[1]):
            with torch.no_grad():
                image_v = Variable(images_v[:, index, :, :].unsqueeze(0).to(device))
            try:
                _, output = model(image_v)
                output = torch.argmax(output, dim=1).float()
                stacked_img = torch.cat((stacked_img, output))
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise e
        pred, original_msk = save_prediction_image(stacked_img, name, i_iter, save_dir, original_msk)
        dim = pred.shape

        dice, jac = dice_coeff(pred, original_msk)
        count = count + 1

        total_dice = total_dice + dice
        total_jac = total_jac + jac

        print("%d.  val_jac is:%f . val_dice is:%f " % (i_pic, jac, dice))

        pred_total = np.append(pred_total, pred)
        original_msk_total = np.append(original_msk_total, original_msk)

    D3_dice, D3_jac = dice_coeff(pred_total, original_msk_total)
    D2_dice = total_dice / count
    D2_jac = total_jac / count
    print('3D dice: %4f' % D3_dice, '3D jac: %4f' % D3_jac,
          '2D dice: %4f' % D2_dice, '2D jac: %4f' % D2_jac)

    pred_total = pred_total.reshape(count, dim[0], dim[1])
    original_msk_total = original_msk_total.reshape(count, dim[0], dim[1])
    return pred_total, original_msk_total

def test_augmentation(testmodel, pred_ori, input_size_target, cfg, save_dir):
    pred_final = pred_ori
    for test_aug in range(4):
        print('the %d test_aug' % test_aug, 'for %s' % save_dir)
        testloader = data.DataLoader(
            targetDataSet_test(cfg.DATA.data_dir_val,
                               cfg.DATA.data_dir_val_label,
                               cfg.DATA.data_list_val,
                               test_aug,
                               crop_size=input_size_target),
            batch_size=1, shuffle=False)

        pred_total, original_msk_total = test_model(testmodel, testloader, save_dir, 1000)
        pred_total = postpre(pred_total, save_dir, test_aug)
        D3_dice, D3_jac = dice_coeff(pred_total, original_msk_total)

        total_dice = 0
        total_jac = 0
        pics = pred_total.shape[0]
        for i in range(pics):
            pred = pred_total[i, :, :]
            msk = original_msk_total[i, :, :]
            dice, jac = dice_coeff(pred, msk)
            total_dice = total_dice + dice
            total_jac = total_jac + jac

        print('3D dice: %4f' % D3_dice, '3D jac: %4f' % D3_jac,
              '2D dice: %4f' % (total_dice / (pics)), '2D jac: %4f' % (total_jac / (pics)))

        if test_aug == 0:
            msk_final = original_msk_total
        if test_aug == 1:
            for i in range(pred_total.shape[0]):
                pred_total[i, :, :] = cv2.flip(pred_total[i, :, :], 1)
        if test_aug == 2:
            for i in range(pred_total.shape[0]):
                pred_total[i, :, :] = cv2.flip(pred_total[i, :, :], 0)
        if test_aug == 3:
            for i in range(pred_total.shape[0]):
                pred_total[i, :, :] = cv2.flip(pred_total[i, :, :], -1)

        pred_final = pred_final + pred_total

    pred_final = pred_final / 4
    pred_final[pred_final >= 0.5] = 1
    pred_final[pred_final < 0.5] = 0

    desired_path = save_dir + 'final' + '/'
    if not os.path.exists(desired_path):
        os.makedirs(desired_path)
    export_name = 'test.nii.gz'
    save_array_as_nii_volume(pred_final, desired_path + export_name)

    return pred_final, msk_final
