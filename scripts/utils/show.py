'''
Description: 
Author: weihuang
Date: 2021-11-15 17:37:00
LastEditors: Please set LastEditors
LastEditTime: 2021-11-21 17:13:38
'''
import os
import cv2
import math
from numpy.lib.npyio import save
import torch
import numpy as np
from PIL import Image

from utils.pre_processing import division_array, image_concatenate

def polarize(img):
    ''' Polarize the value to zero and one
    Args:
        img (numpy): numpy array of image to be polarized
    return:
        img (numpy): numpy array only with zero and one
    '''
    img[img >= 0.5] = 1
    img[img < 0.5] = 0
    return img

def convert2png(img):
    img = img.data.cpu().numpy()
    img = np.squeeze(img)
    img[img<0] = 0
    img[img>1] = 1
    img = (img * 255).astype(np.uint8)
    return img

def show_training(iters, img, label, pred, save_path, tag='d'):
    img = convert2png(img)
    label = convert2png(label)
    pred = torch.argmax(pred, dim=0).float()
    pred = convert2png(pred)
    concat = np.concatenate([img, label, pred], axis=1)
    Image.fromarray(concat).save(os.path.join(save_path, str(iters).zfill(6)+'_%s.png' % tag))

def show_training_allresults(iters,
                            cimg,
                            clabel,
                            cpred,
                            aimg,
                            alabel,
                            apred,
                            dlabel,
                            dpred,
                            ccross,
                            across,
                            save_path,
                            tag='s'):
    cimg = convert2png(cimg)
    aimg = convert2png(aimg)
    clabel = convert2png(clabel)
    alabel = convert2png(alabel)
    dlabel = convert2png(dlabel)
    ccross = convert2png(ccross)
    across = convert2png(across)
    cpred = torch.argmax(cpred, dim=0).float()
    cpred = convert2png(cpred)
    apred = torch.argmax(apred, dim=0).float()
    apred = convert2png(apred)
    dpred = torch.argmax(dpred, dim=0).float()
    dpred = convert2png(dpred)
    concat1 = np.concatenate([cimg, clabel, cpred, dpred, ccross], axis=1)
    concat2 = np.concatenate([aimg, alabel, apred, dlabel, across], axis=1)
    concat = np.concatenate([concat1, concat2], axis=0)
    Image.fromarray(concat).save(os.path.join(save_path, str(iters).zfill(6)+'_%s.png' % tag))

def show_test(preds, labels, raw_path, save_path):
    num = labels.shape[0]
    for k in range(num):
        img = np.asarray(Image.open(os.path.join(raw_path, str(k).zfill(3)+'.png')))
        pred = preds[k]
        label = labels[k]
        img = draw_label(img, pred, label)
        cv2.imwrite(os.path.join(save_path, str(k).zfill(3)+'.png'), img)

def draw_label(img, pred, label):
    if img.max() <= 1:
        img = (img * 255).astype(np.uint8)
    if pred.max() <= 1:
        pred = (pred * 255).astype(np.uint8)
    else:
        pred = pred.astype(np.uint8)
    if label.max() <= 1:
        label = (label * 255).astype(np.uint8)
    else:
        label = label.astype(np.uint8)
    if len(img.shape) == 2:
        img = img[:,:,np.newaxis]
        img = np.repeat(img, 3, 2)
    contours_lb, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(img, contours_lb, -1, (0,0,255), 2)
    contours_pred, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(img, contours_pred, -1, (0,255,0), 2)
    return img

def save_prediction_image(stacked_img, im_name, iters, save_folder_name, original_msk):
    """save images to save_path
    Args:
        stacked_img (numpy): stacked cropped images
        save_folder_name (str): saving folder name
        division_array(388, 2, 3, 768, 1024):
                388: label patch size
                2, divide num in heigh
                3, divide num in width
                768: image height
                1024: image width
    """
    
    crop_size = stacked_img[0].size()

    maxsize = original_msk.shape[1:]

    output_shape = original_msk.shape[1:]
    crop_n1 = math.ceil(output_shape[0] / crop_size[0])
    crop_n2 = math.ceil(output_shape[1] / crop_size[1])
    if crop_n1 == 1:
        crop_n1 = crop_n1
    else:
        crop_n1 = crop_n1 + 1
    if crop_n2 == 1:
        crop_n2 = crop_n2
    else:
        crop_n2 = crop_n2 + 1

    div_arr = division_array(stacked_img.size(1), crop_n1, crop_n2, output_shape[0], output_shape[1])
    img_cont = image_concatenate(stacked_img.cpu().data.numpy(), crop_n1, crop_n2, output_shape[0], output_shape[1])

    img_cont = polarize((img_cont) / div_arr)
    img_cont_np = img_cont.astype('uint8')

    img_cont = Image.fromarray(img_cont_np * 255)
    # organize images in every epoch
    desired_path = os.path.join(save_folder_name, str(iters).zfill(6))
    # desired_path = save_folder_name + '_iter_' + str(iter) + '/'
    # Create the path if it does not exist
    if not os.path.exists(desired_path):
        os.makedirs(desired_path)
    # Save Image!
    export_name = str(im_name) + '.png'
    # img_cont.save(desired_path + export_name)
    img_cont.save(os.path.join(desired_path, export_name))
    return img_cont_np, original_msk
