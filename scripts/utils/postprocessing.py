import cv2
import numpy as np
import os
from utils.tools_self import save_array_as_nii_volume
from skimage import measure
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology

def postpre_single(img, test_aug=0):
    if test_aug == 5:
        img_fill = fillHole_single(img)
        img_open = openimg_single(img_fill, k1_size=10, k2_size=10)
        img_out = remove_small_object_single(img_open, area=600)
    else:
        img_fill = fillHole_single(img)
        img_open = openimg_single(img_fill, k1_size=15, k2_size=15)
        img_out = remove_small_object_single(img_open, area=400)
    return img_out

def fillHole_single(img):
    img = (img*255).astype(np.uint8)

    mask = 255 - img
    marker = np.zeros_like(img)
    marker[0,:] = 255
    marker[-1,:] = 255
    marker[:,0] = 255
    marker[:,-1] = 255

    SE = cv2.getStructuringElement(shape = cv2.MORPH_CROSS, ksize = (3,3))
    while True:
        marker_pre = marker
        dilation = cv2.dilate(marker, kernel = SE)
        marker = np.min((dilation,mask), axis = 0)
        if (marker_pre == marker).all():
            break
    dst = 255 - marker
    dst = dst / 255.0
    return dst

def openimg_single(img, k1_size, k2_size):
    k1_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1_size, k1_size))
    k2_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2_size, k2_size))

    img_eroded = cv2.erode(img, k1_erode)
    img_out = cv2.dilate(img_eroded, k2_dilate)
    return img_out

def remove_small_object_single(img, area):
    img = (img).astype(np.int)
    labels = measure.label(img, connectivity=1)
    img_out = morphology.remove_small_objects(labels, min_size=area, connectivity=1, in_place=False)
    img_out[img_out>0]=1
    img_out = img_out.astype('uint8')
    return img_out

def postpre(img,save_dir,test_aug):

    # test: testing subset parameter
    if test_aug == 5:
        img_fill = fillHole(img,save_dir,test_aug)

        img_open = openimg(img_fill,save_dir,test_aug,k1_size=10,k2_size=10)

        img_out = remove_small_object(img_open,save_dir,test_aug,area = 600)

    else:
        img_fill = fillHole(img,save_dir,test_aug)

        img_open = openimg(img_fill,save_dir,test_aug,k1_size=15,k2_size=15)

        img_out = remove_small_object(img_open,save_dir,test_aug,area = 400)
    # test: training subset parameter
    # if test_aug == 5:
    #     img_fill = fillHole(img, save_dir, test_aug)
    #
    #     img_open = openimg(img_fill, save_dir, test_aug, k1_size=20, k2_size=20)
    #
    #     img_out = remove_small_object(img_open, save_dir, test_aug, area=600)
    #
    # else:
    #     img_fill = fillHole(img, save_dir, test_aug)
    #
    #     img_open = openimg(img_fill, save_dir, test_aug, k1_size=15, k2_size=15)
    #
    #     img_out = remove_small_object(img_open, save_dir, test_aug, area=600)

    # organize images in every epoch
    desired_path = save_dir + '/_postpre'+'_'+str(test_aug)+'/'
    # Create the path if it does not exist
    if not os.path.exists(desired_path):
        os.makedirs(desired_path)

    export_name = 'test.nii.gz'
    save_array_as_nii_volume(img_out, desired_path + export_name)

    return img_out

def fillHole(img_arr,save_dir,test_aug):

    num = 0

    img_arr = (img_arr*255).astype(np.uint8)
    img_arr_out = img_arr.copy()

    size = img_arr.shape

    for i in range(size[0]):
        img = img_arr[i,:,:]

        mask = 255 - img

        marker = np.zeros_like(img)
        marker[0,:] = 255
        marker[-1,:] = 255
        marker[:,0] = 255
        marker[:,-1] = 255

        marker_0 = marker.copy()

        SE = cv2.getStructuringElement(shape = cv2.MORPH_CROSS,ksize = (3,3))
        while True:
            marker_pre = marker
            dilation = cv2.dilate(marker,kernel = SE)
            marker = np.min((dilation,mask),axis = 0)
            if (marker_pre == marker).all():
                break
        dst = 255-marker

        img_arr_out[i,:,:] = dst/255

    return img_arr_out

def openimg(img,save_dir,test_aug,k1_size,k2_size):

    k1_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1_size, k1_size))
    k2_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2_size, k2_size))

    size = img.shape
    img_arr = img
    img_arr_out = img.copy()

    for i in range(size[0]):
        pics = img_arr[i, :, :]

        img_eroded = cv2.erode(pics,k1_erode)
        img_out = cv2.dilate(img_eroded,k2_dilate)

        img_arr_out[i, :, :] = img_out

    return img_arr_out

def remove_small_object(img,save_dir,test_aug,area):

    img = (img).astype(np.int)
    labels = measure.label(img,connectivity=1)
    # print(np.unique(labels))
    label_att = measure.regionprops(labels)
    # for i in range(len(label_att)):
    #     print(label_att[i].area)
    img_out = morphology.remove_small_objects(labels, min_size=area, connectivity=1, in_place=False)
    img_out[img_out>0]=1
    img_out = img_out.astype('uint8')

    return img_out



