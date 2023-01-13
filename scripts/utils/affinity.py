'''
Descripttion: 
version: 0.0
Author: Wei Huang
Date: 2022-01-07 21:20:55
'''

import torch
import numpy as np

def binary_func(img):
    img[img <= 0.5] = 0
    img[img > 0.5] = 1
    return img

def difference(img1, img2, shift):
    if shift == 0:
        diff = np.bitwise_xor(img1, img2)
        diff = diff.astype(np.float32)
        return diff
    else:
        img1_shift = np.zeros_like(img1)
        img1_shift[shift:, :] = img1[:-shift, :]
        diff1 = np.bitwise_xor(img1_shift, img2)
        diff1 = diff1.astype(np.float32)

        img1_shift = np.zeros_like(img1)
        img1_shift[:, shift:] = img1[:, :-shift]
        diff2 = np.bitwise_xor(img1_shift, img2)
        diff2 = diff2.astype(np.float32)
        return [diff1, diff2]

def gen_affs(img1, img2, shifts=[0]):
    assert len(shifts) > 0, "shifts must be [0, ...]"

    img1 = binary_func(img1)
    img2 = binary_func(img2)

    if img1.max() > 1:
        img1 = img1.astype(np.float32) / 255
    img1 = img1.astype(bool)
    if img2.max() > 1:
        img2 = img2.astype(np.float32) / 255
    img2 = img2.astype(bool)

    length = len(shifts)
    h, w = img1.shape
    affs = np.zeros((length*2-1, h, w), dtype=np.float32)
    for i, shift in enumerate(shifts):
        diff = difference(img1.copy(), img2.copy(), shift)
        if shift == 0:
            affs[0] = diff
        else:
            affs[2*i-1] = diff[0]
            affs[2*i] = diff[1]
    return affs

def recover_previous_single(img2, affs, shift):
    if shift == 0:
        img1 = np.abs(affs - img2)
        img1 = binary_func(img1)
        return img1
    else:
        affs1 = affs[0]
        affs2 = affs[1]
        img1_1 = np.abs(affs1 - img2)
        img1_1_shift = np.zeros_like(img1_1)
        img1_1_shift[:-shift, :] = img1_1[shift:, :]
        img1_1_shift = binary_func(img1_1_shift)
        img1_2 = np.abs(affs2 - img2)
        img1_2_shift = np.zeros_like(img1_2)
        img1_2_shift[:, :-shift] = img1_2[:, shift:]
        img1_2_shift = binary_func(img1_2_shift)
        return [img1_1_shift, img1_2_shift]

def recover_subsequent_single(img1, affs, shift):
    if shift == 0:
        img2 = np.abs(affs - img1)
        img2 = binary_func(img2)
        return img2
    else:
        affs1 = affs[0]
        affs2 = affs[1]
        img2_1_shift = np.zeros_like(img1)
        img2_1_shift[shift:, :] = img1[:-shift, :]
        img2_1 = np.abs(affs1 - img2_1_shift)
        img2_1 = binary_func(img2_1)
        img2_2_shift = np.zeros_like(img1)
        img2_2_shift[:, shift:] = img1[:, :-shift]
        img2_2 = np.abs(affs2 - img2_2_shift)
        img2_2 = binary_func(img2_2)
        return [img2_1, img2_2]

def recover_previous(img2, affs, shifts):
    img1_recovered = np.zeros_like(affs, dtype=np.float32)
    for i, shift in enumerate(shifts):
        if shift == 0:
            temp_img1 = recover_previous_single(img2, affs[0], shift)
            img1_recovered[0] = temp_img1
        else:
            temp_img1 = recover_previous_single(img2, [affs[2*i-1], affs[2*i]], shift)
            img1_recovered[2*i-1] = temp_img1[0]
            img1_recovered[2*i] = temp_img1[1]
    return img1_recovered

def recover_subsequent(img1, affs, shifts):
    img2_recovered = np.zeros_like(affs, dtype=np.float32)
    for i, shift in enumerate(shifts):
        if shift == 0:
            temp_img2 = recover_subsequent_single(img1, affs[0], shift)
            img2_recovered[0] = temp_img2
        else:
            temp_img2 = recover_subsequent_single(img1, [affs[2*i-1], affs[2*i]], shift)
            img2_recovered[2*i-1] = temp_img2[0]
            img2_recovered[2*i] = temp_img2[1]
    return img2_recovered

def recover(img1, img2, affs, shifts=[0], binary=True):
    if binary:
        img1 = binary_func(img1)
        img2 = binary_func(img2)
        affs = binary_func(affs)
    img1_recovered = np.zeros_like(affs, dtype=np.float32)
    img2_recovered = np.zeros_like(affs, dtype=np.float32)
    for i, shift in enumerate(shifts):
        if shift == 0:
            temp_img1 = recover_previous_single(img2, affs[0], shift)
            img1_recovered[0] = temp_img1
            temp_img2 = recover_subsequent_single(img1, affs[0], shift)
            img2_recovered[0] = temp_img2
        else:
            temp_img1 = recover_previous_single(img2, [affs[2*i-1], affs[2*i]], shift)
            img1_recovered[2*i-1] = temp_img1[0]
            img1_recovered[2*i] = temp_img1[1]
            temp_img2 = recover_subsequent_single(img1, [affs[2*i-1], affs[2*i]], shift)
            img2_recovered[2*i-1] = temp_img2[0]
            img2_recovered[2*i] = temp_img2[1]
    return img1_recovered, img2_recovered

def recover_previous_single_torch(img2, affs, shift):
    if shift == 0:
        img1 = torch.abs(affs - img2)
        return img1
    else:
        affs1 = affs[0]
        affs2 = affs[1]
        img1_1 = torch.abs(affs1 - img2)
        img1_1_shift = torch.zeros_like(img1_1)
        img1_1_shift[:, :-shift, :] = img1_1[:, shift:, :]
        img1_2 = torch.abs(affs2 - img2)
        img1_2_shift = torch.zeros_like(img1_2)
        img1_2_shift[:, :, :-shift] = img1_2[:, :, shift:]
        return [img1_1_shift, img1_2_shift]

def recover_subsequent_single_torch(img1, affs, shift):
    if shift == 0:
        img2 = torch.abs(affs - img1)
        return img2
    else:
        affs1 = affs[0]
        affs2 = affs[1]
        img2_1_shift = torch.zeros_like(img1)
        img2_1_shift[:, shift:, :] = img1[:, :-shift, :]
        img2_1 = torch.abs(affs1 - img2_1_shift)
        img2_2_shift = torch.zeros_like(img1)
        img2_2_shift[:, :, shift:] = img1[:, :, :-shift]
        img2_2 = torch.abs(affs2 - img2_2_shift)
        return [img2_1, img2_2]

def recover_torch(img1, img2, affs, shifts=[0]):
    # img1 = torch.squeeze(img1, dim=0)
    # img2 = torch.squeeze(img2, dim=0)
    # affs = torch.squeeze(affs, dim=0)
    img1_recovered = torch.zeros_like(affs)
    img2_recovered = torch.zeros_like(affs)
    for i, shift in enumerate(shifts):
        if shift == 0:
            temp_img1 = recover_previous_single_torch(img2, affs[:, 0], shift)
            img1_recovered[:, 0] = temp_img1
            temp_img2 = recover_subsequent_single_torch(img1, affs[:, 0], shift)
            img2_recovered[:, 0] = temp_img2
        else:
            temp_img1 = recover_previous_single_torch(img2, [affs[:, 2*i-1], affs[:, 2*i]], shift)
            img1_recovered[:, 2*i-1] = temp_img1[0]
            img1_recovered[:, 2*i] = temp_img1[1]
            temp_img2 = recover_subsequent_single_torch(img1, [affs[:, 2*i-1], affs[:, 2*i]], shift)
            img2_recovered[:, 2*i-1] = temp_img2[0]
            img2_recovered[:, 2*i] = temp_img2[1]
    # img1_recovered = torch.unsqueeze(img1_recovered, dim=0)
    # img2_recovered = torch.unsqueeze(img2_recovered, dim=0)
    return img1_recovered, img2_recovered

def img_mean(img):
    mean_img = torch.mean(img, dim=1, keepdim=False)
    return mean_img

if __name__ == '__main__':
    import os
    from PIL import Image

    data_path = '../data/VNC3/training_groundtruth'
    img1 = np.asarray(Image.open(os.path.join(data_path, '000.png'))).astype(np.float32) / 255.0
    img2 = np.asarray(Image.open(os.path.join(data_path, '001.png'))).astype(np.float32) / 255.0

    shifts = [0,1,3,5,9]

    img1 = binary_func(img1)
    img2 = binary_func(img2)

    affs = gen_affs(img1, img2, shifts)
    print(affs.shape)

    # shift = 1
    # diff = affs[shift]
    # diff = (diff * 255).astype(np.uint8)
    # Image.fromarray(diff).save(os.path.join(data_path, 'diff_%d.png' % shift))

    img1_recovered, img2_recovered = recover(img1, img2, affs, shifts, binary=True)
    print(img1_recovered.shape)
    print(img2_recovered.shape)

    num = img2_recovered.shape[0]
    img1 = img1[:-9, :-9]
    for i in range(num):
        # img2_recovered_tmp = img2_recovered[i]
        # delta = np.sum(np.abs(img2 - img2_recovered_tmp))
        img1_recovered_tmp = img1_recovered[i]
        img1_recovered_tmp = img1_recovered_tmp[:-9, :-9]
        delta = np.sum(np.abs(img1 - img1_recovered_tmp))
        print(delta)
