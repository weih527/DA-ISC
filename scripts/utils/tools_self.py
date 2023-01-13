import SimpleITK as sitk
import numpy as np
import nibabel, math


def cal_crop_num(img_size, in_size):
    if img_size[0] % in_size[0] == 0:
        crop_n1 = math.ceil(img_size[0] / in_size[0]) + 1
    else:
        crop_n1 = math.ceil(img_size[0] / in_size[0])

    if img_size[1] % in_size[1] == 0:
        crop_n2 = math.ceil(img_size[1] / in_size[1]) + 1
    else:
        crop_n2 = math.ceil(img_size[1] / in_size[1])

    if img_size[2] % in_size[2] == 0:
        crop_n3 = math.ceil(img_size[2] / in_size[2]) + 1
    else:
        crop_n3 = math.ceil(img_size[2] / in_size[2])
    return crop_n1, crop_n2, crop_n3


def save_array_as_nii_volume(data, filename, reference_name = None):
    """
    save a numpy array as nifty image
    inputs:
        data: a numpy array with shape [Depth, Height, Width]
        filename: the ouput file name
        reference_name: file name of the reference image of which affine and header are used
    outputs: None
    """
    # data = np.flipud(data)
    # data = np.fliplr(data)
    # data =  np.transpose(data, [2, 0, 1])
    img = sitk.GetImageFromArray(data)
    if(reference_name is not None):
        img_ref = sitk.ReadImage(reference_name)
        img.CopyInformation(img_ref)
    sitk.WriteImage(img, filename)

def find_last(string,str):
    last_position=-1
    while True:
        position=string.find(str,last_position+1)
        if position==-1:
            return last_position
        last_position=position

def load_nifty_volume_as_array(filename, with_header = False):
    """
    load nifty image into numpy array, and transpose it based on the [z,y,x] axis order
    The output array shape is like [Depth, Height, Width]
    inputs:
        filename: the input file name, should be *.nii or *.nii.gz
        with_header: return affine and hearder infomation
    outputs:
        data: a numpy data array
    """
    img = nibabel.load(filename)
    data = img.get_data()
    data = np.transpose(data, [2, 1, 0])
    if with_header:
        return data, img.affine, img.header
    else:
        return data


if __name__ == "__main__":
    print(find_last('t_t1_t2_t3','_'))
    ttt = 't/d/g/g/s/df/d/gggg'
    n = find_last(ttt,'/')
    print(ttt)
    print(ttt[:n])

