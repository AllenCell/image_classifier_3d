import numpy as np
from aicsimageio import imread
from skimage.morphology import ball, dilation
from scipy.ndimage import zoom


def build_one_cell(crop_raw, crop_seg, down_ratio=0.5):

    # load raw
    img_raw = imread(crop_raw)
    img_raw = np.squeeze(img_raw)

    # raw image normalization
    img_raw = img_raw[0:2, :, :, :].astype(np.float32)
    for dim in range(0, 2):
        img_ch = img_raw[dim, :, :, :].copy()
        low = np.percentile(img_ch, 0.05)
        upper = np.percentile(img_ch, 99.5)
        img_ch[img_ch > upper] = upper
        img_ch[img_ch < low] = low
        img_ch = (img_ch - low) / (upper - low)
        img_raw[dim, :, :, :] = img_ch

    # load seg
    img_seg = imread(crop_seg)
    img_seg = np.squeeze(img_seg)
    img_seg = img_seg.astype(np.float32)
    img_seg[img_seg > 0] = 1

    mem_seg = img_seg[1, :, :, :]
    mem_seg = dilation(mem_seg > 0, ball(3))

    z_range = np.where(np.any(mem_seg, axis=(1, 2)))
    y_range = np.where(np.any(mem_seg, axis=(0, 2)))
    x_range = np.where(np.any(mem_seg, axis=(0, 1)))
    z_range = z_range[0]
    y_range = y_range[0]
    x_range = x_range[0]

    roi = [
        max(z_range[0] - 2, 0),
        min(z_range[-1] + 4, mem_seg.shape[0]),
        max(y_range[0] - 5, 0),
        min(y_range[-1] + 5, mem_seg.shape[1]),
        max(x_range[0] - 5, 0),
        min(x_range[-1] + 5, mem_seg.shape[2]),
    ]

    mem_seg = mem_seg[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]]

    mem_img = img_raw[1, roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]]
    dna_img = img_raw[0, roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]]

    mem_seg = zoom(mem_seg, down_ratio, order=0)
    mem_img = zoom(mem_img, down_ratio, order=2)
    dna_img = zoom(dna_img, down_ratio, order=2)

    mem_img[mem_seg == 0] = 0
    dna_img[mem_seg == 0] = 0

    # merge seg and raw
    img_out = np.stack((dna_img, mem_img), axis=0)

    return img_out
