import os
import numpy as np
import random
import sys

# from glob import glob
from scipy.ndimage import zoom
import scipy.ndimage as ndi
import torch
import pandas as pd

# from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from typing import List, Union
from .utils import build_one_cell


class basic_loader(Dataset):
    def __init__(self, filenames, buffer_size=-1):

        self.img = []
        self.label = []

        random.shuffle(filenames)
        if buffer_size > 0 and len(filenames) > buffer_size:
            filenames = filenames[:buffer_size]

        print("initializing data loader ...")
        self.filenames = filenames
        self.label = [int(os.path.basename(fn)[0]) for fn in filenames]
        print("data loader initialization is done")
        # self.img = [np.load(fn) for fn in filenames]
        # self.label = [int(os.path.basename(fn)[0]) for fn in filenames]

    def __getitem__(self, index):

        img = np.load(self.filenames[index])
        image_tensor = torch.tensor(img.astype(np.float16))
        label_tensor = torch.tensor(np.int(self.label[index]))
        # image_tensor = from_numpy(self.img[index].astype(np.float16))
        # label_tensor = from_numpy(self.label[index])

        return image_tensor, label_tensor

    def __len__(self):
        return len(self.filenames)


class adaptive_padding_loader(Dataset):
    def __init__(
        self,
        filenames: Union[List[str], str],
        out_shape=[64, 128, 128],
        buffer_size=-1,
        test_flag="T",
    ):

        self.img = []
        self.label = []
        self.out_shape = out_shape
        self.test_flag = test_flag

        if test_flag == "T":  # --> training

            random.shuffle(filenames)
            if buffer_size > 0 and len(filenames) > buffer_size:
                filenames = filenames[:buffer_size]

            print("initializing data loader ...")
            self.filenames = filenames
            self.label = [int(os.path.basename(fn)[0]) for fn in filenames]
            print("data loader initialization is done")

        elif test_flag == "F":  # folder (i.e., testing on a folder of .npy)

            print("initializing data load for testing a folder of npy ...")
            self.filenames = filenames
            self.label = [int(os.path.basename(fn)[0]) for fn in filenames]
            print("test load is done")

        elif test_flag == "C":  # CSV (i.e., testing by csv)
            df = pd.read_csv(filenames)
            self.df = df.reset_index(drop=True)

        else:
            print("unsupported test type")
            print(test_flag)
            sys.exit(0)

    def __getitem__(self, index):

        if self.test_flag == "C":
            # build the cell from raw/seg
            crop_raw = self.df["crop_raw"].iloc[index]
            crop_seg = self.df["crop_seg"].iloc[index]

            npy_fn = os.path.dirname(crop_raw) + "/for_mito_prediction.npy"

            if os.path.exists(npy_fn):
                img = np.load(npy_fn)
            else:
                # check name_dict #TODO
                # print('ready to build the cell')
                img = build_one_cell(crop_raw, crop_seg)
                # print('cell is ready to be padded')
                np.save(npy_fn, img)
        else:
            # load the image from .npy
            fn = self.filenames[index]
            img = np.load(fn)

        # if larger than out_shape, do scaling first to fit into out_shape
        if (
            img.shape[1] > self.out_shape[0]
            or img.shape[2] > self.out_shape[1]
            or img.shape[3] > self.out_shape[2]
        ):
            scaling_ratio = min(
                [
                    self.out_shape[0] / img.shape[1],
                    self.out_shape[1] / img.shape[2],
                    self.out_shape[2] / img.shape[3],
                ]
            )
            img_temp = []
            for ch in range(img.shape[0]):
                img_temp.append(zoom(img[ch, :, :, :], scaling_ratio, order=1))
            img = np.stack(img_temp, axis=0)

        # padding into out_shape
        assert img.shape[1] <= self.out_shape[0]
        assert img.shape[2] <= self.out_shape[1]
        assert img.shape[3] <= self.out_shape[2]

        to_pad_z = self.out_shape[0] - img.shape[1]
        to_pad_y = self.out_shape[1] - img.shape[2]
        to_pad_x = self.out_shape[2] - img.shape[3]

        if not self.test_flag == "T":  # not training, then center pad
            rand_z = int(round(0.5 * to_pad_z))
            rand_y = int(round(0.5 * to_pad_y))
            rand_x = int(round(0.5 * to_pad_x))
        else:
            rand_z = random.randint(
                int(round(2 * to_pad_z / 5)), int(round(3 * to_pad_z / 5))
            )
            rand_y = random.randint(
                int(round(2 * to_pad_y / 5)), int(round(3 * to_pad_y / 5))
            )
            rand_x = random.randint(
                int(round(2 * to_pad_x / 5)), int(round(3 * to_pad_x / 5))
            )

        img = np.pad(
            img,
            (
                (0, 0),
                (rand_z, to_pad_z - rand_z),
                (rand_y, to_pad_y - rand_y),
                (rand_x, to_pad_x - rand_x),
            ),
            "constant",
        )

        if self.test_flag == "T":  # training, then do augmentation
            # decide if flip
            if random.random() < 0.5:
                img = np.flip(img, axis=-1)

            # random rotation
            rand_angle = random.randint(0, 180)
            for ch in range(img.shape[0]):
                for zz in range(img.shape[1]):
                    img[ch, zz, :, :] = ndi.rotate(
                        img[ch, zz, :, :], rand_angle, reshape=False, order=2
                    )

        image_tensor = torch.tensor(img.astype(np.float16))

        if self.test_flag == "C":
            return image_tensor, self.df["CellId"].iloc[index]
        else:
            label_tensor = torch.tensor(np.int(self.label[index]))

            if self.test_flag == "F":  # testing on a folder of npy
                return image_tensor, label_tensor, fn
            else:  # training
                return image_tensor, label_tensor

    def __len__(self):
        if self.test_flag == "T" or self.test_flag == "F":
            return len(self.filenames)
        elif self.test_flag == "C":
            return len(self.df)


class adaptive_loader(Dataset):
    def __init__(self, filenames, test_flag=False):

        self.img = []
        self.label = []

        random.shuffle(filenames)

        print("initializing data loader ...")
        self.filenames = filenames
        self.label = [int(os.path.basename(fn)[0]) for fn in filenames]
        print("data loader initialization is done")

        self.test_flag = test_flag

    def __getitem__(self, index):

        fn = self.filenames[index]
        img = np.load(fn)

        # decide if flip
        if random.random() < 0.5:
            img = np.flip(img, axis=-1)

        # random rotation
        rand_angle = random.randint(0, 180)
        # decide the new shape
        tmp = ndi.rotate(img[0, 0, :, :], rand_angle, reshape=True, order=2)
        new_img = np.zeros(
            (img.shape[0], img.shape[1], tmp.shape[0], tmp.shape[1]), dtype=np.float32
        )
        for ch in range(img.shape[0]):
            for zz in range(img.shape[1]):
                new_img[ch, zz, :, :] = ndi.rotate(
                    img[ch, zz, :, :], rand_angle, reshape=True, order=2
                )

        img = np.expand_dims(img, axis=0)
        image_tensor = torch.tensor(img.astype(np.float16))
        label_tensor = torch.tensor(np.int(self.label[index]))

        if self.test_flag:
            return image_tensor, label_tensor, fn
        else:
            return image_tensor, label_tensor

    def __len__(self):
        return len(self.filenames)
