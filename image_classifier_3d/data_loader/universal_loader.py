import os
import numpy as np
import pandas as pd
import random
from typing import List, Union
import importlib

from scipy.ndimage import zoom
import scipy.ndimage as ndi
import torch
from torch.utils.data import Dataset


class basic_loader(Dataset):
    """
    Basic DataLoader:

        Only support problem with no more than 10 classes. All files are 
        in .npy format instead of images. During training, all images will 
        only be loaded when they are being used in a training iteration. 
        Only class labels are pre-loaded, no images will be pre-loaded 
        (ideal for large dataset). During inference, currently basic 
        dataloader only take preprocessed images as .npy files. This will
        be improved for more flexible data loading
    """

    def __init__(self, filenames: List):
        """
        Parameters:
        -------------
        filenames: List
            a list of filenames for all data. Every filename has the format
            X_CELLID.npy, where X can be any integer from 0 to num_class-1
            (assuming num_class <= 10), abd CELLID is a unique name for the
            cell (e.g., using uuid).
        """

        self.img = []
        self.label = []

        print("initializing data loader ...")
        self.filenames = filenames
        self.label = [int(os.path.basename(fn)[0]) for fn in filenames]
        print("data loader initialization is done")

    def __getitem__(self, index):

        img = np.load(self.filenames[index])
        image_tensor = torch.tensor(img.astype(np.float16))
        label_tensor = torch.tensor(np.int(self.label[index]))

        return image_tensor, label_tensor

    def __len__(self):
        return len(self.filenames)


class adaptive_padding_loader(Dataset):
    """
    Adaptive padding DataLoader:

        Adaptive padding data loader will pad all images to the same size 
        defined by "out_shape" when constructing the data loader. During 
        training, random flip and rotaion will be applied. No augmentation
        for testing or evaluation. In addition, all images will 
        only be loaded when they are being used in a training iteration. 
        Only class labels are pre-loaded, no images will be pre-loaded 
        (ideal for large dataset).
    """

    def __init__(
        self,
        filenames: Union[List[str], str],
        out_shape: List = [64, 128, 128],
        flag: str = "train",
        building_wrapper_path: str = "image_classifier_3d.data_loader.utils",
        building_func_name: str = "build_one_cell",
    ):
        """
        Parameters:
        -------------
        filenames: Union[List[str], str]
            This could be a filename (only csv file supported) or a list of 
            filenames for all data. For the later case, every filename has 
            the format X_CELLID.npy, where X can be any integer from 0 to 
            num_class-1 (assuming num_class <= 10), and CELLID is a unique 
            name for the cell (e.g., using uuid).

        out_shape: List
            the size of which all input images will be padded into. If an image
            is larger than out_shape, it will be resized down to fit under 
            out_shape, and then padded to out_shape.

        flag: str
            "flag" is a key parameter for determining how data loadinh works
            in different scenarios: "train" | "val" | "test_csv" | "test_folder".

            When flag == "train" :

            All data should be saved in a folder with filenames in the format 
            X_CELLID.npy (see detail above). Random flip and random rotation 
            in XY plane are used for data augmentation.

            when flag == "val":

            All data should be saved in a folder with filenames in the format 
            X_CELLID.npy (see detail above). No data augmentation.

            when flag == "test_csv":

            Filenames should be the path to a csv file with record of all cells.
            The csv file should contains at least three columns, "CellId",
            "crop_raw" and "crop_seg". The last two are the read paths for 
            raw image and segmentation. "crop_raw" assumes a 4D image tiff file
            (multi-channel z-stack, channel order: 0 = dna, 1 = mem, other
            channels will not be used). "crop_seg" assumes a 4D image tiff file
            (multi-channel z-stack, channel order: 0 = dna segmentation, 
            1 = cell segmentation, other channels will not be used). If a file 
            with name "for_mito_prediction.npy" exists under the same
            folder as "crop_raw", then it will be directly loaded and used
            as input to your model. Otherwise, buildinng_wrapper_path and
            building_func_name will be used to load a function defining how
            to prepare the input data using crop_raw and crop_seg. For example,
            you can have a file "C:/projects/demo/preprocessing.py" with a
            function called "my_preprocessing" defined in the script. Then,
            buildinng_wrapper_path = "C:/projects/demo/preprocessing.py" and
            building_func_name = "my_preprocessing".

            when flag == "test_folder":

            All data should be saved in a folder with filenames in the format 
            X_CELLID.npy (see detail above). No data augmentation.

        buildinng_wrapper_path: str
            where to load the wrapper for building one cell (see above when
            flag == "train_csv")

        building_func_name: str
            the function to load for building one cell (see above when
            flag == "train_csv")
        """

        self.img = []
        self.label = []
        self.out_shape = out_shape
        self.flag = flag

        if flag == "train" or flag == "val" or flag == "test_folder":
            print("initializing data loader ...")
            self.filenames = filenames
            self.label = [int(os.path.basename(fn)[0]) for fn in filenames]
            print("data loader initialization is done")

        elif flag == "test_csv":  # CSV (i.e., testing by csv)
            df = pd.read_csv(filenames)
            self.df = df.reset_index(drop=True)

            # load module from a customized wrapper
            if os.path.exists(building_wrapper_path):
                spec = importlib.util.spec_from_file_location(
                    building_func_name,
                    building_wrapper_path,
                )
                self.process_image = importlib.util.module_from_spec(spec)
            # default module
            else:
                module_name = importlib.import_module(building_wrapper_path)
                self.process_image = getattr(module_name, building_func_name)

        else:
            raise NotImplementedError(f"unsupported type: {flag}")

    def __getitem__(self, index):

        if self.flag == "test_csv":
            # build the cell from raw/seg
            crop_raw = self.df["crop_raw"].iloc[index]
            crop_seg = self.df["crop_seg"].iloc[index]

            npy_fn = os.path.dirname(crop_raw) + "/for_mito_prediction.npy"

            if os.path.exists(npy_fn):
                img = np.load(npy_fn)
            else:
                img = self.process_image(crop_raw, crop_seg)
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

        if not self.flag == "train":  # not training, then center pad
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

        if self.flag == "train":  # training, then do augmentation
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

        if self.flag == "test_csv":
            return image_tensor, self.df["CellId"].iloc[index]
        elif self.flag == "test_folder" or self.flag == "val":
            label_tensor = torch.tensor(np.int(self.label[index]))
            return image_tensor, label_tensor, fn
        elif self.flag == "train":
            label_tensor = torch.tensor(np.int(self.label[index]))
            return image_tensor, label_tensor

    def __len__(self):
        if self.flag == "train" or self.flag == "val" or self.flag == "test_folder":
            return len(self.filenames)
        elif self.flag == "test_csv":
            return len(self.df)


class adaptive_loader(Dataset):
    """
    Adaptive DataLoader:

        Adaptive data loader will collect images of different sizes
        into mini-batches. No padding applied. Random flip and rotaion will be
        applied, for all training, testing or evaluation.

        All training data should be saved in a folder with filenames of 
        format X_CELLID.npy, where X can be any integer from 0 to num_class-1
        (assuming num_class <= 10), and CELLID is a unique name for the cell
        (e.g., using uuid). All images will only be loaded when they are being 
        used in a training iteration. Only class labels are pre-loaded, no
        images will be pre-loaded (ideal for large dataset). During inference,
        currently only preprocessed images as .npy files are supported. 
        This will be improved for more flexible data loading
    """

    def __init__(self, filenames: List, test_flag=False):
        """
        Parameters:
        -------------
        filenames: List
            a list of filenames for all data. Every filename has the format
            X_CELLID.npy, where X can be any integer from 0 to num_class-1
            (assuming num_class <= 10), abd CELLID is a unique name for the
            cell (e.g., using uuid).
        test_flag: bool
            when for test_dataloader, default is False. When testing, filename
            will be returned in a batch
        """

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
