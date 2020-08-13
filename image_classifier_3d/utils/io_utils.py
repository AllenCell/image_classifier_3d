import numpy as np
import scipy.ndimage as ndi
import random
from scipy.ndimage import zoom


def padded_adaptive_loading_single_preprocessed_cell(
    fn, out_shape, runtime_aug=0, data_type="npy"
):

    if data_type == "npy":
        img = np.load(fn)

    # if larger than out_shape, do scaling first to fit into out_shape
    if (
        img.shape[1] > out_shape[0]
        or img.shape[2] > out_shape[1]
        or img.shape[3] > out_shape[2]
    ):
        scaling_ratio = min(
            [
                out_shape[0] / img.shape[1],
                out_shape[1] / img.shape[2],
                out_shape[2] / img.shape[3],
            ]
        )
        img_temp = []
        for ch in range(img.shape[0]):
            img_temp.append(zoom(img[ch, :, :, :], scaling_ratio, order=1))
        img = np.stack(img_temp, axis=0)

    # do random padding into out_shape
    assert img.shape[1] <= out_shape[0]
    assert img.shape[2] <= out_shape[1]
    assert img.shape[3] <= out_shape[2]

    to_pad_z = out_shape[0] - img.shape[1]
    to_pad_y = out_shape[1] - img.shape[2]
    to_pad_x = out_shape[2] - img.shape[3]

    if runtime_aug > 0:

        images = []

        for ii in range(runtime_aug):

            rand_z = random.randint(
                int(round(2 * to_pad_z / 5)), int(round(3 * to_pad_z / 5))
            )
            rand_y = random.randint(
                int(round(2 * to_pad_y / 5)), int(round(3 * to_pad_y / 5))
            )
            rand_x = random.randint(
                int(round(2 * to_pad_x / 5)), int(round(3 * to_pad_x / 5))
            )

            img_pad = np.pad(
                img,
                (
                    (0, 0),
                    (rand_z, to_pad_z - rand_z),
                    (rand_y, to_pad_y - rand_y),
                    (rand_x, to_pad_x - rand_x),
                ),
                "constant",
            )

            # decide if flip
            if random.random() < 0.5:
                img_pad = np.flip(img_pad, axis=-1)

            # random rotation
            rand_angle = random.randint(0, 180)
            for ch in range(img_pad.shape[0]):
                for zz in range(img.shape[1]):
                    img_pad[ch, zz, :, :] = ndi.rotate(
                        img_pad[ch, zz, :, :], rand_angle, reshape=False, order=2
                    )

            images.append(img_pad)

        out_img = np.stack(images, axis=0)
        return out_img.astype(np.float16)

    else:
        rand_z = int(to_pad_z // 2)
        rand_y = int(to_pad_y // 2)
        rand_x = int(to_pad_x // 2)

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

        # create the batch dimension
        out_img = np.expand_dims(img, axis=0)
        return out_img.astype(np.float16)


def adaptive_loading_single_preprocessed_cell(fn, runtime_aug=0, data_type="npy"):

    if data_type == "npy":
        img = np.load(fn)

    if runtime_aug > 0:

        images = []

        for ii in range(runtime_aug):

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

            images.append(img)

        out_img = np.stack(images, axis=0)
        return out_img.astype(np.float16)

    else:

        # create the batch dimension
        out_img = np.expand_dims(img, axis=0)
        return out_img.astype(np.float16)
