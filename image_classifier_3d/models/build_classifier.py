"""
PyTorch Lightning model class for mitotic classifier
"""
import os

import numpy as np
import pandas as pd
import random
from glob import glob
import torch
from torch import optim
import pytorch_lightning as pl
import importlib
from typing import List, Dict

# from torchvision import transforms
# from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader

# import pdb
import torch.distributed as dist


class mitotic_classifier(pl.LightningModule):
    """ define a project class, consistent with project_name in config file """

    def __init__(self, hparams) -> None:
        super(mitotic_classifier, self).__init__()

        # load the model architecture based on model_params
        m = hparams.model_params
        if m["name"] == "resnet":
            from .resnet import generate_model

            self.model = generate_model(
                model_depth=m["depth"],
                n_classes=m["num_classes"],
                n_input_channels=m["in_channels"],
                shortcut_type="B",
                conv1_t_size=7,
                conv1_t_stride=1,
                no_max_pool=False,
                widen_factor=1.0,
            )
        elif m["name"] == "resnet-gn":
            from .resnet_GN import generate_model

            self.model = generate_model(
                model_depth=m["depth"],
                n_classes=m["num_classes"],
                n_input_channels=m["in_channels"],
                shortcut_type="B",
                conv1_t_size=7,
                conv1_t_stride=1,
                no_max_pool=False,
                widen_factor=1.0,
            )
        elif m["name"] == "densenet":
            from .densenet import generate_model

            self.model = generate_model(
                model_depth=m["depth"],
                n_classes=m["num_classes"],
                n_input_channels=m["in_channels"],
                conv1_t_size=7,
                conv1_t_stride=1,
                no_max_pool=False,
            )
        elif hparams.model_params["name"] == "resnet18_pretrain":
            from torchvision.models.video import r3d_18

            self.model = r3d_18(pretrained=True, progress=True)

        # load/initialize parameters
        self.hparams = hparams
        self.test_params = None
        self.test_results = []
        self.test_type = None
        self.using_mix_batch = False

        if "test_data_loader" not in self.hparams:  # train

            # TODO: support selecting loss function from config file

            self.exp_params = hparams.exp_params
            self.dataloader_param = hparams.exp_params["dataloader"]

            # load weight for each class
            if "class_weight" in m:
                self.class_weight = torch.tensor(m["class_weight"])
            else:
                self.class_weight = None

            # load train/valid files
            train_filenames = glob(
                hparams.exp_params["training_data_path"] + os.sep + "*.npy"
            )
            self.train_filenames = train_filenames

            val_filenames = glob(
                hparams.exp_params["validation_data_path"] + os.sep + "*.npy"
            )
            self.val_filenames = val_filenames

            assert len(self.val_filenames) > 0, "no validation file found"
            assert len(self.train_filenames) > 0, "no training file found"

        else:  # inference/evaluation
            if os.path.isfile(hparams.test_data_loader["data_path"]):
                self.test_type = "df"
                _fn, file_extension = os.path.splitext(
                    hparams.test_data_loader["data_path"]
                )
                assert file_extension == ".csv", "only csv is supported"
            else:
                self.test_type = "folder"

        # define final layer for test and evaluation
        self.final_layer = torch.nn.Softmax(dim=1)

    def forward(self, x, **kwargs):
        """ forward pass """

        if self.using_mix_batch:
            # each image in a batch may have different shapes
            # need to run one by one
            y = []
            for i, x_i in enumerate(x):
                # x_i is an image.
                y_i = self.model(x_i, **kwargs)
                if i == 0:
                    y = y_i
                else:
                    y = torch.cat((y, y_i), dim=0)
            return y
        else:
            return self.model(x, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        # get data in this batch: [input, ground_truth]
        x, y = batch

        # make forward pass
        y_hat = self(x)

        # computer loss
        if self.class_weight is None:
            loss = F.cross_entropy(y_hat, y)
        else:
            loss = F.cross_entropy(y_hat, y, self.class_weight.cuda(), reduction="mean")
        tensorboard_logs = {"train_loss": loss}

        # get prediction labels and calculate number of correct predictions
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()

        # insert info to training log
        pred_prob = y_hat.cpu().data.float()
        pred_dict = {"idx": batch_idx, "pred": pred_prob.numpy(), "label": y}

        return {
            "loss": loss,
            "log": tensorboard_logs,
            "train_n_correct_pred": n_correct_pred,
            "train_n_pred": len(x),
            "pred_record": pred_dict,
        }

    def training_epoch_end(self, outputs):

        # gather loss in this epoch
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        # gather accuracy in this epoch, average over all GPUs
        train_acc = sum([x["train_n_correct_pred"] for x in outputs]) / sum(
            x["train_n_pred"] for x in outputs
        )
        train_acc = torch.tensor(train_acc, dtype=torch.float64).cuda()
        dist.all_reduce(train_acc)
        train_acc /= dist.get_world_size()

        # insert info into training log
        tensorboard_logs = {
            "train_epoch_loss": avg_loss,
            "train_epoch_acc": train_acc.item(),
        }

        return {"loss": avg_loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx, optimizer_idx=0):

        # get data in this batch [input, ground_truth, file_name]
        # file_name can be used for debugging
        x, y, _fn_ = batch
        y_hat = self.forward(x)

        # compute loss
        if self.class_weight is None:
            val_loss = F.cross_entropy(y_hat, y)
        else:
            val_loss = F.cross_entropy(
                y_hat, y, self.class_weight.cuda(), reduction="mean"
            )

        # get prediction labels and calculate number of correct predictions
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()

        # insert info to training log
        pred_prob = y_hat.cpu().data.float()
        pred_dict = {
            "idx": batch_idx,
            "pred": [pred_prob.numpy()],
            "pred_label": labels_hat,
            "label": y,
            "fn": _fn_,
        }

        return {
            "val_loss": val_loss,
            "n_correct_pred": n_correct_pred,
            "n_pred": len(x),
            "pred_record": pred_dict,
        }

    def validation_epoch_end(self, outputs):

        # gather loss in this epoch
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        # gather accuracy in this epoch, average over all GPUs
        val_acc = sum([x["n_correct_pred"] for x in outputs]) / sum(
            x["n_pred"] for x in outputs
        )
        val_acc = torch.tensor(val_acc, dtype=torch.float64).cuda()
        dist.all_reduce(val_acc)
        val_acc /= dist.get_world_size()

        # insert info into training log
        tensorboard_logs = {"val_loss": avg_loss, "val_acc": val_acc.item()}

        # make dubug record
        if "debug_path" in self.exp_params:
            df = pd.DataFrame([x["pred_record"] for x in outputs])
            csv_fn = (
                self.exp_params["debug_path"]
                + "/"
                + str(random.randint(1, 9000))
                + ".csv"
            )
            df.to_csv(csv_fn)

        return {
            "val_loss": avg_loss,
            "val_acc": val_acc.item(),
            "log": tensorboard_logs,
        }

    def test_step(self, batch, batch_idx, optimizer_idx=0):
        if self.test_type == "folder":
            x, y, fn = batch
        elif self.test_type == "df":
            x, cid = batch
        else:
            raise NotImplementedError("unsupported test type")

        y_pred = self.forward(x)
        # labels_hat = torch.argmax(y_hat, dim=1)
        # n_correct_pred = torch.sum(y == labels_hat).item()
        y_hat = self.final_layer(y_pred)  # y_pred
        pred_prob = y_hat.cpu().data.float()

        if self.test_type == "folder":
            pred_dict = {"fn": fn, "pred": pred_prob.numpy(), "label": y}
        elif self.test_type == "df":
            pred_dict = {
                "CellId": np.asarray(cid),  # cid.cpu().data.int().numpy()
                "pred": pred_prob.numpy(),
            }
        else:
            raise NotImplementedError("unsupported test type")

        return {"n_pred": len(x), "pred_record": pred_dict}

    def test_epoch_end(self, outputs):
        df = []
        for batch_out in outputs:
            if self.test_type == "folder":
                batch_size = len(batch_out["pred_record"]["fn"])
                for data_idx in range(batch_size):
                    df.append(
                        {
                            "fn": batch_out["pred_record"]["fn"][data_idx],
                            "pred": batch_out["pred_record"]["pred"][data_idx],
                        }
                    )
            else:
                batch_size = len(batch_out["pred_record"]["CellId"])
                for data_idx in range(batch_size):
                    df.append(
                        {
                            "CellId": batch_out["pred_record"]["CellId"][data_idx],
                            "pred": batch_out["pred_record"]["pred"][data_idx],
                        }
                    )

        self.test_results.append(pd.DataFrame(df))
        return {"test": "done"}

    def configure_optimizers(self):

        optims = []
        scheds = []
        exp_m = self.exp_params

        # basic optimizer
        optimizer = optim.Adam(
            self.model.parameters(), lr=exp_m["LR"], weight_decay=exp_m["weight_decay"]
        )
        optims.append(optimizer)

        # check if using a scheduler
        if "scheduler_name" in exp_m and exp_m["scheduler_name"] is not None:
            scheduler_module = importlib.import_module("torch.optim.lr_scheduler")
            scheduler_class = getattr(scheduler_module, exp_m["scheduler_name"])
            scheduler = scheduler_class(optims[0], **exp_m["scheduler_params"])
            scheds.append(scheduler)
            return optims, scheds
        else:
            print("WARNING: no scheduler is used")
            return optims

    ###############################################################
    # define dataloader for train/test/validation
    #
    # this can be defined outside the task class, we do this because
    # the dataloaders are specific to this task.
    ################################################################
    @staticmethod
    def get_dataloader(data_m: Dict, filenames: List, flag: str):
        if data_m["name"] == "AdaptivePaddingBatch":
            from ..data_loader.universal_loader import adaptive_padding_loader

            my_loader = DataLoader(
                adaptive_padding_loader(
                    filenames, out_shape=data_m["shape"], flag=flag
                ),
                batch_size=data_m["batch_size"],
                num_workers=data_m["num_worker"],
            )
        elif data_m["name"] == "AdaptiveMixBatch":
            from ..data_loader.universal_loader import adaptive_loader
            from ..utils.misc_utils import mix_collate

            my_loader = DataLoader(
                adaptive_loader(filenames),
                batch_size=data_m["batch_size"],
                collate_fn=mix_collate,
                num_workers=data_m["num_worker"],
            )
        else:
            # assuming basic
            from ..data_loader.universal_loader import basic_loader

            my_loader = DataLoader(
                basic_loader(filenames),
                batch_size=data_m["batch_size"],
                num_workers=data_m["num_worker"],
            )

        return my_loader

    def train_dataloader(self):
        # skip this if doing testing
        if "test_data_loader" in self.hparams:
            pass

        if self.dataloader_param["name"] == "AdaptiveMixBatch":
            # each batch contains patches of different sizes
            self.using_mix_batch = True

        return mitotic_classifier.get_dataloader(
            self.dataloader_param, self.train_filenames, "train"
        )

    def val_dataloader(self):
        # skip this if doing testing
        if "test_data_loader" in self.hparams:
            pass

        return mitotic_classifier.get_dataloader(
            self.dataloader_param, self.val_filenames, "val"
        )

    def test_dataloader(self):
        # skip this if doing training
        if "test_data_loader" not in self.hparams:
            pass

        data_t = self.hparams.test_data_loader
        if self.test_type == "df":
            assert (
                data_t["name"] == "AdaptivePaddingBatch"
            ), "only adaptive paddng loader is support when using csv"
            from ..data_loader.universal_loader import adaptive_padding_loader

            test_set_loader = DataLoader(
                adaptive_padding_loader(
                    data_t["data_path"], out_shape=data_t["shape"], flag="test_csv"
                ),
                batch_size=data_t["batch_size"],
                num_workers=data_t["num_worker"],
            )

        elif self.test_type == "folder":
            filenames = glob(data_t["data_path"] + os.sep + "*.npy")
            filenames.sort()
            test_set_loader = mitotic_classifier.get_dataloader(
                data_t, filenames, "test_folder"
            )
        else:
            raise NotImplementedError("unsupported test type")

        if data_t["name"] == "AdaptiveMixBatch":
            # each batch contains patches of different sizes
            self.using_mix_batch = True

        return test_set_loader
