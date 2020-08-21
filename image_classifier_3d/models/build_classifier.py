import os
import sys

# import numpy as np
import pandas as pd
import random
from glob import glob
import torch
from torch import optim
import pytorch_lightning as pl

# from torchvision import transforms
# from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader

# import pdb
import torch.distributed as dist


class Mitotic_Classifier(pl.LightningModule):
    def __init__(self, hparams) -> None:
        super(Mitotic_Classifier, self).__init__()

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
        self.exp_params = hparams.exp_params
        self.dataloader_param = hparams.exp_params["dataloader"]
        self.test_type = None

        if "test_data" not in self.hparams:  # train

            # TODO: support selecting loss function from config file

            # load weight for each class
            self.class_weight = torch.tensor(m["class_weight"])

            # load train/valid files
            train_filenames = glob(
                hparams.exp_params["training_data_path"] + os.sep + "*.npy"
            )
            self.train_filenames = train_filenames

            val_filenames = glob(
                hparams.exp_params["validation_data_path"] + os.sep + "*.npy"
            )
            self.val_filenames = val_filenames

            assert len(self.val_filenames) > 0
            assert len(self.train_filenames) > 0

        else:  # testing
            if os.path.isfile(hparams.test_data["data_path"]):
                _fn, file_extension = os.path.splitext(hparams.test_data["data_path"])
                assert file_extension == ".csv", "only csv is supported"
                self.test_type = "df"
            else:
                self.test_type = "folder"

        # define final layer for test and validation
        self.final_layer = torch.nn.Softmax(dim=1)
        self.final_layer.to(torch.device("cuda:0"))

    def forward(self, x, **kwargs):

        if self.dataloader_param["name"] == "AdaptiveMixBatch":
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
        x, y = batch
        y_hat = self(x)
        if self.class_weight is None:
            loss = F.cross_entropy(y_hat, y)
        else:
            loss = F.cross_entropy(y_hat, y, self.class_weight.cuda(), reduction="mean")
        tensorboard_logs = {"train_loss": loss}

        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()

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
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        train_acc = sum([x["train_n_correct_pred"] for x in outputs]) / sum(
            x["train_n_pred"] for x in outputs
        )

        train_acc = torch.tensor(train_acc, dtype=torch.float64).cuda()

        dist.all_reduce(train_acc)
        train_acc /= dist.get_world_size()

        tensorboard_logs = {
            "train_epoch_loss": avg_loss,
            "train_epoch_acc": train_acc.item(),
        }

        return {"loss": avg_loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        x, y, _fn_ = batch
        y_hat = self.forward(x)

        # val_loss = self.criterion(y_hat, y)
        if self.class_weight is None:
            val_loss = F.cross_entropy(y_hat, y)
        else:
            val_loss = F.cross_entropy(
                y_hat, y, self.class_weight.cuda(), reduction="mean"
            )

        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()

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
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_acc = sum([x["n_correct_pred"] for x in outputs]) / sum(
            x["n_pred"] for x in outputs
        )

        val_acc = torch.tensor(val_acc, dtype=torch.float64).cuda()

        dist.all_reduce(val_acc)
        val_acc /= dist.get_world_size()

        tensorboard_logs = {"val_loss": avg_loss, "val_acc": val_acc.item()}

        # make dubug record
        df = pd.DataFrame([x["pred_record"] for x in outputs])
        csv_fn = (
            self.exp_params["debug_path"]
            + "val/"
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
        else:
            x, cid = batch

        y_pred = self.forward(x)
        # labels_hat = torch.argmax(y_hat, dim=1)
        # n_correct_pred = torch.sum(y == labels_hat).item()
        y_hat = self.final_layer(y_pred)  # y_pred
        pred_prob = y_hat.cpu().data.float()

        if self.test_type == "folder":
            pred_dict = {"fn": fn, "pred": pred_prob.numpy(), "label": y}
        else:
            pred_dict = {
                "CellId": cid.cpu().data.int().numpy(),
                "pred": pred_prob.numpy(),
            }

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

        if exp_m["scheduler"]["name"] is not None:
            if exp_m["scheduler"]["name"] == "ExponentialLR":
                from torch.optim.lr_scheduler import ExponentialLR

                assert exp_m["scheduler"]["gamma"] > 0
                scheduler = ExponentialLR(optims[0], gamma=exp_m["scheduler"]["gamma"])

            elif exp_m["scheduler"]["name"] == "CosineAnnealingLR":
                from torch.optim.lr_scheduler import CosineAnnealingLR as CALR

                assert exp_m["scheduler"]["T_max"] > 0
                scheduler = CALR(optims[0], T_max=exp_m["scheduler"]["T_max"])

            elif exp_m["scheduler"]["name"] == "StepLR":
                from torch.optim.lr_scheduler import StepLR

                assert exp_m["scheduler"]["step_size"] > 0
                assert exp_m["scheduler"]["gamma"] > 0
                scheduler = StepLR(
                    optims[0],
                    step_size=exp_m["scheduler"]["step_size"],
                    gamma=exp_m["scheduler"]["gamma"],
                )

            scheds.append(scheduler)
            return optims, scheds
        else:
            print("no scheduler is used")
            return optims

    def train_dataloader(self):
        # ## may load customized data transformation function here
        # transform = self.data_transforms()

        data_m = self.dataloader_param
        # train_set_loader = None
        if data_m["name"] == "AdaptivePaddingBatch":
            from ..data_loader.universal_loader import adaptive_padding_loader

            train_set_loader = DataLoader(
                adaptive_padding_loader(
                    self.train_filenames, out_shape=data_m["shape"]
                ),
                batch_size=data_m["batch_size"],
                num_workers=data_m["num_worker"],
            )
        elif data_m["name"] == "AdaptiveMixBatch":
            from ..data_loader.universal_loader import adaptive_loader
            from ..utils.misc_utils import mix_collate

            train_set_loader = DataLoader(
                adaptive_loader(self.train_filenames),
                batch_size=data_m["batch_size"],
                collate_fn=mix_collate,
                num_workers=data_m["num_worker"],
            )
        else:
            # assuming basic
            from ..data_loader.universal_loader import basic_loader

            train_set_loader = DataLoader(
                basic_loader(self.train_filenames),
                batch_size=data_m["batch_size"],
                num_workers=data_m["num_worker"],
            )

        return train_set_loader

    def val_dataloader(self):
        # ## may load customized data transformation function here
        # transform = self.data_transforms()

        data_m = self.dataloader_param
        # val_set_loader = None
        if data_m["name"] == "AdaptivePaddingBatch":
            from ..data_loader.universal_loader import adaptive_padding_loader

            val_set_loader = DataLoader(
                adaptive_padding_loader(
                    self.val_filenames, out_shape=data_m["shape"], test_flag="F"
                ),
                batch_size=data_m["batch_size"],
                num_workers=data_m["num_worker"],
            )
        elif data_m["name"] == "AdaptiveMixBatch":
            from ..data_loader.universal_loader import adaptive_loader
            from ..utils.misc_utils import mix_collate

            val_set_loader = DataLoader(
                adaptive_loader(self.val_filenames),
                batch_size=data_m["batch_size"],
                collate_fn=mix_collate,
                num_workers=data_m["num_worker"],
            )
        else:
            # assume basic
            from ..data_loader.universal_loader import basic_loader

            val_set_loader = DataLoader(
                basic_loader(self.val_filenames),
                batch_size=data_m["batch_size"],
                num_workers=self.dataloader_param["num_worker"],
            )

        return val_set_loader

    def test_dataloader(self):

        if "test_data" not in self.hparams:
            pass

        data_m = self.dataloader_param
        data_t = self.hparams.test_data
        test_set_loader = None
        if self.test_type == "df":
            assert (
                data_m["name"] == "AdaptivePaddingBatch"
            ), "only adaptive paddng loader is support when using csv"
            from ..data_loader.universal_loader import adaptive_padding_loader

            test_set_loader = DataLoader(
                adaptive_padding_loader(
                    data_t["data_path"], out_shape=data_m["shape"], test_flag="C"
                ),
                batch_size=data_t["batch_size"],
                num_workers=data_t["num_worker"],
            )

        elif self.test_type == "folder":
            filenames = glob(data_t["data_path"] + os.sep + "*.npy")
            filenames.sort()

            if data_m["name"] == "AdaptivePaddingBatch":
                from ..data_loader.universal_loader import adaptive_padding_loader

                test_set_loader = DataLoader(
                    adaptive_padding_loader(
                        filenames, out_shape=data_m["shape"], test_flag="F"
                    ),
                    batch_size=data_t["batch_size"],
                    num_workers=data_t["num_worker"],
                )
            elif data_m["name"] == "AdaptiveMixBatch":
                from ..data_loader.universal_loader import adaptive_loader
                from ..utils.misc_utils import mix_collate

                test_set_loader = DataLoader(
                    adaptive_loader(filenames),
                    batch_size=data_t["batch_size"],
                    collate_fn=mix_collate,
                    num_workers=data_t["num_worker"],
                    test_flag=True,
                )
            else:
                # asume basic
                from ..data_loader.universal_loader import basic_loader

                test_set_loader = DataLoader(
                    basic_loader(filenames),
                    batch_size=data_t["batch_size"],
                    num_workers=data_t["num_worker"],
                )
        else:
            print("unsupported test type")
            sys.exit(0)

        return test_set_loader

    """
    # an example of customized data transformation
    def data_transforms(self):

        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))

        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.CenterCrop(148),
                                        transforms.Resize(self.params['img_size']),
                                        transforms.ToTensor(),
                                        SetRange])

        return transform
    """
    """
    def prepare_data(self):

        exp_m = self.exp_params
        # do train/valid split
        train_filenames = glob(exp_m['training_data_path'] + os.sep + '*.npy')
        # random.shuffle(train_filenames)
        self.train_filenames = train_filenames

        val_filenames = glob(exp_m['validation_data_path'] + os.sep + '*.npy')
        # random.shuffle(val_filenames)
        self.val_filenames = val_filenames

        assert len(self.val_filenames) > 0
        assert len(self.train_filenames) > 0

        # total_num = len(filenames)
        # num_val = int(np.floor(exp_m['val_ratio'] * total_num))
        # self.val_filenames = filenames[:num_val]
        # self.train_filenames = filenames[num_val:]
    """
