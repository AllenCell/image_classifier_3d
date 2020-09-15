#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Imports should be grouped into:
# Standard library imports
# Related third party imports
# Local application / relative imports
# in that order

# Standard library
import logging
import argparse
from typing import Union
from pathlib import Path

import numpy as np
import yaml
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from .models import build_classifier


###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class ProjectTrainer(object):
    """
    Main class for training a new classifier project

    Parameters
    ----------
    config_filename: Union(str,Path)
        path to the configuration file (.yaml)
    """

    def __init__(self, config_filename: Union[str, Path]):

        # load configuration
        with open(config_filename, "r") as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                log.info(exc)

        self.config = config

    def run_trainer(self):
        """
        do the training
        """

        hparams = argparse.Namespace(**self.config)

        # For reproducibility
        torch.manual_seed(self.config["exp_params"]["manual_seed"])
        np.random.seed(self.config["exp_params"]["manual_seed"])
        # cudnn.deterministic = True # may hurt accuracy and speed
        # cudnn.benchmark = False

        # initialize the model
        if self.config["project"] == "mitotic_classifier":
            classifier_model = build_classifier.Mitotic_Classifier(hparams)
        elif self.configconfig["project"] == "mnist":
            from models import model_mnist

            classifier_model = model_mnist.LightningMNISTModel(hparams)
        else:
            raise ValueError(
                f"selected project {self.config['project']} is not support yet"
            )

        # check if need to load from existing weights
        if self.config["model_params"]["load_from"] is not None:
            classifier_model = classifier_model.load_from_checkpoint(
                self.config["model_params"]["load_from"]
            )

        # set up checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            save_top_k=5,
            verbose=True,
            monitor="val_acc",
            mode="max",
        )

        if self.config["project"] == "mnist":
            trainer = Trainer(
                default_root_dir=self.config["logging_params"]["save_dir"],
                checkpoint_callback=checkpoint_callback,
                gpus=hparams.gpus,
                distributed_backend=hparams.distributed_backend,
                precision=16 if hparams.use_16bit else 32,
            )
        elif self.config["project"] == "mitotic_classifier":
            trainer = Trainer(
                default_root_dir=self.config["logging_params"]["save_dir"],
                min_epochs=50,
                checkpoint_callback=checkpoint_callback,
                resume_from_checkpoint=self.config["model_params"]["resume"],
                # num_sanity_val_steps=1,
                # val_check_interval=1.0,
                # train_percent_check=1.0,
                # val_percent_check=0.1,
                **self.config["trainer_params"],
            )

        log.info(f"======= Training {self.config['project']} =======")
        trainer.fit(classifier_model)
