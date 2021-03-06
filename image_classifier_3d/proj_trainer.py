#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import argparse
from typing import Union
from pathlib import Path
import importlib

import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class ProjectTrainer(object):
    """ Main class for training a new classifier project """

    def __init__(self, config_filename: Union[str, Path]):
        """
        Parameters
        ----------
        config_filename: Union(str,Path)
            path to the configuration file (.yaml)
        """

        # load configuration
        with open(config_filename, "r") as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                log.info(exc)

        self.config = config

    def run_trainer(self):
        """ do the training """

        hparams = argparse.Namespace(**self.config)

        # define logging
        tb_logger = pl_loggers.TensorBoardLogger("logs/")

        # initialize the model, according to project type
        build_classifier_module = importlib.import_module(
            "image_classifier_3d.models.build_classifier"
        )
        ClassiferModule = getattr(build_classifier_module, self.config["project"])
        classifier_model = ClassiferModule(hparams)
        try:
            classifier_model = ClassiferModule(hparams)
        except Exception as e:
            print(f"failed to load classifier {self.config['project']}, {e}")

        # check if need to load from existing weights
        if self.config["model_params"]["load_from"] is not None:
            classifier_model = classifier_model.load_from_checkpoint(
                self.config["model_params"]["load_from"]
            )

        # set up checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            **self.config["checkpoint_params"],
        )

        trainer = Trainer(
            callbacks=[checkpoint_callback],
            logger=tb_logger,
            **self.config["trainer_params"],
        )

        log.info(f"======= Training {self.config['project']} =======")
        trainer.fit(classifier_model)
