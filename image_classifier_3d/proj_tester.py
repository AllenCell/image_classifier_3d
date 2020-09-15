#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import logging
from typing import Union, List, Optional
from pathlib import Path
import argparse

import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
from pytorch_lightning import Trainer

from .models import build_classifier

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class ProjectTester(object):
    """
    Main class for applying a new classifier on validation data or new data
    """

    @staticmethod
    def _load_config(yaml_path: Union[str, Path]) -> List:
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file)

        hparams = argparse.Namespace(**config)
        return [config, hparams]

    @staticmethod
    def _report_results(df_merge, out_path):

        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        import matplotlib.pyplot as plt

        df_merge["true_label"] = df_merge.apply(
            lambda row: int(os.path.basename(row.fn)[0]), axis=1
        )

        # TODO: need to make this plot customizable
        conf = confusion_matrix(
            df_merge["true_label"],
            df_merge["pred_label"],
            labels=np.array([0, 1, 2, 3, 4, 5]),
        )
        plt.figure()
        disp = ConfusionMatrixDisplay(
            confusion_matrix=conf,
            display_labels=["M0", "M12", "M3", "M45", "M67_e", "M6M7_h"],
        )
        disp = disp.plot(
            include_values=True, xticks_rotation="horizontal", cmap="viridis", ax=None
        )
        plt.show()
        plt.savefig(Path(out_path) / "cf.png")

        df_merge.to_csv(Path(out_path) / "pred.csv")

    @staticmethod
    def _merge_results_for_new(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        merge the results in csv
        """
        df_merge = df[0].copy()
        if len(df) > 1:
            for df_idx, df_x in enumerate(df):
                if df_idx == 0:
                    continue
                else:
                    for row in df_x.itertuples():
                        cid = row.CellId
                        df_merge.at[df_merge["CellId"] == cid, "pred"] = (
                            df_merge.loc[df_merge["CellId"] == cid]["pred"].values
                            + df_x.loc[df_x["CellId"] == cid]["pred"].values
                        )

        # generate the final prediction
        df_merge["pred_label"] = df_merge.apply(
            lambda row: np.argmax(row.pred, axis=0), axis=1
        )
        return df_merge

    @staticmethod
    def _merge_results_for_validation(
        df: List[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        merge the results in csv
        """
        df_merge = df[0].copy()
        if len(df) > 1:
            for df_idx, df_x in enumerate(df):
                if df_idx == 0:
                    continue
                else:
                    for row in df_x.itertuples():
                        fn = row.fn
                        df_merge.at[df_merge["fn"] == fn, "pred"] = (
                            df_merge.loc[df_merge["fn"] == fn]["pred"].values
                            + df_x.loc[df_x["fn"] == fn]["pred"].values
                        )

        # generate the final prediction
        df_merge["pred_label"] = df_merge.apply(
            lambda row: np.argmax(row.pred, axis=0), axis=1
        )
        return df_merge

    @staticmethod
    def _run_prediction(config, hparams, output_path) -> List[pd.DataFrame]:
        """
        run prediction [TBA]
        """

        device = torch.device("cuda:0")
        trainer = Trainer(**config["trainer_params"])

        # run through the list of models (maybe > 1, as ensemble)
        df = []
        for model_item in config["model_params"]["trained_model"]:

            classifier_model = None
            if config["project"] == "mitotic_classifier":
                classifier_model = build_classifier.Mitotic_Classifier(hparams)
            else:
                raise ValueError(
                    f"selected project {config['project']} is not support yet"
                )

            state = torch.load(model_item["model"], map_location=device)
            classifier_model.load_state_dict(state["state_dict"])
            classifier_model.hparams = hparams

            # move to gpu
            classifier_model.eval()
            # classifier_model.freeze()
            # classifier_model.half() # only need when NOT using trainer.test()
            classifier_model.to(device)
            # final_layer = torch.nn.Softmax(dim=1)
            # final_layer.to(device)

            test_params = config["test_data"]
            for test_iter in tqdm(range(test_params["runtime_aug"])):
                trainer.test(classifier_model)

            df_this_model = classifier_model.test_results[0]
            df.append(df_this_model)

            out_fn = (
                Path(output_path) / f"mitotic_prediction_result_model_{len(df)}.csv"
            )
            df_this_model.to_csv(out_fn)

        return df

    def __init__(self, save_model_output: bool = False):

        self.save_model_output = save_model_output

    def run_tester_csv(
        self,
        csv_filename: Union[str, Path],
        output_path: Union[str, Path],
        return_df: bool = False,
        project_name: str = "mitotic_classifier",
    ) -> Optional[pd.DataFrame]:
        """
        do testing using data from a csv file
        """
        predefined_yaml = (
            Path(__file__).parent / f"../model_zoo/test_config_{project_name}.yaml"
        )
        [config, hparams] = self._load_config(predefined_yaml)

        # update data path with the csv filename
        config["test_data"]["data_path"] = csv_filename

        # run prediction
        df = self._run_prediction(config, hparams, output_path)

        # collect the results
        df_merge = ProjectTester._merge_results_for_new(df)

        # save the results
        if return_df:
            return df_merge
        else:
            df_merge.to_csv(Path(output_path) / "mitotic_prediction_result.csv")

    def run_tester_config(self, config_filename: Union[str, Path]):
        """
        do testing using a config file (.yaml)
        """

        [config, hparams] = self._load_config(config_filename)
        out_path = config["test_data"]["output_path"]

        # run prediction
        df = self._run_prediction(config, hparams, out_path)

        # collect the results
        df_merge = ProjectTester._merge_results_for_validation(df)

        # save validation results
        self._report_results(df_merge, out_path)
