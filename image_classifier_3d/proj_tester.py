#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import logging
from typing import Union, List, Optional
from pathlib import Path
import argparse
import importlib

import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
from pytorch_lightning import Trainer

from image_classifier_3d.utils.quilt_utils import validate_model

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class ProjectTester(object):
    """
    Main class for applying a new classifier on evaluation data or new data
    """

    def __init__(self, mode: str = "inference", save_model_output: bool = False):
        """initialization

        Parameters:
        -------------
        mode: str
            running mode, either "inference" (no ground truth) or "evaluation"
            (ground truth is available)
        save_model_output: bool
            whether to save the raw prediction from model(s)
        """
        self.mode = mode
        self.save_model_output = save_model_output

    @staticmethod
    def _load_config(yaml_path: Union[str, Path]) -> List:
        """ load configuration from yaml file """
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file)

        hparams = argparse.Namespace(**config)
        return [config, hparams]

    @staticmethod
    def _report_results(df_merge, out_path, class_label):
        """ report evaluation results, confusion matrix """

        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        import matplotlib.pyplot as plt

        num_label = len(class_label)

        # pull class label from filename, assume the first character (0-9) is
        # the class index
        df_merge["true_label"] = df_merge.apply(
            lambda row: int(os.path.basename(row.fn)[0]), axis=1
        )

        # generate confusion matrix
        conf = confusion_matrix(
            df_merge["true_label"],
            df_merge["pred_label"],
            labels=np.arange(num_label),
        )
        plt.figure()
        disp = ConfusionMatrixDisplay(
            confusion_matrix=conf,
            display_labels=class_label,
        )
        disp = disp.plot(
            include_values=True, xticks_rotation="horizontal", cmap="viridis", ax=None
        )
        plt.show()
        plt.savefig(Path(out_path) / "cf.png")

    @staticmethod
    def _merge_results_for_new(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge the results in csv when dealing with new data. The dataframe is
        organized by "CellId" (may not have class label embeded).

        Parameters:
        ------------
        df: List[pd.DataFrame]
            a list of dataframes of predictions from each model (>=1)

        Return:
        ------------
        a dataframe of the merged results of probabilities and translate to class index
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
    def _merge_results_for_evaluation(
        df: List[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Merge the results in csv for evaluation. The class labels are supposed to
        be embeded in the filename (i.e., 'fn'), in order to compare the prediction
        and ground truth


        Parameters:
        ------------
        df: List[pd.DataFrame]
            a list of dataframes of predictions from each model (>=1)

        Return:
        ------------
        a dataframe of the merged results of probabilities and translate to class index
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
    def _run_prediction(
        config,
        hparams,
        mode,
        output_path,
        save_model_output: bool = False,
    ) -> List[pd.DataFrame]:
        """core function for running the prediction

        Parameters:
        --------------
        config:
            the configuration dictionary
        hparams:
            Argparse.Namespace object for model hyperparameters
        output_path:
            where to save the predictions of each model
        save_model_output: bool
            whether to save the prediction from each individual model,
            default is False
        """

        trainer = Trainer(**config["trainer_params"])

        # run through the list of models (maybe > 1, as ensemble)
        df = []
        for model_item in config["model_params"]["trained_model"]:

            # define the model as a PLT module
            build_classifier_module = importlib.import_module(
                "image_classifier_3d.models.build_classifier"
            )
            ClassiferModule = getattr(build_classifier_module, config["project"])
            try:
                classifier_model = ClassiferModule(hparams)
            except Exception as e:
                print(f"failed to load classifier {config['project']}, {e}")

            # load trained model weights
            state = torch.load(model_item["model"])
            classifier_model.load_state_dict(state["state_dict"])
            classifier_model.hparams = hparams

            # move to gpu
            classifier_model.eval()
            # classifier_model.to(device)

            # check if runtime augmentation is requested. Depending on the dataloader,
            # the runtime augmentation may or may not be helpful. For example, if some
            # random padding is used, then runtime augmentation will be beneficial.
            test_params = config["test_data_loader"]
            for test_iter in tqdm(range(test_params["runtime_aug"])):
                trainer.test(classifier_model)

            df_this_model = classifier_model.test_results[0]
            df.append(df_this_model)

            if save_model_output:
                # save the prediction from each individual models for debug
                out_fn = Path(output_path) / f"prediction_result_model_{len(df)}.csv"
                df_this_model.to_csv(out_fn)

        # collect the results
        if mode == "evaluation":
            df_merge = ProjectTester._merge_results_for_evaluation(df)
            # save evaluation results
            ProjectTester._report_results(
                df_merge, output_path, config["model_params"]["class_label"]
            )
        elif mode == "inference":
            df_merge = ProjectTester._merge_results_for_new(df)
        else:
            raise NotImplementedError("either inference or evaluation mode")

        return df_merge

    def run_tester_csv(
        self,
        csv_filename: Union[str, Path],
        output_path: Union[str, Path],
        return_df: bool = False,
        project_name: str = "mitotic_classifier",
        config_yaml: Union[str, Path] = "default",
    ) -> Optional[pd.DataFrame]:
        """do testing using data from a csv file

        Parameters:
        -------------
        csv_filename: Union[str, Path]
            the filepath to the csv file
        output_path: Union[str, Path]
            where to save the prediction outputs (as a CSV file)
        return_df: bool
            whether a dataframe will be returned besides saving to CSV, default is False
        project_name: str
            which project (i.e., which kind of classifier) to use
        config_yaml: Union[str, Path]
            the configuration file to use, if it is "default", the a yaml at
            ../model_zoo/test_config_{project_name}.yaml is used by default
        """
        if config_yaml == "default":
            config_yaml = (
                Path(__file__).parent / f"../model_zoo/test_config_{project_name}.yaml"
            )
        [config, hparams] = self._load_config(config_yaml)

        # validate if the models exist locally, download when needed and saved to
        # a temp location "_local_model".
        validate_model(config, hparams, output_path / Path("_local_model"))

        # update data path with the csv filename
        config["test_data_loader"]["data_path"] = csv_filename

        # run prediction
        df_merge = self._run_prediction(
            config,
            hparams,
            mode=self.mode,
            output_path=output_path,
            save_model_output=self.save_model_output,
        )

        # save the results
        if return_df:
            return df_merge
        else:
            df_merge.to_csv(
                Path(output_path) / f"{project_name}_result.csv", index=False
            )

    def run_tester_config(
        self,
        config_filename: Union[str, Path],
        output_path: Union[str, Path],
        return_df: bool = False,
    ):
        """do testing using a config yaml file

        Parameters:
        -------------
        config_filename: Union[str, Path]
            the configuration file to use
        output_path: Union[str, Path]
            where to save the prediction outputs (as a CSV file)
        return_df: bool
            whether a dataframe will be returned besides saving to CSV, default is False
        """

        [config, hparams] = self._load_config(config_filename)

        # run prediction
        df_merge = self._run_prediction(
            config,
            hparams,
            mode=self.mode,
            output_path=output_path,
            save_model_output=self.save_model_output,
        )

        # save the results
        if return_df:
            return df_merge
        else:
            df_merge.to_csv(Path(output_path) / "prediction_result.csv", index=False)
