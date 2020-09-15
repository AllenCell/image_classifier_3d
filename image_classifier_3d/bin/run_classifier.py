#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import sys
import traceback

# import torch.backends.cudnn as cudnn

from image_classifier_3d.proj_trainer import ProjectTrainer
from image_classifier_3d.proj_tester import ProjectTester

# Global object
TRAIN_MODE = "train"
VALID_MODE = "validate"
INFER_MODE = "inference"

###############################################################################

log = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)4s:%(lineno)4s %(asctime)s] %(message)s"
)

###############################################################################


class Args(argparse.Namespace):
    def __init__(self):
        # Arguments that could be passed in through the command line
        self.debug = False
        self.__parse()

    def __parse(self):
        p = argparse.ArgumentParser(
            description="runner for training a new model",
        )
        p.add_argument(
            "--debug",
            action="store_true",
            dest="debug",
            help=argparse.SUPPRESS,
        )

        subparsers = p.add_subparsers(dest="mode")
        subparsers.required = True

        parser_train = subparsers.add_parser(TRAIN_MODE)
        parser_train.add_argument(
            "--config", dest="filename", help="configuration filename for training"
        )

        parser_validation = subparsers.add_parser(VALID_MODE)
        parser_validation.add_argument(
            "--config", dest="filename", help="configuration filename for validation"
        )

        parser_inference = subparsers.add_parser(INFER_MODE)
        parser_inference.add_argument(
            "--csv",
            dest="csv_filename",
            help="path to the csv file of data to be applied on",
        )
        parser_inference.add_argument(
            "--output_path", help="path to save prediction results"
        )

        p.parse_args(namespace=self)


###############################################################################


def main():
    try:
        args = Args()
        dbg = args.debug

        if args.mode == TRAIN_MODE:
            exe = ProjectTrainer(args.filename)
            exe.run_trainer()
        elif args.mode == VALID_MODE:
            exe = ProjectTester(save_model_output=True)
            exe.run_tester_config(args.filename)
        elif args.mode == INFER_MODE:
            exe = ProjectTester(save_model_output=False)
            exe.run_tester_csv(args.csv_filename, args.output_path)

    except Exception as e:
        log.error("=============================================")
        if dbg:
            log.error("\n\n" + traceback.format_exc())
            log.error("=============================================")
        log.error("\n\n" + str(e) + "\n")
        log.error("=============================================")
        sys.exit(1)


###############################################################################
# Allow caller to directly run this module (usually in development scenarios)

if __name__ == "__main__":
    main()
