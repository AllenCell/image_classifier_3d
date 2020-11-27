#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch


def mix_collate(batch):
    """
    A util function for handling data when images in each mini-batch
    have different sizes. Conventionally, all images in each mini-batch
    during training should have the same size. This is a work-around.
    Inspired by https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3  # noqa E501
    """
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]
