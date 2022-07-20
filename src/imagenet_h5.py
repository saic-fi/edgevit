# Copyright (c) Facebook, Inc. and its affiliates.

import json
import numpy as np
import os
import random
import re
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms as transforms_tv

import h5py, io, time
import os.path as osp


class Imagenet(torch.utils.data.Dataset):
    """ImageNet dataset."""

    def __init__(self, root_path, train=True, transform=None):
        if train:
            self.mode = 'train'
        else:
            self.mode = 'val'
        self.data_path = root_path

        # logger.info("Constructing ImageNet {}...".format(mode))
        self._construct_imdb_h5()
        self.transform = transform

    def safe_record_loader(self, raw_frame, attempts=10, retry_delay=1):
        for j in range(attempts):
            try:
                img = Image.open(io.BytesIO(raw_frame)).convert('RGB')
                return img
            except OSError as e:
                print(f'Attempt {j}/{attempts}: failed to load\n{e}', flush=True)
                if j == attempts - 1:
                    raise
            time.sleep(retry_delay)

    def _construct_imdb_h5(self):
        # def __init__(self, h5_path, transform=None):
        # self.h5_fp = h5_path
        self.h5_fp = os.path.join(self.data_path, self.mode+'.h5')
        # logger.info("{} data path: {}".format(self.mode, self.h5_fp))

        assert osp.isfile(self.h5_fp), "File not found: {}".format(self.h5_fp)
        self.h5_file = None
        h5_file = h5py.File(self.h5_fp, 'r')
        self._imdb = []
        # labels = list(h5_file.keys())
        labels = sorted(list(h5_file.keys()))
        for key, value in h5_file.items():
            target = labels.index(key)
            for img_name in value.keys():
                self._imdb.append({'image_name': img_name, 'class_name': key, 'class': target})
        self.num_videos = len(self._imdb)
        # logger.info("Number of images: {}".format(len(self._imdb)))
        # logger.info("Number of classes: {}".format(len(labels)))

    def __load_h5__(self, index):
        try:
            # Load the image
            if self.h5_file is None:
                self.h5_file = h5py.File(self.h5_fp, 'r')
            record = self._imdb[index]
            raw_frame = self.h5_file[record['class_name']][record['image_name']][()]
            img = self.safe_record_loader(raw_frame)
            return img
        except Exception:
            return None

    def __getitem__(self, index):
        im = self.__load_h5__(index)
        if self.transform is not None:
            im = self.transform(im)
        # Retrieve the label
        label = self._imdb[index]["class"]
        return im, label

    def __len__(self):
        return len(self._imdb)
