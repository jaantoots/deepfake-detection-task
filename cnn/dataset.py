"""Deepfake dataset"""
import glob

# pylint: disable=too-few-public-methods
import logging
from enum import Enum
from pathlib import Path

import torch.utils.data as data
from PIL import Image


class Split(Enum):
    """Dataset splits"""

    TRAIN = ("train",)
    VAL = ("val",)


class Faces(data.Dataset):
    """Dataset of faces"""

    def __init__(self, root, tf=None, split=Split.TRAIN):
        self.root = Path(root)
        self.transform = tf
        self.split = split
        self.data = list(self._files())

    def _files(self):
        for split in self.split.value:
            for name, label in (("real", 0), ("fake", 1)):
                for filename in glob.glob(str(self.root / split / name / "*.jpg")):
                    yield filename, label

    def __getitem__(self, idx):
        filename, label = self.data[idx]
        try:
            image = Image.open(filename).convert("RGB")
        except OSError as err:
            logging.warning(
                "Unable to open image (%d, '%s', '%s'): %s", idx, label, filename, err,
            )
            image = Image.new("RGB", (224, 224))
        if self.transform is not None:
            image = self.transform(image)
        return image, label, filename

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = f"Dataset {self.__class__.__name__} {self.split.name}\n"
        fmt_str += f"    Number of datapoints: {self.__len__()}\n"
        fmt_str += f"    Number of real: {len([x for _, x in self.data if x == 0])}\n"
        fmt_str += f"    Number of fake: {len([x for _, x in self.data if x == 1])}\n"
        fmt_str += f"    Root location: {self.root}"
        return fmt_str
