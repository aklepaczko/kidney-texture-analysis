from pathlib import Path
import random
from typing import Final, Optional

import cv2
import numpy as np
import torch
from torch.utils import data
from torchvision.transforms import v2

IMG_SIZE: Final[int] = 224


class DixonVibeDataset(data.Dataset):

    def __init__(self,
                 root_path: Path,
                 patients: list[int],
                 categories: Optional[list],
                 augment: bool = True):
        if categories is None:
            categories = [1, 2, 3]
        self.categories = categories
        self.image_files = self._prepare_image_list(root_path, patients, categories)
        if augment:
            self.transform = v2.Compose([
                v2.ToTensor(),
                v2.Resize(size=(IMG_SIZE, IMG_SIZE)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(degrees=10),
                v2.RandomVerticalFlip(p=0.5),
                v2.ToDtype(torch.float32),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                # v2.Normalize(mean=[0.544, 0.565, 0.645], std=[0.145, 0.135, 0.123])
            ])
        else:
            self.transform = v2.Compose([
                v2.ToTensor(),
                v2.Resize(size=(IMG_SIZE, IMG_SIZE)),
                v2.ToDtype(torch.float32),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                # v2.Normalize(mean=[0.544, 0.565, 0.645], std=[0.145, 0.135, 0.123])
            ])

    @staticmethod
    def _prepare_image_list(root_path: Path,
                            patients: list[int],
                            categories: Optional[list]) -> list[tuple[Path, int]]:
        image_filenames = []
        image_categories = []
        for category in root_path.iterdir():
            if int(category.stem) not in categories:
                continue
            for patient in category.iterdir():
                if int(patient.stem[2:]) not in patients:
                    continue
                patient_list = list(patient.glob('*.png'))
                image_filenames += patient_list
                image_categories += [category] * len(patient_list)
        data_list = list(zip(image_filenames, image_categories))
        random.shuffle(data_list)
        return data_list

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        filename, category = self.image_files[item]
        # im_arr = np.load(str(filename))  # used in case images stored as *.npy files
        im_arr = cv2.imread(str(filename))
        im_arr = cv2.cvtColor(im_arr, cv2.COLOR_BGR2RGB)
        return self.transform(im_arr), self.categories.index(int(category.stem))
