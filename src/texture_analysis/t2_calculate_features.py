# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 14:08:48 2022

@author: Artur Klepaczko
"""

import csv
from itertools import product
from pathlib import Path
from typing import Final

import numpy as np

from glcm_extractor import get_glcm_feature_vector


ROOT: Final[Path] = Path(R'D:\TexturesKidneys\NPY-T2')

dataset = []

class_labels = {'1': 'control', '2': 'active', '3': 'chronic'}

for category_folder in ROOT.iterdir():
    if category_folder.name not in class_labels.keys():
        continue
    class_label = class_labels[category_folder.name]
    for subject in category_folder.iterdir():
        patient_id = subject.name
        image = subject / f'{patient_id}_slice_mask.npy'
        im_arr = np.load(str(image))
        data_vector = [patient_id]
        data_vector += get_glcm_feature_vector(im_arr[:, :, 0], sigma_normalization=True)
        data_vector += [class_label]
        dataset += [data_vector]

feature_names = ['dissimilarity', 'contrast', 'correlation', 'energy', 'homogeneity', 'ASM']
distances = ['1', '3', '5']
angles = ['0', '45', '90', '135']

tex_feature_names_compounds = product(feature_names, distances, angles)

header = ['Patient_ID']

for combination in tex_feature_names_compounds:
    header += ['_'.join(combination)]
header += ['class']

with open('t2_kidney_texture_features.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(dataset)