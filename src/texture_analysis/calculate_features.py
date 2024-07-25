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
from shape_extractor import get_shape_feature_vector


ROOT: Final[Path] = Path(R'D:\TexturesKidneys\NPY_v2')

dataset = []

class_labels = {'1': 'control', '2': 'active', '3': 'chronic'}
channels = ('opp', 'in', 'W')
for category_folder in ROOT.iterdir():
    class_label = class_labels[category_folder.name]
    for subject in category_folder.iterdir():
        patient_id = subject.name
        subject_slices = list(subject.glob('*.npy'))
        for image in subject_slices:
            im_arr = np.load(str(image))
            # im_max = np.max(im_arr)
            # im_min = np.min(im_arr)
            # im_arr = (im_arr - im_min) / (im_max - im_min)
            data_vector = [patient_id, image.stem[-3:]]
            data_vector += get_shape_feature_vector(im_arr[:, :, 0].squeeze())
            for c in range(len(channels)):
                im_tex_features = get_glcm_feature_vector(
                    im_arr[:, :, c],
                    sigma_normalization=True)
                data_vector += im_tex_features
            data_vector += [np.count_nonzero(im_arr)]
            data_vector += [class_label]
            dataset += [data_vector]

feature_names = ['dissimilarity', 'contrast', 'correlation', 'energy', 'homogeneity', 'ASM']
distances = ['1', '3', '5']
angles = ['0', '45', '90', '135']

tex_feature_names_compounds = product(channels, feature_names, distances, angles)

header = ['Patient_ID', 'Slice', 'MeshSurface', 'PixelSurface', 'Perimeter', 'PerimeterSurface',
          'Sphericity', 'SphericalDisproportion', 'MaximumDiameter', 'MajorAxis', 'MinorAxis', 'Elongation']

for combination in tex_feature_names_compounds:
    header += ['_'.join(combination)]
header += ['area']
header += ['class']

with open('kidney_shape_texture_features_123_NPYv2.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(dataset)