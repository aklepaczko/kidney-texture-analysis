# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 15:35:54 2022

@author: Artur Klepaczko
"""

from skimage.feature import graycomatrix, graycoprops
import numpy as np


feature_names = ['dissimilarity', 'contrast', 'correlation', 'energy', 'homogeneity', 'ASM']
distances = [1, 3, 5]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

num_levels = 64

def get_glcm_feature_vector(image_2d, sigma_normalization=True):
    mask = image_2d > 0
    if sigma_normalization:
        mu = np.mean(image_2d[mask])
        sigma = np.std(image_2d[mask])
        mask_3s_min = image_2d < mu - 3 * sigma
        mask_3s_max = image_2d > mu + 3 * sigma
        maxmin = 6 * sigma + 1
        image_2d -= (mu - 3 * sigma)
        image_2d[mask_3s_max] = maxmin
        image_2d[mask_3s_min] = 0
        image_2d *= 256 / maxmin
        
    image_2d = (image_2d / np.max(image_2d) * (num_levels - 1)).astype('uint8')
    image_2d[~mask] = num_levels
    glcm = graycomatrix(image_2d, distances, angles, levels=(num_levels+1),
                        symmetric=True)
    glcm = glcm[:num_levels, :num_levels]
    feature_vector = []
    for feature in feature_names:
        feature_vector.extend(graycoprops(glcm, feature).flatten())
    return feature_vector