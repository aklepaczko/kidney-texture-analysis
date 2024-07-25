# -*- coding: utf-8 -*-
"""
Created on Tue Feb 6 19:49:54 2024

@author: Artur Klepaczko
"""
import numpy as np
from radiomics import shape2D
import SimpleITK as sitk


def get_shape_feature_vector(image_2d: np.ndarray):
    image = sitk.GetImageFromArray(image_2d)
    mask = image_2d > 0
    mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    extractor = shape2D.RadiomicsShape2D(image, mask)
    return [extractor.getMeshSurfaceFeatureValue(),
            extractor.getPixelSurfaceFeatureValue(),
            extractor.getPerimeterFeatureValue(),
            extractor.getPerimeterSurfaceRatioFeatureValue(),
            extractor.getSphericityFeatureValue(),
            extractor.getSphericalDisproportionFeatureValue(),
            extractor.getMaximumDiameterFeatureValue(),
            extractor.getMajorAxisLengthFeatureValue(),
            extractor.getMinorAxisLengthFeatureValue(),
            extractor.getElongationFeatureValue()
            ]
