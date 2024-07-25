# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 19:33:49 2022

@author: Artur Klepaczko
"""

import nibabel as nii
import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Final


NII_ROOT: Final[Path] = Path(R'D:\TexturesKidneys\NIFTI-T2')

OUT_ROOT: Final[Path] = Path(R'D:\TexturesKidneys\NPY-T2')

EDGE: Final[int] = 64

use_roi = True

labeled_files = list(NII_ROOT.glob('**/*label.nii.gz'))
for patient_mask in tqdm(labeled_files, total=len(labeled_files)):
    roi_nii = nii.load(patient_mask)
    roi_data = roi_nii.get_fdata()
    arg_mask = np.argwhere(roi_data)
    z = np.unique(arg_mask[:, -1])

    assert len(z)==1, print('Error in mask file processing.')
    
    z = z.item()

    patient_name = patient_mask.parent.stem

    out_dir = OUT_ROOT / patient_name
    out_dir.mkdir(parents=True, exist_ok=True)
            
    roi = roi_data[:, :, z]
    roi_x, roi_y = np.nonzero(roi)
    x_min = np.min(roi_x)
    x_max = np.max(roi_x)
    y_min = np.min(roi_y)
    y_max = np.max(roi_y)

    x_center = x_min + (x_max - x_min) // 2
    y_center = y_min + (y_max - y_min) // 2
            
    x_min = np.max([x_center - EDGE // 2, 0])
    y_min = np.max([y_center - EDGE // 2, 0])
    
    roi = roi[x_min:x_min + EDGE, y_min:y_min + EDGE]
    roi = np.rot90(roi)
    roi = np.fliplr(roi)

    out_arr = np.zeros((EDGE, EDGE, 2))
    
    image_filename = str(patient_mask).replace('-label', '')
    image = nii.load(image_filename)
    image_arr = image.get_fdata()

    slice_im = image_arr[x_min:x_min + EDGE, y_min:y_min + EDGE, z]
    slice_im = np.rot90(slice_im)
    slice_im = np.fliplr(slice_im)

    data_max = np.max(slice_im)
    if use_roi:
        slice_im = slice_im * roi

    slice_im = (slice_im / data_max * 255).astype(np.uint8)

    out_arr[:, :, 0] = slice_im
    out_arr[:, :, 1] = roi

    file_path = out_dir / f'{patient_name}_slice.png'
    cv2.imwrite(str(file_path), slice_im, (cv2.IMWRITE_PNG_COMPRESSION, 0))

    np.save(out_dir / f'{patient_name}_mask.npy', roi)
    np.save(out_dir / f'{patient_name}_slice_mask.npy', out_arr)

