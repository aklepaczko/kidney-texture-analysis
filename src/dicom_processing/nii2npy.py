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


NII_ROOT: Final[Path] = Path(R'.')
ROI_ROOT: Final[Path] = Path(R'D:\TexturesKidneys')
NPY_ROOT: Final[Path] = Path(R'D:\TexturesKidneys\NPY-PNG-112-ROI-v3')

NATIVES: Final[Path] = ROI_ROOT / Path('Natywne_ROI')
NIFTI: Final[Path] = NII_ROOT / Path('NiFTI')
OUT_ROOT: Final[Path] = NPY_ROOT
EDGE: Final[int] = 112

file_name = 'X t1_vibe_dixon_cor_Y_converted.nii.gz'

INFO_ROOT = Path(R'C:\Users\Artur Klepaczko\Documents\Science\TexturesKidneys')

with open(INFO_ROOT / 'patients_info_123_20240108.json', 'r') as f:
    patients_info = json.load(f)

filters = {'category': 1,
           'both_sides_ids': ['PT6', 'PT30', 'PK2', 'PK4', 'PK5', 'PK6', 'PK8', 'PK10'],
           'modes': ('opp', 'in',)}

mode_inds = {'opp': 2,
             'in': 1,
             'F': 3,
             'W': 4}

TYPE_DIR = OUT_ROOT / str(filters['category'])

dataset = []

use_roi = True

for patient in tqdm(patients_info):
    patient_files = patients_info[patient]
    class_label = patient_files['class']
    patient_id = int(patient[2:])
    if class_label != filters['category']:
        continue
    if 'right' not in patient_files:
        continue
    sides = ['right']
    if 'left' in patient_files and patient in filters['both_sides_ids']:
        sides += ['left']
    for side in sides:
        out_dir = TYPE_DIR / patient
        out_dir.mkdir(parents=True, exist_ok=True)

        roi_path = NATIVES / patient / patient_files[side][0]
        roi_image = nii.load(roi_path)
        roi_data = roi_image.get_fdata()
    
        z_range = patient_files[f'slices_{side[0]}']
        z_min = z_range[0]
        z_max = z_range[1]
    
        for z in range(z_min, z_max+1):
            
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

            out_arr = np.zeros((EDGE, EDGE, 3))
            
            for ch, mode in enumerate(filters['modes']):
                mode_id = mode_inds[mode]
                file_id = patient_files[side][mode_id]
                image_filename = file_name.replace('X', str(file_id))
                image_filename = image_filename.replace('Y', mode)
                image_path = NIFTI / patient / image_filename
                image = nii.load(image_path)
                image_arr = image.get_fdata()
    
                data_max = np.max(image_arr)

                slice_im = image_arr[x_min:x_min + EDGE, y_min:y_min + EDGE, z]
                slice_im = np.rot90(slice_im)
                slice_im = np.fliplr(slice_im)

                if use_roi:
                    slice_im = slice_im * roi

                out_arr[:, :, ch] = slice_im
            
                slice_im = (slice_im / data_max * 255).astype(np.uint8)
                file_path = out_dir / f'{patient}_{file_id}_{side}_{mode}_{z:03d}.jpg'
                cv2.imwrite(str(file_path), slice_im, (cv2.IMWRITE_JPEG_QUALITY, 100))
        
            out_arr[:, :, 2] = roi
            np.save(out_dir / f'{patient}_{file_id}_{side}_{z:03d}.npy', out_arr)
            file_path = out_dir / f'{patient}_{file_id}_{side}_OppInMask_{z:03d}.png'
            # cv2.imwrite(str(file_path), out_arr, (cv2.IMWRITE_JPEG_QUALITY, 100))
            cv2.imwrite(str(file_path), out_arr, (cv2.IMWRITE_PNG_COMPRESSION, 0))

