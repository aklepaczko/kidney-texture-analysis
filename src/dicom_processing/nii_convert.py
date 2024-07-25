# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 20:57:03 2022

@author: Artur Klepaczko
"""

import nibabel as nii
import numpy as np
from pathlib import Path


root = Path(r'D:\TexturesKidneys\NiFTI-T2')

# patient = 'PK11'

# patient_path = root / patient

for nii_path in root.glob('**/*.nii.gz'):
    in_image = nii.load(nii_path)
    header_info = in_image.header
    
    header_info['quatern_b'] = 0
    header_info['quatern_c'] = 0
    header_info['quatern_d'] = 0
    header_info['qoffset_x'] = 0
    header_info['qoffset_y'] = 0
    header_info['qoffset_z'] = 0
    
    pixel_spacing = in_image.header['pixdim']
    affine = np.eye(4)
    np.fill_diagonal(affine, pixel_spacing[1:5])
    nii_image = nii.Nifti1Image(np.flip(in_image.get_fdata()), affine, header=header_info)
    nii.save(nii_image, str(nii_path).replace('.nii.gz', '_converted.nii.gz'))
    