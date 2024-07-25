# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 23:06:03 2023

@author: Artur Klepaczko
"""
from pathlib import Path
import random

from tqdm import tqdm

npy_root = Path(R'D:\TexturesKidneys\NPY_v2')
exp_root = Path(R'D:\TexturesKidneys\OnePatientOut')

patients = []

for category_dir in npy_root.iterdir():
    for patient in category_dir.iterdir():
        patients += [patient]

patient_ids = [patient.name[2:] for patient in patients]

for patient in tqdm(patients):
    train_dir = exp_root / patient.name
    train_dir.mkdir(exist_ok=True, parents=True)
    
    train_file_name = train_dir / f'{patient.name}_train.txt'
    valid_file_name = train_dir / f'{patient.name}_valid.txt'        
    test_file_name = train_dir / f'{patient.name}_test.txt'
    
    with open(test_file_name, 'w') as test_fp:
        test_fp.write(patient.name[2:])

    patient_ids_current = patient_ids[:]
    patient_ids_current.remove(patient.name[2:])
    
    patient_ids_current = [f'{x}\n' for x in patient_ids_current]
    
    valid_ids = random.sample(patient_ids_current, 12)
    train_ids = patient_ids_current[:]
    for valid in valid_ids:
        train_ids.remove(valid)
    
    with open(train_file_name, 'w') as train_fp:
        train_fp.writelines(train_ids)
    
    with open(valid_file_name, 'w') as valid_fp:
        valid_fp.writelines(valid_ids)
    
    