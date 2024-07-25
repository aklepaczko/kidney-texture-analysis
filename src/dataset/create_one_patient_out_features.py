# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 22:31:38 2023

@author: Artur Klepaczko
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.read_csv('kidney_texture_features_123_NPYv2-pruned.csv', sep=',')

root = Path('OnePatientOut_features')

patients = df['Patient_ID'].unique()

# selected = ['Patient_ID', 'W_contrast_5_0', 'opp_contrast_1_90', 'opp_dissimilarity_1_90',
#        'W_correlation_5_45', 'W_correlation_5_90', 'opp_correlation_1_90',
#        'W_contrast_5_135', 'in_correlation_5_135', 'opp_dissimilarity_1_45',
#        'in_contrast_5_135','class']

selected = ['Patient_ID', 'W_correlation_5_90', 'in_correlation_3_90', 'W_contrast_5_90',
       'in_correlation_5_90', 'W_dissimilarity_5_90', 'W_correlation_3_90',
       'in_correlation_1_90', 'opp_contrast_1_90', 'opp_correlation_1_90',
       'opp_dissimilarity_1_90', 'class']

df = df[selected]

accuracies = []

for patient_id in patients:
    df_patient_train = df[df['Patient_ID']!=patient_id]
    df_patient = df[df['Patient_ID']==patient_id]
    out_dir = root / patient_id
    out_dir.mkdir(exist_ok=True, parents=True)
    out_train = out_dir / 'train.csv'
    out_test = out_dir / 'test.csv'
    
    df_patient.drop(['Patient_ID'], axis=1, inplace=True)
    df_patient.to_csv(out_test, sep=',', index=False)
    df_patient_train.drop(['Patient_ID'], axis=1, inplace=True)
    df_patient_train.to_csv(out_train, sep=',', index=False)

    scaler = StandardScaler()
    
    X_train = df_patient_train.to_numpy()[:, :-1]
    y_train = df_patient_train.to_numpy()[:, -1]
    
    X_test = df_patient.to_numpy()[:, :-1]
    y_test = df_patient.to_numpy()[:, -1]
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    predictor = SVC(kernel='rbf', gamma=0.05, C=1, degree=5, coef0=5.0)
    predictor.fit(X_train_scaled, y_train)
    
    acc = predictor.score(X_test_scaled, y_test)
    accuracies += [(patient_id, np.unique(y_test), acc)]

_, _, scores = zip(*accuracies)
avg_score = np.mean(scores)
accuracies += [('Average', avg_score)]

header = ['Patient_ID', 'SVM accuracy']
with open('SVM_texture_features_123_NPYv2-pruned.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(accuracies)