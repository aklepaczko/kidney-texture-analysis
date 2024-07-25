# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 21:27:14 2023

@author: Artur Klepaczko
"""

import pandas as pd
from skfeature.function.similarity_based import fisher_score

df = pd.read_csv('t2_kidney_texture_features.csv', sep=',')

data = df.to_numpy()

X = data[:, 1:-1]
y = data[:, -1]

score = fisher_score.fisher_score(X, y)

idx = fisher_score.feature_ranking(score)

print(score.max())
print(list(zip(df.columns[idx[:10]+2], score[idx[:10]])))