from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import numpy as np
import random
import time
import os
import random

import pathlib
import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf
# import tensorflow_docs as tfdocs
# import tensorflow_docs.plots
# import tensorflow_docs.modeling

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#tf.enable_eager_execution()

project_path = 'E:/Project Stuff/Data Analysis Stuff/V5/NFL Data V1'
Model_Data_Path = f'{project_path}/Model Data/Regression WDVOA Model Data.csv'
label_col = "EGO To Result Diff"

dataset = pd.read_csv(Model_Data_Path)
dataset = dataset.drop('Spread Result Class', axis=1)

labels = dataset.pop(label_col)
column_names = list(dataset)
# self.column_names = list(self.train_dataset)
num_columns = len(column_names)
print(num_columns)


# create model
model = keras.Sequential()
model.add(layers.Dense(num_columns, input_dim=num_columns, kernel_initializer='normal', activation='relu'))
# self.model.add(layers.Dense(6, kernel_initializer='normal', activation='relu'))
model.add(layers.Dense(1, kernel_initializer='normal'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')


estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
train_data = dataset.values
train_labels = labels.values
print(train_data)
print(train_labels)
results = cross_val_score(pipeline, train_data, train_labels, cv=kfold)
print("Model: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# Evaluate_Model_2(model, train_dataset, train_labels)