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

#tf.enable_eager_execution()


class NFL_DATA_MODEL():
    def __init__(self):
        self.time = time
        self.project_path = 'E:/Project Stuff/Data Analysis Stuff/V5/NFL Data V1'
        self.Model_Data_Path = f'{self.project_path}/Model Data/Regression Model Data.csv'
        self.Save_Picks_Path = f'{self.project_path}/Prediction Data/Regression Prediction Data.csv'

        self.label_col = "EGO To Result Diff"
        self.Model_Drop_Cols = ["Spread Result Class"] 
        self.Class_Names = ["Upset", "Covered"]
        self.n_classes = 2 


    def Listify_Labels_2_Classes(self, Labels):
        New_labels = [[] for i in range(self.n_classes)]
        for label in range(0,len(Labels)):
            if Labels[label] == -1:
                New_labels[0].append(1)
                New_labels[1].append(0)
            elif Labels[label] == 1:
                New_labels[0].append(0)
                New_labels[1].append(1)
        new_label_df = pd.DataFrame()
        for class_type in range(0,self.n_classes):
            #print(self.Class_Names[class_type])
            new_label_df[self.Class_Names[class_type]] = New_labels[class_type]
        Labels = new_label_df.values
        # print(Labels)
        # time.sleep(10)
        return Labels

    def build_model(self):
        model = keras.Sequential([
            layers.Dense(self.num_columns, activation='relu', input_shape=[len(self.train_dataset.keys())]),
            # layers.Dense(self.num_columns, activation='relu'),
            layers.Dense(1)
        ])

        optimizer = tf.keras.optimizers.RMSprop(0.001)

        model.compile(loss='mse',
                        optimizer=optimizer,
                        metrics=['mae', 'mse'])
        return model

    def Train_Model(self):
        self.EPOCHS = 100
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        self.history = self.model.fit(
        self.normed_train_data, self.train_labels,
        epochs=self.EPOCHS, validation_split = 0.2, verbose=0)#,
        # callbacks=[early_stop])#,tfdocs.modeling.EpochDots()])

        self.Visualize_Training()

        loss, mae, mse = self.model.evaluate(self.normed_test_data, self.test_labels, verbose=2)
        print("Testing set Mean Abs Error: {:5.2f}".format(mae))

    def Visualize_Training(self):
        hist = pd.DataFrame(self.history.history)
        hist['epoch'] = self.history.epoch
        print(hist.head())
        print(hist.tail())

    def Test_Model(self):
        self.test_predictions = self.model.predict(self.normed_test_data).flatten()

        a = plt.axes(aspect='equal')
        plt.scatter(self.test_labels, self.test_predictions)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        lims = [-50, 50]
        plt.xlim(lims)
        plt.ylim(lims)
        _ = plt.plot(lims, lims)
        plt.show()

        error = self.test_predictions - self.test_labels
        plt.hist(error, bins = 25)
        plt.xlabel("Prediction Error")
        _ = plt.ylabel("Count")
        plt.show()

    def norm(self, x):
        return (x - self.train_stats['mean']) / self.train_stats['std']

    def Read_Data(self):
        self.dataset = pd.read_csv(self.Model_Data_Path)
        # self.dataset = self.dataset.drop('Spread Result Class', axis=1)
        # self.dataset = self.dataset[[c for c in self.dataset if c not in self.Drop_Cols] + ['Spread Result Class']]# ['Spread Result Class']]
        print(self.dataset.head())
        print(self.dataset.shape)
        print(self.dataset.tail())  
        # self.dataset = self.dataset.loc[~self.dataset['Spread to Result Difference'].isin(self.Drop_Spread_Results)]
        # print(self.dataset.head())
        # print(self.dataset.shape)
        # print(self.dataset.tail())  
    
    def Get_Features_Labels(self):
        self.column_names = list(self.train_dataset)
        # self.column_names = list(self.train_dataset)
        self.num_columns = len(self.column_names)
        print(self.num_columns)
        self.feature_names = self.column_names[:-1]
        self.label_name = self.column_names[-1]
        # self.label_name = "SRC"
        print(self.label_name)

    def Split_Data(self):
        #Split the data into testing and training
        self.train_dataset = self.dataset.sample(frac=0.8,random_state=0)
        self.test_dataset = self.dataset.drop(self.train_dataset.index)

    def Inspect_Data(self):
        sns.pairplot(self.train_dataset[[self.label_col, "DVOA EGO", "Opp Rushing Against Rating Correlation", "Opp Passing Against Rating"]], diag_kind="kde")
        # plt.show()
        self.train_stats = self.train_dataset.describe()
        self.train_stats.pop(self.label_col)
        self.train_stats = self.train_stats.transpose()
        print(self.train_stats)

    def Setup_Data(self):
        self.train_labels = self.train_dataset.pop(self.label_col)
        self.test_labels = self.test_dataset.pop(self.label_col)
        self.normed_train_data = self.norm(self.train_dataset)
        self.normed_test_data = self.norm(self.test_dataset)
        print(self.normed_train_data.head())

    def Print_Input_Data(self):
        print(self.train_labels)
        print(self.test_labels)
        print(self.normed_train_data)
        print(self.normed_test_data)
        self.time.sleep(5)
        print(len(self.train_labels))
        print(len(self.test_labels))

    # def Save_Picks(self):
    #     self.Save_Picks_DF['Labels'] = self.labels
    #     self.Save_Picks_DF['Prediction'] = self.predictions
    #     # self.Matchup_DF = self.Matchup_DF[[c for c in self.Matchup_DF if c in 'Spread Result Class']]
    #     print(self.Save_Picks_DF)
    #     self.Save_Picks_DF.to_csv(self.Save_Picks_Path, index=False)

    def Do_Stuff(self):
        self.Read_Data()
        self.Split_Data()
        self.Get_Features_Labels()
        self.Inspect_Data()
        self.Setup_Data()
        self.Print_Input_Data()

        self.model = self.build_model()
        self.model.summary()
        self.Train_Model()
        self.Test_Model()


Making_Model = NFL_DATA_MODEL()
# Making_Model.Setup_Data()
Making_Model.Do_Stuff()
# Making_Model.Initialize_Model()