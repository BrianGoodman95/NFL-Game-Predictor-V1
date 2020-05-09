import pandas as pd
import numpy as np
import random
import time
import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn import datasets
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn import preprocessing

from scipy.stats import pearsonr
import operator
import pickle

# from __future__ import absolute_import, division, print_function
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns


#Need model to predict score of game then manually subtract spread - Convert to 3 classes - upset, expected, covered (-3 --> +3 = close)
#Compare to 3 class results of actual 3 class spread to result
#Try having spread in or out of model when predicting score
#For real, will predict a game score margin, manually subtract psread, then decide how to bet

#tf.enable_eager_execution()


class NFL_DATA_MODEL():
    def __init__(self):
        #Data is in the form of team stats up to week #, opp stats up to week #, data about game (home team, day of week, bye week, spread) and result class of game - what I want to predict
        self.time = time
        self.project_path = 'E:/Project Stuff/Data Analysis Stuff/V5/NFL Data V1'
        # self.Normalized_Model_Data_Path = f'{self.project_path}/All Seasons Results_For_Model_Normalized.csv'
        self.Model_Data_Path = f'{self.project_path}/Matchup Data.csv'
        self.Save_Picks_Path = f'{self.project_path}/Matchup Prediction Data.csv'
        self.Class_Names = ["Upset", "Covered"]
        self.n_classes = 2 
        # self.Drop_Cols = ['Team', 'Opponent', 'Game Scoring Margin', 'Spread to Result Difference', 'Spread Result Class']
        self.Model_Col_Names = ["Week", "Spread", "SRD", "HT", "OPFPred", "OPFCor", "OPAPred", "OPACor", "OSAPred", "OSACor", "OSFPred", "OSFCor", "ORFPred", "ORFCor", "ORAPred", "ORACor"]#, "SRC"]
        self.Model_Drop_Cols = ["Team", "Year", "Opponent", "Game Scoring Margin"]
        # self.Drop_Spread_Results = [-2.5, -2, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]
        self.Drop_Spread_Results = [0]
        self.test_years = [2002, 2006, 2011, 2019]     
        self.train_years = [y for y in range(2001,2020) if y not in self.test_years]   

    # def Normalize_Data(self, dataset):
    #     min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    #     print(dataset)
    #     data = min_max_scaler.fit_transform(dataset)
    #     print(data)
    #     # data = dataset.values
    #     return data

    def Listify_Labels_2_Classes(self, Labels):
        # self.Training_Labels = dataset[[c for c in dataset if c in ['Spread Result Class']]]
        # Labels = dataset[[self.label_name]].values
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

    def Normalize_DF(self, df):
        df_data = df.values
        print(df_data)
        print(df_data.shape)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
        print(df_data)
        Normalized_Matchup_DF = pd.DataFrame(min_max_scaler.fit_transform(df_data), index=df.index, columns=self.Model_Col_Names)
        return Normalized_Matchup_DF
        
    def Make_Matchup_Model(self, df):
        for col_name in self.Model_Drop_Cols:
            df.drop(col_name, axis=1, inplace=True)
        self.col_names = list(df)
        print(len(self.col_names))
        print(self.Model_Col_Names)
        Normalized_Matchup_DF = self.Normalize_DF(df)
        #Lets also drop any rows that have SRD of <= 2 and then the column SRD and SRC
        # Normalized_Matchup_DF = Normalized_Matchup_DF.loc[~Normalized_Matchup_DF['SRD'].isin(self.Drop_Spread_Results)]
        Normalized_Matchup_DF = Normalized_Matchup_DF.drop('SRD', axis=1)
        # Normalized_Matchup_DF = Normalized_Matchup_DF.drop('SRC', axis=1)
        # self.MatchupDF.columns = self.Model_Col_Names
        # self.Normalized_Matchup_DF.to_csv(self.Matchup_Model_Data_Path, index=False)
        return Normalized_Matchup_DF

    def Read_Data(self):
        self.dataset = pd.read_csv(self.Model_Data_Path)
        # self.dataset = self.dataset[[c for c in self.dataset if c not in self.Drop_Cols] + ['Spread Result Class']]# ['Spread Result Class']]
        print(self.dataset.head())
        print(self.dataset.shape)
        print(self.dataset.tail())  
        self.dataset = self.dataset.loc[~self.dataset['Spread to Result Difference'].isin(self.Drop_Spread_Results)]
        print(self.dataset.head())
        print(self.dataset.shape)
        print(self.dataset.tail())  
    
    def Get_Features_Labels(self):
        self.column_names = list(self.dataset)
        # self.column_names = list(self.train_dataset)
        self.num_columns = len(self.column_names)
        print(self.num_columns)
        self.feature_names = self.column_names[:-1]
        self.label_name = self.column_names[-1]
        # self.label_name = "SRC"
        print(self.label_name)

    def Split_Data(self):
        #Split the data by year
        self.train_dataset = self.dataset.loc[(self.dataset['Year'].isin(self.train_years))]
        self.test_dataset = self.dataset.loc[(self.dataset['Year'].isin(self.test_years))]
        self.Save_Picks_DF = self.test_dataset #Save the dataset for later
        #Only train on data where SRD is bigger than 3
        # self.train_dataset = self.train_dataset.loc[~self.train_dataset['Spread to Result Difference'].isin(self.Drop_Spread_Results)]

        # #Split The dataset 80/20 to train/test
        # seed = random.randint(0,100)
        # self.train_dataset = self.dataset.sample(frac=0.8,random_state=seed)
        # self.test_dataset = self.dataset.drop(self.train_dataset.index)
        # #Shuffle the train and test data set to random order
        # self.test_dataset = self.test_dataset.sample(frac=1).reset_index(drop=True)
        # self.train_dataset = self.train_dataset.sample(frac=1).reset_index(drop=True)
        #Make 2D lists of the label columns
        self.label_name = "Spread Result Class"
        self.train_labels = self.train_dataset[self.label_name].values
        self.test_labels = self.test_dataset[self.label_name].values
        print(self.train_labels)
        print(self.test_labels)
        # print(self.trn_lbls)
        # time.sleep(10)
        #Make the feature datasets
        self.train_dataset = self.train_dataset[[c for c in self.train_dataset if c not in ['Spread Result Class']]]
        self.test_dataset = self.test_dataset[[c for c in self.test_dataset if c not in ['Spread Result Class']]]
        print(self.test_dataset)

    def Setup_Data(self):
        self.train_data = self.Make_Matchup_Model(self.train_dataset)
        self.train_data = self.train_data.values
        # self.train_data = self.Normalize_Data(self.train_dataset)
        self.test_data = self.Make_Matchup_Model(self.test_dataset)
        self.test_data = self.test_data.values
        # self.test_data = self.Normalize_Data(self.test_dataset)
        self.train_labels = self.Listify_Labels_2_Classes(self.train_labels)
        self.test_labels = self.Listify_Labels_2_Classes(self.test_labels)

        # self.normed_train_dataset = self.norm(self.train_dataset, train_stats)
        # self.normed_test_dataset = self.norm(self.test_dataset, test_stats)

    def Test_Correlations(self):
        headers = []
        label_header = ["Spread Result Class"]#['Game Scoring Margin']
        label_data = self.dataset[self.label_name].tolist()
        # print(label_data)
        self.num_features = len(self.feature_names)
        print(self.num_features)
        Correlation_Dict = {}
        Correlations = [[] for i in range(len(self.feature_names))]
        for name in range(0,len(self.feature_names)):
            feature_data = self.dataset[self.feature_names[name]].tolist()
            corr, p_value = pearsonr(label_data, feature_data)
            Correlation_Dict.update({self.feature_names[name]:abs(corr)})
            Correlations[0].append(self.feature_names[name])
            Correlations[1].append(abs(corr))
            print(name)
            headers.append(self.feature_names[name])
            if name % 4 == 0:
                print("graph")
                compare_headers = [self.label_name]+headers
                sns.pairplot(self.dataset[compare_headers], diag_kind="kde")
                headers = []
        sorted_x = sorted(Correlation_Dict.items(), key=operator.itemgetter(1))
        print(sorted_x)
        self.time.sleep(5)


    def Run_Model(self):

        n_nodes_hl1 = self.Num_Nodes
        n_nodes_hl2 = self.Num_Nodes
        n_nodes_hl3 = self.Num_Nodes

        self.x = tf.placeholder('float')
        self.y = tf.placeholder('float')
        print(f'Len of traindata[1] is: {len(self.train_data[1])}')
        # time.sleep(10)
        self.hidden_1_layer = {'f_fum':n_nodes_hl1,
                        'weight':tf.Variable(tf.random_normal([len(self.train_data[1]), n_nodes_hl1])),
                        'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

        self.hidden_2_layer = {'f_fum':n_nodes_hl2,
                        'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                        'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

        self.hidden_3_layer = {'f_fum':n_nodes_hl3,
                        'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                        'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}

        self.output_layer = {'f_fum':None,
                        'weight':tf.Variable(tf.random_normal([n_nodes_hl3, self.n_classes])),
                        'bias':tf.Variable(tf.random_normal([self.n_classes])),}

        self.train_neural_network_two()

    def neural_network_model_two(self):
        l1 = tf.add(tf.matmul(self.x,self.hidden_1_layer['weight']), self.hidden_1_layer['bias'])
        l1 = tf.nn.relu(l1)

        l2 = tf.add(tf.matmul(l1,self.hidden_2_layer['weight']), self.hidden_2_layer['bias'])
        l2 = tf.nn.relu(l2)

        l3 = tf.add(tf.matmul(l2,self.hidden_3_layer['weight']), self.hidden_3_layer['bias'])
        l3 = tf.nn.relu(l3)

        self.prediction = tf.matmul(l3,self.output_layer['weight']) + self.output_layer['bias']


    def train_neural_network_two(self):
        self.neural_network_model_two() 
        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.y))
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.prediction, labels=self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            
            for epoch in range(self.hm_epochs):
                epoch_loss = 0
                i=0
                while i < len(self.train_data):
                    start = i
                    end = i+self.batch_size
                    batch_x = np.array(self.train_data[start:end])
                    batch_y = np.array(self.train_labels[start:end])

                    _, c = sess.run([optimizer, cost], feed_dict={self.x: batch_x,
                                                                self.y: batch_y})
                    epoch_loss += c
                    i+=self.batch_size
                    
                print('Epoch', epoch+1, 'completed out of',self.hm_epochs,'loss:',epoch_loss)
                self.loss = epoch_loss
                #sess = tf.InteractiveSession()
            
            self.Evaluate_2_Classes()

    def Get_Probabilities(self):         
        # self.label_probabilities = self.y.eval({self.x:self.train_data, self.y:self.train_labels})
        # self.prediction_probabilities = self.prediction.eval({self.x:self.train_data, self.y:self.train_labels})
        self.label_probabilities = self.y.eval({self.x:self.test_data, self.y:self.test_labels})
        self.prediction_probabilities = self.prediction.eval({self.x:self.test_data, self.y:self.test_labels})
        print(self.label_probabilities)
        print(self.prediction_probabilities)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        self.normalized_predictions = min_max_scaler.fit_transform(self.prediction_probabilities)
        print(self.normalized_predictions)

    def Filter_Predictions(self):
        self.Sectioned_Games = [[] for i in range(len(self.probability_sections))]
        self.Filtered_Predictions = [[] for i in range(len(self.probability_sections))]
        self.Filtered_Labels = [[] for i in range(len(self.probability_sections))]
        for gm in range(0,len(self.normalized_predictions)):
            probabilitys = [self.normalized_predictions[gm][0], self.normalized_predictions[gm][1]]
            favorite_probability = max(probabilitys)
            underdog_probability = min(probabilitys)
            for section in range(0,len(self.probability_sections)):
                # if (favorite_probability >= self.probability_sections[section]: #Split by favorite %
                if (favorite_probability-underdog_probability) > self.probability_sections[section]: #Split by favorite-underdog difference %
                    self.Sectioned_Games[section].append(gm)
                    break
        print(self.normalized_predictions[1])
        print(self.normalized_predictions[1][0])
       
    def Evaluate_Accuracy(self):
        Pred_Results = [0 for i in range(4)]
        print(len(self.predictions))
        print(len(self.labels))
        print("My Evaluation")
        # self.time.sleep(10)
        covers = 0
        thought_covers = 0
        thought_not_covers = 0
        not_covers = 0
        for gm in range(0,len(self.predictions)):
            if self.predictions[gm] == 1 : #Think Favorite Covers
                thought_covers += 1
                if self.labels[gm] == 1: #Did cover
                    Pred_Results[0] += 1
                elif self.labels[gm] == 0: #Didn't cover
                    Pred_Results[1] += 1
            elif self.predictions[gm] == 0: #Thought Favorite Didn't cover
                thought_not_covers += 1
                if self.labels[gm] == 0: #Didn't cover
                    Pred_Results[2] += 1
                elif self.labels[gm] == 1: #Covered
                    Pred_Results[3] += 1
        
        for i in self.labels:
            if i == 1:
                covers += 1
            elif i == 0:
                not_covers += 1
        print(covers)
        print(thought_covers)
        print(not_covers)
        print(thought_not_covers)
        try:
            print(f'Thought Covered & Did Cover is: {100*Pred_Results[0]/thought_covers}%')
            print(f'Thought Covered & Didnt Cover is: {100*Pred_Results[1]/thought_covers}%')
            print(f'Thought Not-Covered & Not-Covered is: {100*Pred_Results[2]/thought_not_covers}%')
            print(f'Thought Not-Covered & Covered is: {100*Pred_Results[3]/thought_not_covers}%')
        except:
            pass
        try:
            Totally_Correct = (Pred_Results[0] + Pred_Results[2])/(thought_covers + thought_not_covers)
            # Half_of_Neutrals_Accuracy = ((Pred_Results[1] + Pred_Results[7])/(thought_covers + thought_not_covers))/2 #Would get half of the close games probably
            self.Total_Betting_Accuracy = Totally_Correct #+ Half_of_Neutrals_Accuracy
        except:
            self.Total_Betting_Accuracy = 0
        print(f'Betting Accuracy is {self.Total_Betting_Accuracy}%')

    def Evaluate_2_Classes(self):
        # test_spreads = self.test_dataset.pop('Spread').values.tolist()
        self.labels = tf.argmax(self.y, 1).eval({self.x:self.test_data, self.y:self.test_labels})
        self.predictions = tf.argmax(self.prediction,1).eval({self.x:self.test_data, self.y:self.test_labels})
        # # self.labels = tf.argmax(self.y, 1).eval({self.x:self.train_data, self.y:self.train_labels})
        # # self.predictions = tf.argmax(self.prediction,1).eval({self.x:self.train_data, self.y:self.train_labels})
        # print(self.labels)
        # print(self.predictions)
        # time.sleep(10)
        # self.Evaluate_Accuracy()
        self.Save_Picks()
        self.Get_Probabilities()
        self.Filter_Predictions()
        for game in range(0, len(self.labels)):
            for section in range(0,len(self.Sectioned_Games)):
                for gm in self.Sectioned_Games[section]:
                    if gm == game:
                        self.Filtered_Predictions[section].append(self.predictions[game])
                        self.Filtered_Labels[section].append(self.labels[game])
        print('90%')
        # print(self.Filtered_Predictions[0])            
        # print(self.Filtered_Labels[0])
        # print("60%")
        # print(self.Filtered_Predictions[3])
        # print(self.Filtered_Labels[3])

        for section in range(0,len(self.probability_sections)):
            self.predictions = self.Filtered_Predictions[section]
            self.labels = self.Filtered_Labels[section]
            print(len(self.predictions))
            print(len(self.labels))
            self.Evaluate_Accuracy()
            self.Model_Type_All_Accuracys[section].append(self.Total_Betting_Accuracy)
            self.Model_Type_All_Counts[section].append(len(self.predictions))
            self.Model_Type_All_Losses[section].append(self.loss)

        # for section in range(0,len(self.probability_sections)):
        #     print(self.All_Accuracys[section])

    def Save_Picks(self):
        self.Save_Picks_DF['Labels'] = self.labels
        self.Save_Picks_DF['Prediction'] = self.predictions
        # self.Matchup_DF = self.Matchup_DF[[c for c in self.Matchup_DF if c in 'Spread Result Class']]
        print(self.Save_Picks_DF)
        self.Save_Picks_DF.to_csv(self.Save_Picks_Path, index=False)

    def Do_Stuff(self):
        self.Read_Data()
        self.Split_Data()
        self.Setup_Data()
        # self.Get_Features_Labels()
        # self.Test_Correlations()

        print(self.train_labels)
        print(self.test_labels)
        print(self.train_data)
        print(self.test_data)
        self.time.sleep(5)
        print(len(self.train_labels))
        print(len(self.test_labels))
        print(len(self.train_data[1]))
        print(len(self.test_data))
        # time.sleep(10)
        self.batch_size = 8
        self.hm_epochs = 30
        self.Num_Nodes = 500
        self.learning_rate = 0.0001
        # accs=[]
        # for i in range(25):
        #     # with tf.Session() as self.sess:
        #     #     self.sess.run(tf.initialize_all_variables())
        #     self.Run_Model()
        #     accs.append(self.Total_Betting_Accuracy)
        #     self.Save_Picks()
        # print(accs)
        # betting_results_DF = pd.DataFrame()
        # betting_results_DF["Betting accuracy"] = accs
        # betting_results_DF.to_csv('Model_Type_Results.csv')

        self.probability_sections = [0.4, 0.2, 0] #Sections for favorite % splitting
        # self.probability_sections = [0.35, 0.2, 0.01, 0] #Sections for favorite-underdog % splitting
        results_df = pd.DataFrame()
        results_df[' '] = [f'Accuracy for > {self.probability_sections[sec]*100}% Confidence' for sec in range(0,len(self.probability_sections))] + ['Total']
        results_df.set_index(' ')

        Node_Sizes = [20, 50, 100, 300]
        Learning_Rates = [0.00001, 0.0001, 0.001]
        Batch_Sizes = [16, 8, 4, 1]
        # self.hm_epochs = 30
        Epochs = [200, 50, 30, 10]
        for size in range(0,len(Batch_Sizes)):
            self.batch_size = Batch_Sizes[size]
            for epoch in range(0,len(Epochs)):
                self.hm_epochs = Epochs[epoch]
                for node_num in range(0,len(Node_Sizes)):
                    self.Num_Nodes = Node_Sizes[node_num]
                    for learn_rate in range(0,len(Learning_Rates)):
                        self.learning_rate = Learning_Rates[learn_rate]
                        self.Model_Type_All_Accuracys = [[] for i in range(len(self.probability_sections))]
                        self.Model_Type_All_Counts = [[] for i in range(len(self.probability_sections))]
                        self.Model_Type_All_Losses = [[] for i in range(len(self.probability_sections))]
                        for i in range(5):
                            self.Run_Model()
                            # self.Evaluate_2_Classes()  
                        Total_Accuracys = [] 
                        Just_Accuracys = []
                        for section in range(len(self.probability_sections)):
                            avg_acc = sum(self.Model_Type_All_Accuracys[section])/len(self.Model_Type_All_Accuracys[section])
                            avg_count = sum(self.Model_Type_All_Counts[section])/len(self.Model_Type_All_Counts[section])
                            print(f'Accuracy for favorite probability > {self.probability_sections[section]} is: {avg_acc}')
                            Total_Accuracys.append(f'{avg_acc}, {avg_count}')
                            Just_Accuracys.append(avg_acc)
                        Total_Accuracys.append(f'{sum(Just_Accuracys)/len(Just_Accuracys)}, {self.Model_Type_All_Losses[0]}')
                        results_df[f'BatchSize: {Batch_Sizes[size]}, Epochs: {Epochs[epoch]}, NumberNodes: {Node_Sizes[node_num]}, LearningRate: {Learning_Rates[learn_rate]}'] = Total_Accuracys
                        transposed_df = results_df.transpose()
                        print(transposed_df)
                        print(transposed_df.columns)
                        # transposed_df.rename(columns=lambda x: f'{int(x)*100}% Betting Accuracy', inplace=True)

                        # transposed_df.rename(columns = {f'{self.probability_sections[sec]}': f'{self.probability_sections[sec]*100}% Betting Accuracy'} for sec in range(0,len(self.probability_sections)), inplace = True)
                        # transposed_df.columns = [f'{self.probability_sections[sec]}%' for sec in self.probability_sections]
                        transposed_df.to_csv('Model_Type_Results.csv', header=False)

Making_Model = NFL_DATA_MODEL()
# Making_Model.Setup_Data()
Making_Model.Do_Stuff()
# Making_Model.Initialize_Model()