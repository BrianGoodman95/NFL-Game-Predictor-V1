import pandas as pd
import numpy as np
import random
import time
import os
from scipy.stats import pearsonr
import plotly
import plotly.graph_objs as go
from itertools import combinations 
from sklearn.preprocessing import MinMaxScaler

#For each team of each season get all games up to that week (only week 5 and later)
#For each of the Big 6 columns, get correlation factor between spread/result difference and the advantage
#For the top column, interpolate the spread result based on the column adv and then turn into covered/not
#Check predictions from above with actual result


class PRREDICTION_DATA_EVALUATOR():
    def __init__(self, project_path, DVOA_Type):
        #Data is in the form of team stats up to week #, opp stats up to week #, data about game (home team, day of week, bye week, spread) and result class of game - what I want to predict
        #Multi class classification model needed to predict big upset, upset, expected, blowout, big blowout
        # self.DVOA_Type = input(f'Enter The DVOA Type to use (DVOA or WDVOA): ')
        # self.DVOA_Type = "WDVOA"
        self.DVOA_Type = DVOA_Type
        if self.DVOA_Type != "WDVOA" and self.DVOA_Type != "DVOA":
            print("BAD DVOA TYPE ENTERED")
        self.time = time
        self.project_path = project_path
        self.data_path = self.project_path.replace("Notebooks", "Data")
        self.Total_Prediction_Data_Path = f'{self.data_path}/Total Data/Total {self.DVOA_Type} Prediction Data.csv'
        self.Moving_Avg_Plot_Path = f'{self.project_path}/Evaluations/'
        self.Make_Folder(self.Moving_Avg_Plot_Path)
        self.moving_windows = [250, 500]
        self.Prediction_Cols = ["DVOA Pick Right", "Matchup Adj Pick Rigt"]
        self.EGODiff_Cols = ["DVOA EGO to Spread Diff", "Matchup Adj EGO to Spread Diff"]
        self.legend = []


    def Make_Folder(self, new_path):
        path_exists = False
        try:
            os.mkdir(new_path)
        except:
            print('folder exists')
            path_exists = True
        return path_exists

    def plotly_graph(self, x_data, y_data, save_path, file_name, legend, x_axis_name, y_axis_name):
        # plt_title = ' '.join([file_name, y_axis_name, "vs", x_axis_name])
        plt_title = file_name
        plt_save_name = f'{save_path}/{plt_title}.html'
        # print(len(x_data))
        # print(len(y_data))
        data= []
        line_modes = ['markers', 'lines+markers', 'lines+markers']
        # hovertexts= [self.Hover_Data[:-1], ["lobf" for i in range(len(x_data[1]))], [self.Hover_Data[-1]]]
        for i in range(0,len(legend)):
            # print(i)
            # print(legend[i])
            trace = go.Scatter(
            x=x_data[i],
            y=y_data[i],
            mode = 'lines+markers',
            # mode = line_modes[i],
            name = legend[i],
            # hovertext=hovertexts[i],
            )
            data.append(trace)
        x_axis_label = f'{x_axis_name}'
        y_axis_label = f'{y_axis_name}'
        # plotly.offline.plot({
        # "data": data,
        # "layout": go.Layout(title=go.layout.Title(text=plt_title, xref="paper", x=0.5), showlegend=True, xaxis=dict(title = x_axis_label, autorange='reversed', color="#000", gridcolor="rgb(232, 232, 232)"), yaxis = dict(title = y_axis_label, color="#000", gridcolor="rgb(232, 232, 232)"),  plot_bgcolor='rgba(0,0,0,0)')
        # }, filename = plt_save_name, auto_open=False)
        layout = go.Layout(title=go.layout.Title(text=plt_title, xref="paper", x=0.5), showlegend=True, xaxis=dict(title = x_axis_label, autorange='reversed', color="#000", gridcolor="rgb(232, 232, 232)"), yaxis = dict(title = y_axis_label, color="#000", gridcolor="rgb(232, 232, 232)"),  plot_bgcolor='rgba(0,0,0,0)')
        fig = go.Figure(data=data, layout=layout)
        fig.show(renderer='notebook')

    def Do_Graphing(self):
        # #Make Plotting Hover Data and save path
        # self.Hover_Data = []
        # for game in range(0, len(self.Opps)):
        #     self.Hover_Data.append(f'{self.Opps[game]}-Week{self.Weeks[game]}')
        #Plot
        self.plotly_graph(self.Evaluation_EGODiffs, self.Evaluation_Accs, self.Moving_Avg_Plot_Path, f'{self.DVOA_Type} Moving Average Evaluation', self.legend, self.EGODiff_Cols[0], 'Betting Accuracy')
    
    def Read_Prediction_Data(self):
        self.dataset = pd.read_csv(self.Total_Prediction_Data_Path)
        self.Rows = len(self.dataset.index)
        self.Cols = list(self.dataset)
        print(self.dataset.head())
        print(self.dataset.tail())
        print(self.Rows)
        print(self.Cols)
        # self.time.sleep(5)
    
    def Get_Data_Cols(self):
        self.Predictions = [[] for i in range(len(self.Prediction_Cols))]
        self.EGO_Data = [[] for i in range(len(self.EGODiff_Cols))]
        self.Evaluation_EGODiffs = [[] for i in range(len(self.Prediction_Cols)+len(self.moving_windows))]
        self.Evaluation_Accs = [[] for i in range(len(self.Prediction_Cols)+len(self.moving_windows))]
        for eval_type in range(0, len(self.Prediction_Cols)): #For each of teh DVOA and Matchup Predictions
            #Need to sort dataset by the ego_diff collumn
            sorted_dataset = self.dataset.sort_values(by=f'{self.EGODiff_Cols[eval_type]}', ascending=False)
            self.Predictions[eval_type] = sorted_dataset[f'{self.Prediction_Cols[eval_type]}'].tolist()
            self.EGO_Data[eval_type] = sorted_dataset[f'{self.EGODiff_Cols[eval_type]}'].tolist()
            print(self.EGO_Data[eval_type][0:20])
            self.time.sleep(2)

    def Take_Moving_Avg(self):
        for col in range(0,len(self.Prediction_Cols)):
            self.legend.append(f'{self.Prediction_Cols[col]} - {self.moving_windows[self.avg_num]} points moving average')
            # moded_predictions = [((x*50)+50) for x in self.Predictions[col]] 
            for dp in range(0, len(self.Predictions[col])-self.moving_window):
                avg_egoDiff = sum(self.EGO_Data[col][dp:(dp+self.moving_window)])/self.moving_window
                # avg_acc = sum(moded_predictions[dp:(dp+self.moving_window)])/self.moving_window
                avg_acc = ((sum(self.Predictions[col][dp:(dp+self.moving_window)])/2)+(self.moving_window/2))/self.moving_window
                self.Evaluation_EGODiffs[col+(2*self.avg_num)].append(avg_egoDiff)
                self.Evaluation_Accs[col+(2*self.avg_num)].append(avg_acc)

    
    def Do_Stuff(self):
        #Want to read the df
        #for each prediction col, sort df by the EGODiff col abs value
        #for each datapoint, calculate next 500 points avg EGO_Diff and Accuracy then store the avg ego diff and accuracy in a list and repeat until reach the last value
        #Then plot and save in folder
        self.Read_Prediction_Data()
        self.Get_Data_Cols()
        for self.avg_num in range(0, len(self.moving_windows)):
            self.moving_window = self.moving_windows[self.avg_num]
            self.Take_Moving_Avg()
        self.Do_Graphing()

# min_week = 6
# min_season = 2006
# MatchupCollector = NFL_DATA_MODEL('DVOA')
# MatchupCollector.Analyze()