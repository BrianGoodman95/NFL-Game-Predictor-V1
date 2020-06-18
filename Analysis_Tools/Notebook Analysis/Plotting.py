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

    layout = go.Layout(title=go.layout.Title(text=plt_title, xref="paper", x=0.5), showlegend=True, xaxis=dict(title = x_axis_label, autorange='reversed', color="#000", gridcolor="rgb(232, 232, 232)"), yaxis = dict(title = y_axis_label, color="#000", gridcolor="rgb(232, 232, 232)"),  plot_bgcolor='rgba(0,0,0,0)')
    fig = go.Figure(data=data, layout=layout)
    fig.show(renderer='notebook')
