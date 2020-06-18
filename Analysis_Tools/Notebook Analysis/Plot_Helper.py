import numpy as np
import time
import plotly
import plotly.graph_objs as go
import plotly.express as px


class Plotting():
    def __init__(self, data):
        self.data = data

    # def Standard_Data_Plot(self):
    #     '''
    #     Here we plot the passed dataframe in a plotly express simple figure format
    #     '''
    #     self.plot_data = self.data
    #     self.PX_Quick_Graph()

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


    def PX_Quick_Scatter(self, df, ydata_name, xdata_name, legend_ID):
        plt_title = f'{ydata_name} vs {xdata_name}'
        fig = (px.scatter(df, x=xdata_name, y=ydata_name, color=legend_ID).update_traces(mode='markers', marker=dict(size=3),))
        fig.update_layout(dict(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', title_text = plt_title))
        fig.update_xaxes(dict(title = f'{xdata_name}', color="#000", gridcolor="rgb(232, 232, 232)", ticks="inside"))
        fig.update_yaxes(dict(title = f'{ydata_name}', color="#000", gridcolor="rgb(232, 232, 232)", ticks="inside"))
        fig.show(renderer='notebook')

        return fig