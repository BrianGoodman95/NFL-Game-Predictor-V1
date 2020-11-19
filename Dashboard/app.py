# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import os

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash()#__name__, external_stylesheets=external_stylesheets)

project_path = os.getcwd().split('Dashboard')[0]
data_path = f'{project_path}/Data/Raw Data/DVOA_Based'
model_path = f'{project_path}/Data/Total Data/Models'
data = pd.read_csv(f'{data_path}/All Game Data.csv')
model = pd.read_csv(f'{model_path}/All Seasons Scores Grouped By WDVOA Diff.csv')

colors = {
    'background': 'rgba(0,0,0,0)',
    'text': '#7FDBFF'
}

Locs = ['Away', 'Home']
Leg = ['Avg Game Margin', 'EGO lobf']
Figs = []
for loc in Locs:
    dvoa_ranges = list(model[f'{loc} DVOA Diff Range'])
    avgs = []
    for r in dvoa_ranges:
        vals = r.split(' to ')
        avg = (float(vals[0])+float(vals[1]))/2
        avgs.append(avg)
    model[f'{loc} DVOA Diff'] = avgs
    avg_spreads = [f'Avg Spread: {i}' for i in list(model[f'{loc} Avg Spread'])]
    
    fig = (px.scatter(model, x=f'{loc} DVOA Diff', y=f'{loc} Avg Game Margin').update_traces(mode='markers', marker=dict(size=3),showlegend = True))
    fig.add_scatter(x=model[f'{loc} DVOA Diff'], y=model[f'{loc} EGO'], mode='lines', hovertext= avg_spreads, hoverinfo="text",)
    fig.data[0].name = 'Avg Score Margin'
    fig.data[1].name = 'EGO lobf'
    fig.update_layout(dict(plot_bgcolor=colors['background'], paper_bgcolor=colors['background']))#, height = 500, width=1000))
    Figs.append(fig)
    # fig.show()

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Hello Dash',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.Div(children='Dash: A web application framework for Python.', style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    html.Div([
        html.Div([
            html.H3(children='Away Team Model', style={
                'textAlign': 'center',
                'color': colors['text']
            }),
            dcc.Graph(id='away-team-model', figure=Figs[0])
        ], className='six columns'),
        html.Div([
            html.H3(children='Home Team Model', style={
                'textAlign': 'center',
                'color': colors['text']
            }),            
            dcc.Graph(id='home-team-model', figure=Figs[1])
        ], className='six columns')
    ])
])
app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})
if __name__ == '__main__':
    app.run_server(debug=True)


# app.layout = html.Div(children=[
#     html.Div(children='''
#         Symbol to graph:
#     '''),
#     dcc.Input(id='input', value='', type='text'),
#     html.Div(id='output-graph'),
# ])

# @app.callback(
#     Output(component_id='output-graph', component_property='children'),
#     [Input(component_id='input', component_property='value')]
# )
# def update_value(input_data):
#     start = datetime.datetime(2015, 1, 1)
#     end = datetime.datetime.now()
#     df = web.DataReader(input_data, 'morningstar', start, end)
#     df.reset_index(inplace=True)
#     df.set_index("Date", inplace=True)
#     df = df.drop("Symbol", axis=1)

#     return dcc.Graph(
#         id='example-graph',
#         figure={
#             'data': [
#                 {'x': df.index, 'y': df.Close, 'type': 'line', 'name': input_data},
#             ],
#             'layout': {
#                 'title': input_data
#             }
#         }
#     )
