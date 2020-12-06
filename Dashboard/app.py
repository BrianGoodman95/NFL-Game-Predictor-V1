# -*- coding: utf-8 -*-
# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import threading
import time

import frontend
from parsers.setup import Directory_setup, Dashboard_setup
setup = Directory_setup.Create_Directories()
project_path = setup.project_path

season, week = Dashboard_setup.This_Week()
Data = Dashboard_setup.Data(project_path, season)

colours = Dashboard_setup.colours

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets) 

# app.layout = html.Div(style={'backgroundColor': colours['background']}, children=[
app.layout = html.Div([
    html.Div([
        html.H1('NFL Betting Predictor',
            style={
                'textAlign': 'center',
                'color': colours['title']
            }
        ),
        html.H4('A Data Based Model to Pick Games and Win $$$', 
            style={
            'textAlign': 'center',
            'color': colours['font']
            }
        ),
    ]),
    html.Div([
        html.Div([
            html.H4(f'Week {week} Betting Guide',
                style={
                'textAlign': 'left',
                'color': colours['font'],
                'margin-left': 65,
                }
            ),
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in Data[0].columns],
                data=Data[0].to_dict("rows"),
                style_table={
                    'maxHeight': '75ex',
                    'overflowY': 'scroll',
                    'width': '75%',
                    'minWidth': '75%',
                    "margin-left": 50,
                },
            ),
        ], className="six columns"),
        html.Div([
            html.H3('2020 Betting Results',
                style={
                    'textAlign': 'center',
                    'color': colours['graph_text']
                }
            ),
            dcc.Dropdown(id='season-results',
                options=[{'label': s, 'value': s}
                        for s in list(Data[1])[:-4]],
                value=list(Data[1])[:2],#+Data[1][-1],
                multi=True
                ),
            html.Div(children=html.Div(id='Season_Results')),
            dcc.Interval(
                id='column_update',
                interval=100),
            ], className="six columns"),#,style={'width':'50%','margin-left':10,'margin-right':10,'max-width':50000})
    ], className="row"),  
    html.Div([
        html.Div([
            html.H3('Historical Data',
                style={
                    'textAlign': 'center',
                    'color': colours['graph_text']
                }
            ),
            dcc.Dropdown(id='historical-results',
                options=[{'label': s, 'value': s}
                        for s in list(Data[2])[:-3]],
                # placeholder="Select a metric to analyze the models betting accuracy...",
                value=list(Data[2])[0],
                ),
            html.Div(children=html.Div(id='Historical_Results')),
            dcc.Interval(
                id='data-update',
                interval=100),
            ], className="twelve columns",style={'width':'80%', 'height':'100ex', 'margin-left':175,'margin-right':175,'max-width':50000})
    ], className="row")        
])

@app.callback(
    dash.dependencies.Output('Season_Results','children'),
    [dash.dependencies.Input('season-results', 'value')],
    events=[dash.dependencies.State('column-update', 'interval')]
    )
def season_data(data_names):
    Data = Dashboard_setup.Data(project_path, season)
    Results = []
    # for res, data in enumerate(Data):
    Make_Figure = frontend.Plots()
    fig = Make_Figure.Season_Results(Data[1],data_names)
    # Prediction_Results = frontend.Season_Results(Data[1], season)
    # fig = Prediction_Results.fig
    Results.append(html.Div(dcc.Graph(id='season-data', figure=fig)))

    return Results


@app.callback(
    dash.dependencies.Output('Historical_Results','children'),
    [dash.dependencies.Input('historical-results', 'value')],
    events=[dash.dependencies.State('data-update', 'interval')]
    )
def historical_data(data_names):
    Data = Dashboard_setup.Data(project_path, season)
    Results = []
    # for res, data in enumerate(Data):
    Make_Figure = frontend.Plots()
    fig = Make_Figure.Historical_Results(Data[2], data_names)
    Results.append(html.Div(dcc.Graph(id='historical-data', figure=fig)))

    return Results

# app.css.append_css({
#     'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
# })

if __name__ == '__main__':
    app.run_server(debug=True)
