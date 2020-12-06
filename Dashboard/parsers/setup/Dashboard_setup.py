import datetime
import pandas as pd

def This_Week():
    currentDate = datetime.date.today().isoformat()
    year = currentDate.split('-')[0]
    week_num = datetime.date(int(year), int(currentDate.split('-')[1]), int(currentDate.split('-')[2])).isocalendar()[1]
    day_num = datetime.date(int(year), int(currentDate.split('-')[1]), int(currentDate.split('-')[2])).isocalendar()[2]
    if day_num <= 2: #Monday or Tuesday
        week_num = week_num - 1 #Still reference last week 
    first_week = 36
    w = week_num - first_week
    if w <= 0:
        w = 52-first_week + w
    return int(year), int(w)

def Data(path, season):
    season, week = This_Week()
    data_path = f'{path}'

    #Week Data
    Spread_Targets = pd.read_csv(f'{data_path}/raw data/{season}/Week {week}/Spread Targets.csv')
    Spread_Targets = Spread_Targets.drop('Pick', 1)
    Spread_Targets.columns = ['Game', 'Spread', 'EGO', 'Pick']

    #Season Data
    Season_Results = pd.read_csv(f'{data_path}/{season} Betting Results.csv')
    #ReOrder the columns so important ones are first
    new_cols = [[] for i in range(6)]
    for c in list(Season_Results):
        if "Accuracy" in c:
            if 'EGO' in c:
                new_cols[1].append(c)
            else:
                new_cols[0].append(c)
        elif "Pick" in c:
            new_cols[2].append(c)
        elif "EGO" in c:
            new_cols[3].append(c)
        elif "Games" in c:
            new_cols[4].append(c)
        else:
            new_cols[5].append(c)
    cols = []
    for col in range(len(new_cols)):
        cols+=new_cols[col]
    Season_Results = Season_Results[cols]

    #Historical Data
    Historical_Data = pd.read_csv(f'{data_path}/All Game Data.csv')
    cols = list(Historical_Data)
    Keep_Cols = ['Season', 'Week', 'Betting Spread', 'WDVOA Delta',	'EGO', 'Spread to EGO Diff', 'Make Pick', 'Pick', 'Pick Right']
    for col in cols:
        if col not in Keep_Cols:
            Historical_Data = Historical_Data.drop(col, 1)
    Historical_Data.columns = Keep_Cols

    #Put it All Together
    Data = [Spread_Targets, Season_Results, Historical_Data]
    return Data

colours = {
    'background': 'rgba(0,0,0,0)',
    'title': '#7FDBFF',
    'font': 'rgba(55,55,55,55)',
    'graph_text': 'rgba(55,55,55,55)',
    'axis_text': 'rgba(55,55,55,55)'
}


# class layout():
#     def __init__(self):
#         self.colors = {
#             'background': 'rgba(0,0,0,0)',
#             'title': '#7FDBFF',
#             'font': 'rgba(255,255,255,255)',
#             'graph_text': 'rgba(255,255,255,255)',
#             'axis_text': 'rgba(255,255,255,255)'
#         }