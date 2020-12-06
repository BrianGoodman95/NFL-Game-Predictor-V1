import time
import pandas as pd
from parsers import Game_Predictor
from parsers import Prediction_Analysis
from parsers.setup import Directory_setup, Dashboard_setup
setup = Directory_setup.Create_Directories()
project_path = setup.project_path

season, week = Dashboard_setup.This_Week()

last_run_time = time.time()-55
while True:
    if time.time() - last_run_time > 60:
        # Run the Weekly Predictor
        last_run_time = time.time()
        Data = Game_Predictor.NFL_Game_Predictor(project_path, week, season, updateType='Week')
        Spread_Target_DF = Data.Spread_Targets
        # print(f'Week {week} Evaluation:')
        picks = Data.picks
        # print(picks)
        # print(Spread_Target_DF)

        #Analyze Season Results
        Results = Prediction_Analysis.Prediction_Analyzer(project_path, season)
        Prediction_Stats = Results.Analyzed_Results
        # print(Prediction_Stats)

        #Add this Week to the Historical Database
        database = f'{project_path}/All Game Data.csv' #Path To Master Data
        database_df = pd.read_csv(database) #Read the Data
        archived_database = database_df.loc[database_df['Season'] != season] #Exclude this season
        this_season = pd.read_csv(f'{project_path}/raw data/{season}/Season Game Data.csv') #Read this seasons updated data
        this_season = this_season.loc[this_season['Week'] != week] #Exclude this week
        combined_df = pd.concat([archived_database, this_season]) #Combine old and this season data
        combined_df.to_csv(database, index=False) #Save
