import pandas as pd
import time
import os
import requests
# from Analysis_Tools.Parsers import Latest_Data_Processor
from Analysis_Tools.Parsers import Prediction_Helper

#Testing
'''
Goal is to collect all the data from the collection dictionary for the current week, save it, then calculate the data for the calculation dictionary, save it then append it to the total from previous weeks for a total season results
Each Week starting from week 5, will collect the data, save it, calculate the other data, save it and append to any previous saved data and save it as total
Weeks = [i for in range(min week, current_week+1)]
For week in weeks:
    COLLECTION DATA
    **From gamecollection.thisweekparsing / thisweekstats
    Get schedule
    Get This week teams and opponents
    Get Home/Away, Short Week, Bye Week
    If week < current week
        Get Results (points for winner - points for loser)
    **From gamecollection.betting_parsing:
    Get Spreads
    **from dvoa_collector.?
    Get WDVOA stats

    Save Data for the week

    CALCULATION DATA
    **From dvoacollector.setupmap
    Get Map
    Interpolate EGO
    Make Prediction
    Calculate EGO/Spread 
    If week < current week
        Calculate EGO/Result Diff
        Evaluate if prediction correct

    Save Data for the week
    If week > min week
        Read previous Season Data 
        Append new data to read data
    Save total data for season
'''

class NFL_Game_Predictor():
    def __init__(self, project_path, current_week=16, current_season=2020):
        self.time = time
        self.Read_Previous_Data = True
        
        #Dictionary of raw stats to collect
        self.Collect_Data_Dict = {
            'Team':[], 'Opponent':[], 'Week':[], 'Year':[], 'Home Team':[], 'WDVOA':[], 'Spread':[], 'Result':[]
        }
        #Dictionary of data to calculate
        self.Calculate_Data_Dict = {
            'EGO':[], 'EGO_Spread_Diff':[], 'EGO_Result_Diff': [], 'Correct':[]
        }
        #Collection parameters
        self.min_week = 5
        self.current_week = current_week
        self.All_Weeks = [w for w in range(self.min_week,self.current_week+1)]
        self.season = current_season
        #Lists and DFs
        self.Week_DF = pd.DataFrame()
        self.All_Weeks_DFs = []
        
        #Save names
        self.final_csvName = f'{self.season}/{self.season} Betting Results'

        #Make save locations
        self.project_path = project_path
        self.raw_data_path = f'{project_path}/Raw Data/{self.season}'
        self.output_data_path = f'{project_path}/Prediction Data/{self.season}'
        self.Make_Folder(self.raw_data_path)
        self.Make_Folder(self.output_data_path)

    ### HELPER FUNCTIONS
    def Make_Folder(self, new_path):
        data_exists = False
        try:
            os.mkdir(new_path)
        except:
            print('folder already exists')
            files = os.listdir(new_path)
            if len(files) > 1:
                data_exists = True
        return data_exists
    
    def Save_DF(self, df, path_name):
        df.to_csv(path_name, index=False)

    def Concat_and_Save(self, df_list, path):
        df = pd.concat(df_list) #Concat the list of dfs into a season df
        df.to_csv(path, index=False)
        return df
       
    def Setup_Map(self):
        #Need to take all the DVOA_Diffs and map them to an EGO via the total home or away map
        #Read the Map DF
        self.Map_DF = pd.read_csv(f'{self.project_path}/Total Data/Models/All Seasons Scores Grouped By WDVOA Diff.csv')
        # self.Map_DF = pd.read_csv(f'{self.Training_DVOA_Stat_save_path}/{self.Training_DVOA_Stat_save_name}')
        self.Maps = [{} for i in range(len(self.Game_Locations))]
        for loc in range(0, len(self.Game_Locations)):
            #Make List of the 2 columns needed
            self.DF_EGOs = self.Map_DF[f'{self.Game_Locations[loc]} EGO'].tolist()
            self.DF_Diffs = self.Map_DF[f'{self.Game_Locations[loc]} DVOA Diff Range'].tolist()
            for diff in range(0, len(self.DF_Diffs)):
                LL = float(self.DF_Diffs[diff].split("to ")[0])
                UL = float(self.DF_Diffs[diff].split("to ")[1])
                if LL == -100:
                    avg = -75
                elif UL == 100:
                    avg = 75
                else:
                    avg = round(((LL + UL)/2),2)
                self.DF_Diffs[diff] = avg
                self.DF_EGOs[diff] = round(float(self.DF_EGOs[diff]),2)
                # self.Map
            print(self.DF_EGOs)
            print(self.DF_Diffs)
            self.Maps[loc] = {self.DF_Diffs[i]:self.DF_EGOs[i] for i in range(len(self.DF_EGOs))}
            print(self.Maps)

    def Predict_Game_Info(self, raw_data_path, week, df):
        #Read in Map as a DF (setup above)
        #For each game,
            #Determine if team is home
            #Get Team WDVOA
            #Look up Opponent --> Opp WDVOA
            #Get WDVOA Diff
            #Get Lookup where row from map WDVOA Diff falls and get ego
            #        EGO = round(np.interp(DVOA_Diff, list(self.Maps[Loc].keys()), list(self.Maps[Loc].values())),2)
        
        #Loc = get_location
        #DVOA Diff = get_teams wdvoa - opp wdvoa
        EGO = round(np.interp(DVOA_Diff, list(self.Maps[Loc].keys()), list(self.Maps[Loc].values())),2)
        Spread = float(self.Spreads[game])
        Score = float(self.Game_Margins[game])
        Spr_EGO_Diff = EGO+Spread
        Scr_EGO_Diff = Score-EGO
        Scr_Spr_Diff = Score+Spread
        #Add EGO to the df
        #Get EGO to Spr Diff & add to df
        #Get EGO Margin Diff & add to df
        #Get Pick (sign of EGO-Spr Diff)

        #Later, if week not current one, evaluate if Pick Right or Wrong
        #Later, highlight picks where EGO-Spr Diff in the target range OR for each game, highlight spread where would be in target 
        
        #Even more later, make fancier visual of current week predictions
        #Even more later, make analysis showing season stats of picking
        return df

    def Process_Game_Info(self, raw_data_path, week, df):
        #Add Season to the Raw DF
        df['Season'] = [f'{self.season}' for i in range(len(list(df['Team'])))]
        #Get Vicotry Margin from the PtsW and ptsL cols
        margins = []
        for game in range(0,len(list(df['Points For']))): #For each game in the df
            try:
                margins.append(int(list(df['Points For'])[game]) - int(list(df['Points Against'])[game]))
            except ValueError: #Nans since games not played yet
                margins.append(0)
        df['Scoring Margin'] = margins
        #Make Spread to Result Diff Column
        srds = []
        for game in range(0,len(margins)):
            srds.append(margins[game] + float(list(df['Betting Spread'])[game]))
        df['SRD'] = srds
        #Delete Date, Unnamed: 7, PtsW, PtsL, YdsW, TOW, YdsL, TOL
        dropCols = ['Date', 'Unnamed: 7', 'Points For', 'Points Against', 'YdsW', 'TOW', 'YdsL', 'TOL']
        for col in dropCols:
            df.drop(f'{col}', axis=1, inplace=True)        
        #Re-Order Columns - Put Season column to begining
        cols = list(df)
        cols = cols[-3:-2] + cols[:-3] + cols[-2:]
        df = df[cols]

        self.Save_DF(df, f'{raw_data_path}/Week {week}/Processed Game Data.csv')
        return df

    def Get_Game_Info(self, raw_data_path, week):
        #Make the folder for data
        week_path = f'{raw_data_path}/Week {week}'
        self.Make_Folder(week_path)
        #WDVOA DATA
        try: #IF CAN READ PREVIOUSLY SAVED DVOA DATA
            WDVOA_DF = pd.read_csv(f'{week_path}/DVOA Data.csv')
        except: #IF NEED TO GET NEW DVOA DATA
            game_info_collector = Prediction_Helper.Game_Info_Parser(week, self.season)
            WDVOA_DF = game_info_collector.WDVOA_DF
            print(WDVOA_DF)
            self.Save_DF(WDVOA_DF, f'{week_path}/DVOA Data.csv') 

        #Schedule+Spread DATA
        #Get the schedule for the week
        game_info_collector = Prediction_Helper.Game_Info_Parser(week, self.season)
        Week_DF = game_info_collector.Week_Sched_DF
        print(Week_DF)
        #Get the spreads for the week
        spread_collector = Prediction_Helper.Spread_Parser(week, raw_data_path)
        Spread_DF = spread_collector.parser_df
        print(Spread_DF)
        # Update Names of Teams to match the WDVOA team names
        raw_dfs = [WDVOA_DF, Week_DF, Spread_DF]
        team_matcher = Prediction_Helper.Team_Matching(raw_data_path, raw_dfs)
        Combined_Raw_DF = team_matcher.Combined_Raw_DF

        Game_Info_DF = Combined_Raw_DF
        self.Save_DF(Game_Info_DF, f'{week_path}/Raw Game Data.csv')

        return Game_Info_DF


    ### ORGANIZERS
    def Do_Stuff(self):
        for week in self.All_Weeks:
            print(f'Week: {week}')
            #Get the Raw Data we need
            self.Raw_Game_Data = self.Get_Game_Info(self.raw_data_path, week)       
            self.Processed_Game_Data = self.Process_Game_Info(self.raw_data_path, week, self.Raw_Game_Data)       
            self.Predict_Game_Data = self.Predict_Game_Info(self.raw_data_path, week, self.Processed_Game_Data)
        # project_excel_name = f'{self.data_path}/{self.final_csv_name}.csv'
        # self.Concat_and_Save(self.All_Seasons_DF_List, project_excel_name)

# NFL_DATA = NFL_Game_Predictor(latest_week)
# NFL_DATA.Do_Stuff()