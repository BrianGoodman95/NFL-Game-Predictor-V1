import pandas as pd
import time
import os
import requests
from parsers import Prediction_Helper
# from Analysis_Tools.Parsers import Latest_Data_Processor
# try:
#     from Parsers.Current import Prediction_Helper
#     from importlib import reload
#     reload(Prediction_Helper)
# except:
#     from Analysis_Tools.Parsers.Current import Prediction_Helper

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
    def __init__(self, project_path, current_week=16, current_season=2020, updateType='Season', Enable_Messaging=False):
        self.time = time
        self.Read_Previous_Data = True
        self.Enable_Messaging = Enable_Messaging
        
        #Dictionary of raw stats to collect
        self.Collect_Data_Dict = {
            'Team':[], 'Opponent':[], 'Week':[], 'Year':[], 'Home Team':[], 'WDVOA':[], 'Spread':[], 'Result':[]
        }
        #Dictionary of data to calculate
        self.Calculate_Data_Dict = {
            'EGO':[], 'EGO_Spread_Diff':[], 'EGO_Result_Diff': [], 'Correct':[]
        }
        #Collection parameters
        min_season=current_season
        self.min_week = 6
        self.current_week = current_week
        print(updateType)
        if updateType == 'Historical':
            Update_Weeks = [w for w in range(self.min_week,16)]
            min_season = 2006
        elif updateType == 'Season':
            Update_Weeks = [w for w in range(self.min_week,self.current_week+1)]
        elif updateType == 'Week':
            Update_Weeks = [current_week-1, current_week]
        else:
            print('INVALID VALUE IN FIELD UPDATETYPE')
            return
        
        for self.season in range(min_season,current_season+1):
            if self.season <= 2019: #Get weeks to go through for Past seasons
                self.season_weeks = [w for w in range(self.min_week,16)]
            else: #Weeks to go through for this season
                self.season_weeks = [w for w in range(self.min_week,self.current_week+1)]
                if updateType == 'Historical':
                    Update_Weeks = self.season_weeks
            #Lists and DFs
            self.Week_DF = pd.DataFrame()
            self.All_Weeks_DFs = []
            
            #Make save locations
            self.project_path = project_path
            self.raw_data_path = f'{project_path}/raw data/{self.season}'
            self.Make_Folder(self.raw_data_path)

            #Update any Data Needed
            for week in Update_Weeks:
                print(f'Analyzing {self.season}, Week: {week} Games ...')
                #Get the Raw Data we need
                self.Raw_Game_Data = self.Get_Game_Info(self.raw_data_path, week)       
                self.Processed_Game_Data = self.Process_Game_Info(self.raw_data_path, week, self.Raw_Game_Data)       
                self.Calculated_Game_Data = self.Calculate_Game_Info(self.raw_data_path, week, self.Processed_Game_Data)
                self.Spread_Targets = self.Picking_Info(self.raw_data_path, week, self.Calculated_Game_Data)
            #Save Final Copy for further analysis
            week_dfs = []
            for week in self.season_weeks:
                df = pd.read_csv(f'{self.raw_data_path}/Week {week}/Calculated Game Data.csv')
                week_dfs.append(df)
            df = pd.concat(week_dfs) #Concat the list of dfs into a season df
            df.to_csv(f'{self.raw_data_path}/Season Game Data.csv', index=False)
        #Save Final Copy for further analysis
        season_dfs = []
        for season in range(min_season,self.season+1):
            season_df = pd.read_csv(f'{project_path}/raw data/{season}/Season Game Data.csv')
            season_dfs.append(season_df)
        season_df = pd.concat(season_dfs) #Concat the list of dfs into a season df
        season_df.to_csv(f'{project_path}/raw data/All Game Data.csv', index=False)

    def Make_Folder(self, new_path):
        data_exists = False
        try:
            os.mkdir(new_path)
        except:
            # print('folder already exists')
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

    def User_Message(self, message, sleep=1):
        if self.Enable_Messaging:
            print(message)
            time.sleep(sleep)

    def Picking_Info(self, raw_data_path, week, df):
        #Save the df passed in for picking after
        new_picks_df = df
        self.User_Message('Determining Spread Targets For Each Game ...')        
        #Output the target spread range for each game
        #sort by date so output is better
        df = df.sort_values(by=['Day', 'Time'], ascending=[0,1])  
        #Only keep the Away Teams' rows  
        df = df[df.Home_Team == 0]
        predictionDF = pd.DataFrame()
        Teams = list(df['Team'])
        Opponents = list(df['Opponent'])
        Games = [f'{Teams[x]} @ {Opponents[x]}' for x in range(len(Teams))]
        predictionDF['Game'] = Games
        target_spreads = list(df['Target Spreads'])
        predictions = []
        spreads = list(df['Betting Spread'])
        EGOs = list(df['EGO'])
        formated_EGOs = []
        formated_spreads = []
        for game in range(len(spreads)):
            spread = spreads[game]
            EGO = EGOs[game]
            #Always write wrt the favorite
            if float(spread) < 0:
                formated_spreads.append(f'{Teams[game]} at {spread}')
            elif float(spread) > 0:
                # s = spread.split('-')[-1]
                formated_spreads.append(f'{Opponents[game]} at -{spread}')   
            else:
                formated_spreads.append('Pick em Game')     
            #Always write wrt the favorite    
            if float(EGO) >= 0:
                formated_EGOs.append(f'{Teams[game]} wins by {EGO}')
            elif float(EGO) < 0:
                ego = str(EGO).split('-')[-1]
                formated_EGOs.append(f'{Opponents[game]} wins by {ego}')
            targets = target_spreads[game]
            closest_target = 100 #some big number
            predictions.append('') #Add a placeholder for now
            if "Missing" in targets or "Close" in targets: #Not a real target so skip
                predictions[game] = targets
            else:
                for target_range in targets:
                    for t in target_range: #Check each spread value there
                        target_to_spread_diff = abs(float(t)-float(spread))
                        if target_to_spread_diff < closest_target: #If closer to actual spread than last one, replace the target range!
                            closest_target = target_to_spread_diff
                            if EGO+float(spread) > 0: # 4+(-2)
                                t = target_range[1]
                                if float(spread) > 0:
                                    s_parts = str(t).split('-')
                                    str_spread = f'+{s_parts[-1]}'
                                else:
                                    str_spread = t
                                predictions[game] = f'{Teams[game]} (at {str_spread} or better)'
                            else:
                                t = target_range[0]
                                if float(spread) < 0:
                                    s_parts = str(t).split('-')
                                    str_spread = f'+{s_parts[-1]}'
                                else:
                                    str_spread = f'-{t}'
                                predictions[game] = f'{Opponents[game]} (at {str_spread} or better)'
                            # predictions[game] = target_range
        for pos, s in enumerate(spreads):
           spreads[pos]=float(s)
        predictionDF['Spread'] = formated_spreads 
        predictionDF['EGO'] = formated_EGOs
        predictionDF['Spread Target'] = predictions
        predictionDF['Pick'] =  list(df['Pick'])
        # print(predictionDF)
        self.Save_DF(predictionDF, f'{raw_data_path}/Week {week}/Spread Targets.csv')

        self.User_Message('Making Picks ...')
        time.sleep(0.5)
        #Get the Picks for the week
        self.picks = list(set(list(df['Pick'])))
        self.picks.remove("")
        # print(self.picks)
        time.sleep(1)
        #Add the data for picks made for this week
        All_Picks = []
        try:
            pick_df = pd.read_csv(f'{raw_data_path}/Week {week}/Picks.csv')
            prev_picks = list(pick_df['Pick'])
        except:
            pick_df = new_picks_df
            prev_picks = list(new_picks_df['Pick'])
        All_Picks.append(pick_df)
        # new_picks_df = new_picks_df.loc[new_picks_df['Team'].isin(self.picks)] #Keep data for games we're picking only
        for team in self.picks: #For each team we've picked
            if team in prev_picks: #Check if our picks already have that team
                pass
            else:
                new_pick_df = new_picks_df.loc[new_picks_df['Team'] == team] #Save this pick
                All_Picks.append(new_pick_df)
        All_Picks_DF = pd.concat(All_Picks)
        self.Save_DF(All_Picks_DF, f'{raw_data_path}/Week {week}/Weekly Picks.csv')

        #Later, if week not current one, evaluate if Pick Right or Wrong        
        #Even more later, make fancier visual of current week predictions
        #Even more later, make analysis showing season stats of picking
        return predictionDF

    def Calculate_Game_Info(self, raw_data_path, week, df):
        #Get the Map
        model_path = self.project_path.split('/data')[0] + '/models'
        self.Map_DF = pd.read_csv(f'{model_path}/All Seasons Scores Grouped By WDVOA Diff.csv')
        #Setup the Helper
        EGO_Analyzer = Prediction_Helper.EGO_Prediction(self.project_path, self.Map_DF)
        #Get the Map/Model
        self.Map = EGO_Analyzer.Map
        #Calculate what we need
        self.Calculated_Data = EGO_Analyzer.Calculate_Data(df)
        #Add to the DataFrame
        for key, val in self.Calculated_Data.items():
            df[key] = val
        self.Save_DF(df, f'{raw_data_path}/Week {week}/Calculated Game Data.csv')
        human_df = df.copy()

        #Make Human Readable Version
        self.User_Message(f'Summarizing Game Data For Week {week} ...')
        Teams = list(df['Team'])
        Opponents = list(df['Opponent'])
        Games = [f'{Teams[x]} @ {Opponents[x]}' for x in range(len(Teams))]
        human_df['Team'] = Games
        human_df = human_df[human_df.Home_Team == 0]
        human_df = human_df.rename(columns={"Team": "Game"})
        del human_df['Opponent']
        del human_df['Home_Team']
        # print(human_df)
        time.sleep(1)
        self.Save_DF(human_df, f'{raw_data_path}/Week {week}/Final Game Data.csv')
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
            try:
                srds.append(margins[game] + float(list(df['Betting Spread'])[game]))
            except ValueError: #A Pk Spread
                srds.append(margins[game])
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
        self.User_Message(f'Retrieving WDVOA Data for week {week} ...')
        try: #IF CAN READ PREVIOUSLY SAVED DVOA DATA
            WDVOA_DF = pd.read_csv(f'{week_path}/DVOA Data.csv')
            cols = list(WDVOA_DF)
            new_cols = [x if x != 'Weighted DVOA' else 'WDVOA' for x in cols] #Rename the WDVOA column
            WDVOA_DF.columns = new_cols
            WDVOA_DF = WDVOA_DF.sort_values(by='TEAM', ascending=True)
            # self.Save_DF(WDVOA_DF, f'{week_path}/DVOA Data.csv') 
        except: #IF NEED TO GET NEW DVOA DATA
            print(f'NEED TO PLACE DOWNLOADED DATA INTO THE WEEK {week} FOLDER')
            # game_info_collector = Prediction_Helper.Game_Info_Parser(week, self.season, raw_data_path)
            # WDVOA_DF = game_info_collector.WDVOA_DF
            # print(WDVOA_DF)
            # self.Save_DF(WDVOA_DF, f'{week_path}/DVOA Data.csv') 

        #Schedule+Spread DATA
        self.User_Message(f'Retrieving Scheudle and Scores for week {week} ...')
        #Get the schedule for the week
        game_info_collector = Prediction_Helper.Game_Info_Parser(week, self.season, raw_data_path)
        Week_DF = game_info_collector.Week_Sched_DF
        # print(Week_DF)
        #Get the spreads for the week
        self.User_Message(f'Retrieving Spreads for week {week} ...')
        spread_collector = Prediction_Helper.Spread_Parser(week, self.current_week, raw_data_path)
        Spread_DF = spread_collector.parser_df
        # print(Spread_DF)
        # Update Names of Teams to match the WDVOA team names
        self.User_Message(f'Combining Retrieved Data ...')
        raw_dfs = [WDVOA_DF, Week_DF, Spread_DF]
        team_matcher = Prediction_Helper.Team_Matching(raw_data_path, raw_dfs)
        Combined_Raw_DF = team_matcher.Combined_Raw_DF

        Game_Info_DF = Combined_Raw_DF
        self.Save_DF(Game_Info_DF, f'{week_path}/Raw Game Data.csv')
        # print(Game_Info_DF)
        return Game_Info_DF

# NFL_DATA = NFL_Game_Predictor(latest_week)
