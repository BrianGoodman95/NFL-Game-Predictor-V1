import pandas as pd
import time
import os
import requests

'''
Goal is to collect all the data from the collection dictionary for the current week, save it, then calculate the data for the calculation dictionary, save it then append it to the total from previous weeks for a total season results
Each Week starting from week 5, will collect the data, save it, calculate the other data, save it and append to any previous saved data and save it as total
Weeks = [i for in range(min week, current_week+1)]
For week in weeks:
    COLLECT DATA
        Want to get the saved Calculated Data csv
        
    CALCULATE DATA
        Want to get the count of predictions made, # correct, number wrong, # games total, # EGOs correct and wrong
        Want to put the data in same format so can make this season version ofthe accuracy vs ego/spread diff plot

    Save Data for the week
        Save the data for the season we find as a DF where each row is a week, each stat is a column

'''

class Prediction_Analyzer():
    def __init__(self, project_path, current_season=2020, Enable_Messaging=False):
        self.time = time
        self.Read_Previous_Data = True
        self.Enable_Messaging = Enable_Messaging
        
        #Dictionary of raw stats to collect
        self.Result_Stats = {
            'Season':[], 'Week':[], 'Games':[], 'EGO Correct':[], 'EGO Wrong':[], 'EGO Accuracy':[], 'Picked Games':[], 'Picks Correct':[], 'Picks Wrong':[], 'Pick Accuracy':[], 'Season EGO Accuracy':[], 'Season Picks Accuracy':[]
        }

        #Collection parameters
        self.season = current_season
        self.min_week = 7
        
        #Make save locations
        self.project_path = project_path
        self.raw_data_path = f'{project_path}'
        self.Make_Folder(self.raw_data_path)
        self.final_csvName = f'{self.raw_data_path}/{self.season} Betting Results.csv'

        #See if need to analyze the week or already done
        self.Analyzed_Results, min_week = self.Check_Results_Exist(self.final_csvName)
        #Update any Data Needed
        for week in range(min_week,16):
            # print(f'Analyzing Week: {week} Picks ...')
            try:
                #Get the Raw Data we need    
                self.Calculated_Game_Data = pd.read_csv(f'{self.raw_data_path}/raw data/{current_season}/Week {week}/Weekly Picks.csv')
                self.Analyze_Data(self.Calculated_Game_Data, self.raw_data_path, week)
            except FileNotFoundError:
                # print('ERROR')
                break
        #Drop last row since is incomplete Data
        self.Analyzed_Results = self.Analyzed_Results[:-1]

        #Save Final Copy for further analysis
        self.Analyzed_Results.to_csv(self.final_csvName)

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

    def User_Message(self, message, sleep=1):
        if self.Enable_Messaging:
            print(message)
            time.sleep(sleep)
    
    def Check_Results_Exist(self, f):
        # try:
        #     df = pd.read_csv(f)
        #     last_week = list(df['Week'])[-1]
        # except:
        df=pd.DataFrame(columns=self.Result_Stats.keys())
        last_week = self.min_week
        return df, last_week

    def Evaluate_Picks(self, df):
        #EGO correct if EGO/Spread Diff same sign as SRD
        EGOs = list(df['Spread to EGO Diff'])
        SRDs = list(df['SRD'])
        Results = [0,0]
        for game in range(0,len(EGOs)):
            if EGOs[game]*SRDs[game] >= 0: #Then same sign so correct
                Results[0]+=1
            else: #Got it wrong
                Results[1]+=1
        correct = Results[0]
        wrong = Results[1]
        return correct, wrong

    def Analyze_Data(self, total_df, raw_data_path, week):
        #Get numbers for this week
        total_df = total_df[total_df.Home_Team == 0]
        total_games = len(list(total_df['Team']))
        total_correct, total_wrong = self.Evaluate_Picks(total_df)        
        picks_df = total_df[total_df['Pick'].notna()]
        picked_correct, picked_wrong = self.Evaluate_Picks(picks_df)
        #Store resulting week stats in the dictionary
        self.Result_Stats['Season'].append(self.season)
        self.Result_Stats['Week'].append(week)
        self.Result_Stats['Games'].append(total_games)
        self.Result_Stats['EGO Correct'].append(total_correct)
        self.Result_Stats['EGO Wrong'].append(total_wrong)
        self.Result_Stats['EGO Accuracy'].append(round(100*(total_correct/total_games),2))
        self.Result_Stats['Picked Games'].append(picked_correct+picked_wrong)
        self.Result_Stats['Picks Correct'].append(picked_correct)
        self.Result_Stats['Picks Wrong'].append(picked_wrong)
        self.Result_Stats['Pick Accuracy'].append(round(100*(picked_correct/(picked_correct+picked_wrong)),2))

        updated_df = pd.DataFrame()
        stats = list(self.Analyzed_Results)
        for s in stats:
            if 'Season ' in s: #If its one of the season tracking stats
                #Get the previous vals from the columns needed then can add to this weeks Result Stats
                correct_stat = s.split('Season ')[1].split('Accuracy')[0] + 'Correct'
                wrong_stat = s.split('Season ')[1].split('Accuracy')[0] + 'Wrong'
                corrects = sum(list(self.Analyzed_Results[correct_stat]))
                wrongs = sum(list(self.Analyzed_Results[wrong_stat]))
                try:
                    corrects+=self.Result_Stats[correct_stat][-1]
                    wrongs+=self.Result_Stats[wrong_stat][-1]
                except TypeError: #No previous values to add
                    pass
                acc = round(100*(corrects/(corrects+wrongs)),2)
                self.Result_Stats[s].append(acc)
            statVals = list(self.Analyzed_Results[s])
            statVals.append(self.Result_Stats[s][-1])
            updated_df[s] = statVals
        self.Analyzed_Results = updated_df
        # print(self.Analyzed_Results)
