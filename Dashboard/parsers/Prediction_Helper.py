import pandas as pd
import numpy as np
import time
import math
import os
import requests
from urllib.request import urlopen

class Spread_Parser():
    def __init__(self, week, current_week, save_path, Enable_Messaging=False):
        self.Enable_Messaging = Enable_Messaging
        self.week = week
        self.current_week = current_week
        self.save_path = save_path
        self.season = int(self.save_path.split('/')[-1])
        # print(self.season)
        self.source = f'https://www.covers.com/sport/football/NFL/odds'
        if self.week == current_week:
            findSavedData = False #default to getting new spreads
        else:
            findSavedData = True #Not current week so get saved spreads
            self.saved_df = pd.read_csv(f'{self.save_path}/Week {self.week}/spreads.csv')
            if week == 8: #Since the csv won't keep the properly formatted data for some reason..
                self.saved_df = pd.read_csv(f'{self.save_path}/Week {self.week}/spreads.xlsx')

        if findSavedData == False and self.season <= 2019: #Get spread data from saved database
            self.formatted_df = self.Get_Old_Spreads()
            self.saved_df = self.save_spreads(self.formatted_df)
        elif findSavedData == False: #Next week doesn't exist, so need to update this week since is the latest
            self.raw_df = self.Get_Bet_Stats()
            self.game_df = self.format_game_data(self.raw_df)
            self.formatted_df = self.format_spreads(self.game_df)
            self.saved_df = self.save_spreads(self.formatted_df)
        self.parser_df = self.format_by_name(self.saved_df)

    def User_Message(self, message, sleep=1):
        if self.Enable_Messaging:
            print(message)
            time.sleep(sleep)

    def Get_Old_Spreads(self):
        data_path = self.save_path.split('/raw data')[0]
        data = pd.read_csv(f'{data_path}/compiled/All Seasons WDVOA Picking Results.csv')
        data = data.loc[(data['Week'] == self.week) & (data['Year'] == self.season) & (data['Home Team'] == 0)] #Get current week of current season and only use away teams
        df = pd.DataFrame()
        df['Team 1'] = list(data['Team'])
        df['Team 2'] = list(data['Opponent'])
        spreads = list(data['Spread'])
        spreads = [f'{s} / {s*-1}' for s in spreads] #Put spreads together
        spreads = [f'+{s}' if float(s.split(' / ')[0])>=0 else s for s in spreads] #Add the + symbol if underdog
        # print(spreads)
        df['Betting Spread'] = spreads #Add spreads to df
        # df['Spread'] = [f'{s} / {s*-1}' for s in spreads] #Add spreads to df
        df['Open'] = spreads #Add same data to the open since not needed any way
        df['Game_Time'] = ['-' for i in range(len(spreads))]
        # print(df)
        return df

    def Get_Bet_Stats(self):
        found_data = False
        attempts = 0
        self.User_Message('Searching For Latest NFL Odds ...')
        while not found_data:
            time.sleep(0.5)
            attempts+=1
            html = requests.get(self.source).content
            try:
                df_list = pd.read_html(html, header=0, index_col=None)
                self.User_Message("Found Data!")
                found_data = True
            except:
                self.User_Message("Couldn't Retrieve Data")
                self.User_Message('Attempting Again')
                found_data = False
            if attempts > 10:
                found_data = True
                self.User_Message('Too many failed attempts. Aborting ...')
        
        self.User_Message('Collecting Data ...')
        time.sleep(0.5)
        #Put the Data into Dataframe format
        Week_DFs = []
        matchup_df = df_list[2] #3rd table
        spreads_df = df_list[3] #4th table
        betting_spreads = spreads_df.iloc[:,-1] #Get the last column
        # betting_spreads = list(spreads_df.iloc[:,-2]) #Get the 2nd last column
        matchup_df['Betting Spread'] = betting_spreads
    
        #Clean up the Data
        all_cols = list(matchup_df)
        good_cols = ['Game', 'Open', 'Betting Spread']
        #Drop Other unneeded columns
        for col in all_cols:
            if col not in good_cols:
                matchup_df.drop(col, axis=1, inplace=True)
        #Drop nan rows
        matchup_df.dropna(axis=0, how="any", inplace=True)
        # matchup_df = matchup_df[1:] #take the data less the header row
        time.sleep(5)
        # matchup_df.to_csv(f'raw_spreads.csv', index=False)
        return matchup_df

    def format_game_data(self, df):
        self.User_Message('Formatting Data ...')
        time.sleep(0.5)
        game_col = df['Game'].tolist()
        newCols = ['Game_Time', 'Team 1', 'Team 2']
        newData = [[] for i in range(len(newCols))]
        for game in game_col:
            game_data = game.split('  ')
            dps = len(game_data)
            if dps > 5: #Game in progress or finished
                team1_pos = dps-5
                team2_pos = dps-3
            else:
                team1_pos = dps-3
                team2_pos = dps-2
            team1 = game_data[team1_pos]
            # team1 = game_data[dps-2-(2*score_pos)]
            team2 = game_data[team2_pos]
            # team2 = game_data[dps-1-score_pos]
            game_time = f'{game_data[0]}'# {game_data[1]}'
            newData[0].append(game_time)
            newData[1].append(team1)
            newData[2].append(team2)
        for i in range(0, len(newData)):
            df[newCols[i]] = newData[i]
        del df['Game']
        cols = list(df)
        ordered_cols = cols[-3:] + cols[:-3]
        df = df[ordered_cols]
        # df = df[df.Game_Time != 'Final'] #If need to exclude last weeks games (since still showing up early in the week)
        # self.User_Message(df)
        return df

    #Format the Spread Values
    def format_spreads(self, df):
        open_spreads = list(df['Open'])
        bet_spreads = list(df['Betting Spread'])
        for game in range(len(open_spreads)):
            open_spread_parts = str(open_spreads[game]).split('  ')
            open_spread = f'{open_spread_parts[0]} / {open_spread_parts[1]}'
            open_spreads[game] = open_spread
            bet_spread_parts = str(bet_spreads[game]).split(' ')
            # self.User_Message(bet_spread_parts)
            bet_spread = f'{bet_spread_parts[0]} / {bet_spread_parts[3]}' 
            bet_spreads[game] = bet_spread
        df['Open'] = open_spreads
        df['Betting Spread'] = bet_spreads
        # self.User_Message(df)
        # print(df)
        return df

    def save_spreads(self, df):
        self.User_Message('Saving Data ...')
        time.sleep(0.5)
        #rename columns to standard format
        df = df.rename(columns={"Open": "Opening Spread"})
        final_cols = ['Opening Spread', 'Team 1', 'Team 2', 'Game_Time', 'Betting Spread']
        df = df.reindex(columns=final_cols)
        # self.User_Message(df)
        # df.to_csv(f'{self.save_path}/Week {self.week}/spreads.xlsx', index=False)
        df.to_csv(f'{self.save_path}/Week {self.week}/spreads.csv', index=False)
        return df

    def format_by_name(self, df):
        self.User_Message('Adding to Parser ...')
        time.sleep(0.5)
        name_formated_df = pd.DataFrame()
        #Get the 2 Team Name Cols we're combining
        Name_Cols = ['Team 1', 'Team 2']
        #Get All Columns of the DF
        Info_Cols = list(df)
        #Make list to store combined team names
        all_teams = []
        #Add the team name cols together and remove these columns from the All Cols list while we're at it
        for col in Name_Cols:
            all_teams += df[col].tolist()
            Info_Cols.remove(col)
        #Put the combined tam names into new df
        name_formated_df['Team'] = all_teams
        #Go through each of the rest of the columns
        for col in Info_Cols:
            #If spread in the col name, then need to split it so that add first half then second half
            if "Spread" in col:
                new_cols = [[] for i in range(len(Name_Cols))]
                for s in list(df[col]):
                    s_parts = s.split(' / ')
                    # print(s_parts)
                    for pos in range(len(new_cols)):
                        spread_val = s_parts[pos]
                        if '+' in spread_val:
                            spread_val = spread_val.replace("+", "")
                        new_cols[pos].append(spread_val)
                total_new_col = []
                for c in range(len(new_cols)):
                    total_new_col += new_cols[c]
            else:
                total_new_col = list(df[col])*2
            total_new_col = [s if s!='PK' else 0 for s in total_new_col]
            name_formated_df[col] = total_new_col
        return name_formated_df


class Game_Info_Parser():
    def __init__(self, week, season, path):
        self.week = week
        self.season = season
        #Schedule Info
        self.Week_Sched_DF = self.Week_Formated_DF()
        #DVOA Info
        # self.WDVOA_DF = self.Get_WDVOA_DF()
        # week_path = f'{path}/Week {week}'
        # self.WDVOA_DF = pd.read_csv(f'{week_path}/DVOA Data.csv')
        # self.season_wdvoaTeams = self.WDVOA_DF['TEAM'].tolist()
        # self.season_wdvoaTeams.sort()

    def Get_WDVOA_DF(self):
        Possible_WeightedDVOA_Names = ["Weighted DVOA", "WEI.DVOA", "WEIGHTED DVOA", "WEIGHTED  DVOA", "WEIGHTEDDVOA", "WEIGHTEDVOA", "DAVE", "TOTAL DAVE", "TOTALDAVE", "TOTAL  DAVE", 'WEI.  DVOA']
        have_df = False
        try:
            URL_BASE = f'https://www.footballoutsiders.com/stats/nfl/team-efficiency/{self.season}'
            # URL_BASE = f'https://www.footballoutsiders.com/dvoa-ratings/{self.season}/week-{self.week-1}-dvoa-ratings'
            html = requests.get(URL_BASE).content
            for head_row in range(0,2):
                df_list = pd.read_html(html, header=head_row, index_col=0)
                # print(df_list)
                for i in range(0, len(df_list)):
                    Pot_WDVOA_DF = df_list[i]
                    # print(Pot_WDVOA_DF)
                    Pot_WDVOA_DF.to_csv(f'{head_row}_{i}_potDF.csv')
                    pot_dvoaHeaders = list(Pot_WDVOA_DF)
                    # print(pot_dvoaHeaders)
                    if len(pot_dvoaHeaders) >= 10 and "TEAM" in pot_dvoaHeaders: #The right df
                        for header in Possible_WeightedDVOA_Names:
                            if header in pot_dvoaHeaders: #Have the weighted df
                                WDVOA_DF = Pot_WDVOA_DF
                                #Save DF and Change the WDVOA column to be named WDVOA
                                cols = list(WDVOA_DF)
                                new_cols = [x if x != header else 'WDVOA' for x in cols]
                                WDVOA_DF.columns = new_cols
                                have_df = True
                                break
                if have_df == True:
                    break
        except ValueError:
            print("error")
            pass
        if have_df == False: #Never found a good df
            print(f'No Good Data: {self.season}, week {self.week}')
            print(pot_dvoaHeaders)
        #Drop any header rows
        WDVOA_DF.drop(WDVOA_DF.loc[WDVOA_DF['TEAM']=="TEAM"].index, inplace=True)
        WDVOA_DF = WDVOA_DF.sort_values(by=["TEAM"])
        return WDVOA_DF

    def Get_Week_Schedule(self):
        URL = f'https://www.pro-football-reference.com/years/{self.season}/games.htm'
        html = requests.get(URL).content
        df_list = pd.read_html(html, header=0, index_col=None)
        schedule_df = df_list[0]
        schedule_headers = list(schedule_df)
        #Make the Week # the index
        schedule_df.set_index(keys=['Week'], drop=False,inplace=True)
        # now we can perform a lookup on a 'view' of the dataframe
        Week_Sched_DF = schedule_df.loc[schedule_df.Week==str(self.week)]
        return Week_Sched_DF
    
    def Week_Formated_DF(self):
        Week_Sched_DF = self.Get_Week_Schedule()
        Week_DF = pd.DataFrame()
        Name_Cols = ['Winner/tie', "Loser/tie"]
        Data_Cols = ['Unnamed: 5', 'PtsW', 'PtsL']
        Info_Cols = list(Week_Sched_DF)
        thisWeek_schedTeams = []
        thisWeek_schedOpps = []
        for col in range(len(Name_Cols)):
            thisWeek_schedTeams += Week_Sched_DF[Name_Cols[col]].tolist()
            thisWeek_schedOpps += Week_Sched_DF[Name_Cols[col-1]].tolist() #Do the Loser/tie then the Winner
            Info_Cols.remove(Name_Cols[col])
        Week_DF['Team'] = thisWeek_schedTeams
        Week_DF['Opponent'] = thisWeek_schedOpps
        for col in Info_Cols:
            if col == 'PtsW':
                Week_DF['Points For'] = list(Week_Sched_DF['PtsW']) + list(Week_Sched_DF['PtsL'])
            elif col == 'PtsL':
                Week_DF['Points Against'] = list(Week_Sched_DF['PtsL']) + list(Week_Sched_DF['PtsW'])
            elif col == 'Unnamed: 5':
                home_away = Week_Sched_DF['Unnamed: 5'].tolist()
                first_teams = []
                second_teams = []
                for g in home_away:
                    if g == '@':
                        first_teams.append(0)
                        second_teams.append(1)
                    else:
                        first_teams.append(1)
                        second_teams.append(0)
                    home_teams = first_teams+second_teams
                Week_DF['Home_Team'] = home_teams
            else:
                Week_DF[col] = Week_Sched_DF[col].tolist()*2
        return Week_DF
    
class Team_Matching():
    def __init__(self, raw_data_path, dfs):
        self.raw_data_path = raw_data_path
        self.name_cols = ['TEAM', 'Team', 'Team']
        self.raw_dfs = dfs
        try:
            self.Name_Map = pd.read_csv(f'{self.raw_data_path}/Names.csv')
            # print(self.Name_Map)
        except:
            self.Name_Map = self.make_name_map(dfs)
        self.Combined_Raw_DF = self.Combine_DFs()

    def make_name_map(self, dfs):
        '''
        #Want to get the 3 columns all mapped to eachother such that:
            #All season teams (the WDVOA teams) are the first collumn, spread teams are second, schedule is 3rd
        #Goal is that for any week, we can map the other 2 columns to the order of the 1st column containing all the teams
        #Will look at each letter for the spread and wdvoa abbreviated names
        #Will look at each first letter for the schedule names
        #For each Wteam in wdvoaTeams:
            #For each Steam in spreadTeam:
                #For each letter of Wteam
        
        #End product is csv where can look up each column for the team name and re-name it to the name in the first column
        #Then every df we read will have the same team names
        #Then for the master df, can look up the values needed from other dfs using the team name as the cross reference point
        '''
        Names = []
        for df in range(len(dfs)):
            Names.append(dfs[df][self.name_cols[df]].tolist())
        # print(Names)
        Mapped_Names = [["" for i in range(len(Names[0]))] for c in range(len(Names)-1)] #New lists where names will be stored
        # print(Mapped_Names)
        Name_Map = pd.DataFrame() #New DF where names will be stored
        Name_Map['Ref Name'] = Names[0] #Put first column already with WDVOA names
        for c in range(1, len(Names)): #For each of the other name columns passed in
            for n in range(len(Names[c])): #For each name in the name column
                Scores = [0 for i in range(len(Names[0]))] #The scores of each team wrt the reference list
                #This will assign a score to each team from the non-reference list for every team from the reference list
                #After each reference team is examined, the position with the highest score will be used to index from the non-reference list the team which corresponds to the reference team
                #For example, if the highest score is in position 3, then the team in the 3rd position from the non reference list will be appended to the new list
                #This appending will put that team in the same position as the reference one we just examined, since we're going in order
                ogName = Names[c][n].upper()
                ogParts = ogName.split(' ')
                if len(ogParts) == 1: #Then it was 1 word meaning an abbreviation. So look at each letter
                    ogName = ogParts[0]
                elif len(ogParts) == 2: #Then its city and mascot so look for first 4 letters of city
                    ogName = ogParts[0][0:3]
                else: #Then it was words meaning city and mascot so look at first letter of each word
                    ogName = [ogParts[i][0] for i in range(len(ogParts))] #Get the first letter from each word
                for Wpos in range(len(Names[0])): #For each WDVOA name
                    refName = Names[0][Wpos] #Name it
                    if len(refName) < 3: #If name is only 2 letters long, add the last letter again since this name is losing points to 3 letter ones that might happen to have the same 2 letters in them as the 2 letter name (LV vs LAR for the LVR)
                        refName = f'{refName}{refName[-1]}'
                    for ogL in range(1, len(ogName)): #For each letter of the team in that spread data
                        #Check if first letter matches. This is a must.
                        if ogName[0] == refName[0]: #Score points when the first letters match. This happens more often (for every letter looked at) than for other letters since first ones are most important
                            Scores[Wpos]+=1
                            for refL in range(1, len(refName)): #For each letter of the team in that original data
                                if ogName[ogL] == refName[refL]:
                                    Scores[Wpos]+=1
                #Go through each score from the teams and put the ogname with the highest score in the reference name position
                last_hs = 0
                for score in range(0, len(Scores)):
                    if Scores[score] > last_hs:
                        last_hs = Scores[score]
                        hs_pos = score
                Mapped_Names[c-1][hs_pos] = Names[c][n]

        for nCol in range(len(Mapped_Names)):
            Name_Map[f'Mapped Name {nCol}'] = Mapped_Names[nCol]
        # print(Name_Map)
        Name_Map.to_csv(f'{self.raw_data_path}/Names.csv', index=False)
       
        return Name_Map


    def Combine_DFs(self):
        '''
        Read in naming map
        Change names in each df such that they match (if name in col1 don't change, else change such that = name from col 1)
        Drop rows where team doesn't exist in Week_DF
        Then can just copy the columns wanted from WDVOA and Spread DF over to Week_DF
        Sort the Week DF after
        '''
        #PART 1 - MATCH the NAMES
        #Put the names together
        #Check each df
        for df in range(1,len(self.raw_dfs)):
            #list the df
            og_df = self.raw_dfs[df]
            # print(og_df)
            #get the teams from that df
            teams = og_df[self.name_cols[df]].tolist()
            for t in range(len(teams)):
                #Look up that team in the name map
                try:
                    name = self.Name_Map.loc[self.Name_Map[f'Mapped Name {df-1}'] == teams[t]]['Ref Name'].tolist()[0]
                except:
                    print(teams[t])
                # Replace the name in the original df with the reference name
                og_df = og_df.replace(teams[t],name)
                self.raw_dfs[df] = og_df
            #Sort the df and make the team name the index
            self.raw_dfs[df] = self.raw_dfs[df].sort_values(by=[self.name_cols[df]], ascending=True)
            self.raw_dfs[df] = self.raw_dfs[df].set_index(self.name_cols[df])
        #Re-index the df that wasn't itterated through
        self.raw_dfs[0] = self.raw_dfs[0].set_index(self.name_cols[0])
        
        #PART 2 - Combine the self.raw_dfs using the Names as the cross reference
        #Define lists we need
        Possible_Additional_Names = ['WDVOA', 'Betting Spread']# ["Betting Spread", "WEI.DVOA", "WEIGHTED DVOA", "WEIGHTEDDVOA", "WEIGHTEDVOA", "DAVE", "TOTAL DAVE", "TOTALDAVE", "TOTAL  DAVE", 'WEI.  DVOA']
        keep_cols = {}
        df_teams = []
        #Get the teams common to every DF
        for df in self.raw_dfs:
            df_teams.append(list(df.index))
        common_teams = set(df_teams[0])
        for s in df_teams[1:]:
            common_teams.intersection_update(s)
        # print(common_teams)
        #Drop any columns from each DF that don't have those tams
        for df in range(len(self.raw_dfs)):
            for t in list(self.raw_dfs[df].index):
                if t not in common_teams:
                    self.raw_dfs[df] = self.raw_dfs[df].drop(t)
        #For each DF, either save the data we want or save the whole DF if it's the schedule DF
        for df in self.raw_dfs:
            cols = list(df)
            for col in cols:
                if 'Date' in col:
                    Combined_DF = df
                else:
                    for name in Possible_Additional_Names:
                        if name == col:
                            keep_cols[name] = df[name].tolist()
        #Add the saved data to the main DF
        for key in keep_cols.keys():
            Combined_DF[key] = keep_cols.get(key)
        Combined_DF = Combined_DF.reset_index()
        # print(Combined_DF)

        return Combined_DF


class EGO_Prediction():
    def __init__(self, project_path, Map_DF):
        self.project_path = project_path
        self.Map = self.Setup_Map(Map_DF)
        self.target_egospr_diffs = [[-3.7,-1.5],[1.5,3.7]]

    def Setup_Map(self, Map_DF):
        #Need to take all the DVOA_Diffs and map them to an EGO via the total home or away map
        Game_Locations = ["Away", "Home"]
        Maps = [{} for i in range(len(Game_Locations))]
        for loc in range(0, len(Game_Locations)):
            #Make List of the 2 columns needed
            DF_EGOs = Map_DF[f'{Game_Locations[loc]} EGO'].tolist()
            DF_Diffs = Map_DF[f'{Game_Locations[loc]} DVOA Diff Range'].tolist()
            for diff in range(0, len(DF_Diffs)):
                LL = float(DF_Diffs[diff].split("to ")[0])
                UL = float(DF_Diffs[diff].split("to ")[1])
                if LL == -100:
                    avg = -75
                elif UL == 100:
                    avg = 75
                else:
                    avg = round(((LL + UL)/2),2)
                DF_Diffs[diff] = avg
                DF_EGOs[diff] = round(float(DF_EGOs[diff]),2)
                # self.Map
            Maps[loc] = {DF_Diffs[i]:DF_EGOs[i] for i in range(len(DF_EGOs))}
            # print(Maps)
        return Maps

    def Target_Spreads(self, EGO, spread):
        target_spreads = [[],[]]
        pick = 0
        for targets in range(len(self.target_egospr_diffs)):
            #Get the target spread values
            for diff in range(len(self.target_egospr_diffs[targets])):
                target_spread = round(EGO+self.target_egospr_diffs[targets][diff],1) #flip the sign since making a spread
                target_spread = -1*round(target_spread*2)/2 #Round to nearest .5 and flip sign since making a spread
                target_spreads[targets].append(target_spread) #flip the sign since making a spread
                #For EGO of 6: [[2.3, 4.5],[7.5,9.7]]
            #Check if the spread is in this range. If so then make a pick
            min_spread = target_spreads[targets][0]
            max_spread = target_spreads[targets][1]
            if spread <= min_spread and spread >= max_spread: #signs are opposite than intuition since spreads are opposite
                pick = 1

        return target_spreads, pick

    def Calculate_Data(self, df):
        self.Calculated_Data = {'WDVOA Delta':[], 'EGO':[], 'Spread to EGO Diff':[], 'Margin to EGO Diff':[], 'Target Spreads':[], 'Make Pick':[], 'Pick':[], 'Pick Right':[]}
        for team_row in range(len(list(df['Season']))):
            #Get needed stats for the team_row
            team = df.iloc[team_row]['Team']
            team_wdvoa = df.iloc[team_row]['WDVOA'].split('%')[0]
            opp = df.iloc[team_row]['Opponent']
            loc = df.iloc[team_row]['Home_Team']
            spread = float(df.iloc[team_row]['Betting Spread'])
            margin = float(df.iloc[team_row]['Scoring Margin'])
            SRD = float(df.iloc[team_row]['SRD'])
            #Get Opponent Stats
            for t in range(len(list(df['Team']))):
                if opp == list(df['Team'])[t]:
                    opp_row = t
                    break
            opp_wdvoa = df.iloc[opp_row]['WDVOA'].split('%')[0]
            #Get the WDVOA Diff and Add to the Data
            wdvoa_diff = float(team_wdvoa)-float(opp_wdvoa)
            self.Calculated_Data['WDVOA Delta'].append(wdvoa_diff)
            #Get the EGO and add to the Data
            EGO = round(np.interp(wdvoa_diff, list(self.Map[loc].keys()), list(self.Map[loc].values())),2)
            self.Calculated_Data['EGO'].append(EGO)
            #Get the spread and margin to EGO result difference
            self.Calculated_Data['Spread to EGO Diff'].append(EGO+spread)
            self.Calculated_Data['Margin to EGO Diff'].append(margin-EGO)
            #Get Target Spread for each EGO
            target_spreads,makePick = self.Target_Spreads(EGO, spread)
            if makePick == 0:
                if abs(EGO+spread)>self.target_egospr_diffs[1][-1]: #If -2+(-3) then 
                    target_spreads = "Missing Something ..."
                elif abs(EGO+spread)<self.target_egospr_diffs[1][0]:
                    target_spreads = "Too Close To Call"
                pick = ""
            else:
                if EGO+spread > 0:
                    pick = team
                else:
                    pick = opp
            self.Calculated_Data['Target Spreads'].append(target_spreads)
            self.Calculated_Data['Make Pick'].append(makePick)
            self.Calculated_Data['Pick'].append(pick)
            #EGO correct if EGO/Spread Diff same sign as SRD
            if margin == 0 and (spread-SRD) == 0: #Game hasn't been played yet since no winner and no difference form spread to scoring margin
                self.Calculated_Data['Pick Right'].append('')
            elif (EGO+spread)*SRD>= 0: #Then same sign so correct
                self.Calculated_Data['Pick Right'].append(1)
            else: #Got it wrong
                self.Calculated_Data['Pick Right'].append(0)
        #final Results
        # print(self.Calculated_Data)
        time.sleep(1)
        return self.Calculated_Data
