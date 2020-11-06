import pandas as pd
import time
import math
import os
import requests
from urllib.request import urlopen

class HISTORICAL_PARSING():

    def __init__(self, Week_DF, This_Week_Teams, All_Season_Teams, season, week):
        self.time = time
        self.season = season
        self.week = week
        self.This_Week_Teams = This_Week_Teams
        self.All_Season_Teams = All_Season_Teams
        self.Week_DF = Week_DF

        self.Result_Levels = [-9, -0.5, 0, 9, 100] #Big Upset, Upset, Neatural, Neutral, Blowout, Beatdown
        self.Result_Markers = [-2, -1, 0, 1, 2]
        self.Locations = ["H", "R"] #Corresponds to [1, 0]
        self.Game_Days = ["Thursday"] #Short Weeks are thursday games, otherwise will consider the game a sunday/saturday/monday which is a normal amount of prep time


    def Get_Spread_Results(self):
        Spread_Result_Diff = []
        Spread_Result_Class = []
        spreads = self.Week_DF['Spread'].tolist()
        print(spreads)
        game_margins = self.Week_DF['Game Scoring Margin'].tolist()
        for tm in range(0,len(spreads)):
            try:
                print(tm)
                print(spreads)
                print(spreads[tm])
                print(game_margins)
                print(game_margins[tm])
                # if math.isnan(int(spreads[tm])): #if nan
                #     Spread_Result_Diff.append("")
                #     Spread_Result_Class.append("")
                #     print("Nan here")
                #     continue
                spread_result = float(spreads[tm])+float(game_margins[tm])
                Spread_Result_Diff.append(spread_result)
                for result_type in range(0, len(self.Result_Levels)):
                    if spread_result <= self.Result_Levels[result_type]: #from really negative to really positive
                        if float(spreads[tm]) <= 0:
                            Spread_Result_Class.append(self.Result_Markers[result_type])
                            break
                        else:
                            Spread_Result_Class.append(-1*self.Result_Markers[result_type]) #Flip the sign
                            break
                    elif math.isnan(float(spreads[tm])):
                        print('NAN detected')
                        Spread_Result_Class.append(0)
                        Spread_Result_Diff[-1]=0
                        break
            except:
                print('HERE')
                print(spreads[tm])
                Spread_Result_Diff.append("")
                Spread_Result_Class.append("")
                # self.Week_DF.drop([tm], inplace=True)
        print(spreads)
        print(len(spreads))
        print(len(Spread_Result_Diff))
        print(len(Spread_Result_Class))
        self.Week_DF['Spread to Result Difference'] = Spread_Result_Diff #Spread+PD - Really negative means favoured team lost or won by less than expected or unfavoured lost by more than expected, really positive means favoured team won by more than expected or unfavoured team won, close to zero means outcome was similar to spread
        self.Week_DF['Spread Result Class'] = Spread_Result_Class

    def Get_Game_Location(self):
        #read df for home teams for this week and put teams into list
        Team_Locaitons = [[] for i in range(len(self.Locations))]
        Week_Locations = []
        for loc_type in range(0,len(self.Locations)):
            self.URL_BASE = f'https://www.pro-football-reference.com/play-index/tgl_finder.cgi?request=1&match=single&year_min={self.season}&year_max={self.season}&game_type=R&game_num_min=0&game_num_max=99&week_num_min={self.week}&week_num_max={self.week}&temperature_gtlt=lt&game_location={self.Locations[loc_type]}&league_id=NFL&c5val=1.0&order_by=points_diff'
            #sorted_df = self.Get_DF()
            html = requests.get(self.URL_BASE).content
            df_list = pd.read_html(html, header=1, index_col=None)
            df = df_list[-1]
            df = df.sort_values(by=['Year', 'Tm'])
            Team_Locaitons[loc_type] = df["Tm"].tolist()
            print(Team_Locaitons[loc_type])
            
        #All_Teams += Team_Locaitons[loc_type
        for team in self.All_Season_Teams:
            location = "" #Default to nothing if team turns out not to play this week
            for hm_team in Team_Locaitons[0]:
                if hm_team == team:
                    location = 1
                    break
            for aw_team in Team_Locaitons[1]:
                if aw_team == team:
                    location = 0
                    break
            Week_Locations.append(location)
        self.Week_DF['Home Team'] = Week_Locations
        #print(self.Week_DF)
    
    def Get_Game_Day(self):
        #read df for home teams for this week and put teams into list
        Short_Week_Teams = []
        Short_Weeks = []
        for day_type in range(5,6):
            self.URL_BASE = f'https://www.pro-football-reference.com/play-index/tgl_finder.cgi?request=1&match=single&year_min={self.season}&year_max={self.season}&game_type=R&game_num_min=0&game_num_max=99&week_num_min={self.week}&week_num_max={self.week}&game_day_of_week={day_type}&temperature_gtlt=lt&league_id=NFL&c5val=1.0&order_by=points_diff'
            try:
                #sorted_df = self.Get_DF()
                html = requests.get(self.URL_BASE).content
                df_list = pd.read_html(html, header=1, index_col=None)
                df = df_list[-1]
                df = df.sort_values(by=['Year', 'Tm'])
                Short_Week_Teams = df["Tm"].tolist()
                print(Short_Week_Teams)
            except:
                pass
        #All_Teams += Team_Locaitons[loc_type
        for team in self.All_Season_Teams:
            short_week = 0 #Default to nothing if team turns out not to play this week
            for sh_team in Short_Week_Teams:
                if sh_team == team:
                    short_week = 1
                    break
            Short_Weeks.append(short_week)
        self.Week_DF['Short Week'] = Short_Weeks

    def Get_Bye_Week(self):
        Off_Bye_Week_Teams = []
        Off_Bye_Weeks = []
        for day_type in range(5,6):
            self.URL_BASE = f'https://www.pro-football-reference.com/play-index/tgl_finder.cgi?request=1&match=single&year_min={self.season}&year_max={self.season}&game_type=R&game_num_min=0&game_num_max=99&week_num_min={self.week}&week_num_max={self.week}&temperature_gtlt=lt&league_id=NFL&game_after_bye=Y&c5val=1.0&order_by=points_diff'
            try:
                #sorted_df = self.Get_DF()
                html = requests.get(self.URL_BASE).content
                df_list = pd.read_html(html, header=1, index_col=None)
                df = df_list[-1]
                df = df.sort_values(by=['Year', 'Tm'])
                Off_Bye_Week_Teams = df["Tm"].tolist()
                print(Off_Bye_Week_Teams)
            except:
                pass
        #All_Teams += Team_Locaitons[loc_type
        for team in self.All_Season_Teams:
            off_bye_week = 0 #Default to nothing if team turns out not to play this week
            for bye_team in Off_Bye_Week_Teams:
                if bye_team == team:
                    off_bye_week = 1
                    break
            Off_Bye_Weeks.append(off_bye_week)
        self.Week_DF['Off Bye Week'] = Off_Bye_Weeks

    def Get_Opponents(self):
        Opponents = ["" for i in range(len(self.This_Week_Teams))] #Empty list of length = num teams
        print(Opponents)
        Comp_Stats = [[] for i in range(5)]
        first_teams = []
        comp_teams = []
        Comp_Stats[0] = self.Week_DF["Team"].tolist()
        Comp_Stats[1] = self.Week_DF["Spread"].tolist()
        Comp_Stats[2] = self.Week_DF["Over/Under"].tolist()
        Comp_Stats[3] = self.Week_DF["Home Team"].tolist()
        Comp_Stats[4] = self.Week_DF["Game Scoring Margin"].tolist()
        solved_team_count = -1
        while len(Comp_Stats[0]) > 0: #Until we delete all teams from list to go through
            for team_num in range(0, len(Comp_Stats[0])): #Do 1 team at a time and shorten list each time
                comp_teams = []
                for stat in range(0,len(Comp_Stats)): #Fill in each of the 4 stats we're comparing
                    if team_num == 0:
                        #if stat == 0:
                        try:
                            first_teams.append(float(Comp_Stats[stat][team_num]))
                            #solved_team_count += 1
                        except ValueError:
                            first_teams.append(Comp_Stats[stat][team_num])
                    else:
                        try:
                            comp_teams.append(float(Comp_Stats[stat][team_num]))
                        except ValueError:
                            comp_teams.append(Comp_Stats[stat][team_num])
                print(first_teams)
                #print(comp_teams)
                if first_teams[1] == "": #No game this week
                    print("No Game this week")
                    break
                elif team_num > 0: #Don't compare first team
                    if first_teams[1] == -1*comp_teams[1] and first_teams[2] == comp_teams[2] and first_teams[3] != comp_teams[3] and first_teams[4] == -1*comp_teams[4]:
                        for team in range(0, len(self.This_Week_Teams)):
                            if self.This_Week_Teams[team] == Comp_Stats[0][team_num]: #Team we're solving for = team 
                                Opponents[team] = Comp_Stats[0][0] #Opponent gets the first team
                            elif self.This_Week_Teams[team] == Comp_Stats[0][0]:
                                Opponents[team] = Comp_Stats[0][team_num] #This teams opponent
                        break
            for stat in range(0,len(Comp_Stats)):
                del Comp_Stats[stat][0] #delete the team we were solving for
                if len(comp_teams) > 0: #only if got a 2nd team
                    del Comp_Stats[stat][team_num-1] #delete the team's match (-1 since the previous delete of first item)

            first_teams = []
            print("Opponents:")
            print(Opponents)
        self.Week_DF['Opponent'] = Opponents
        print(self.Week_DF)


    def Do_Betting_Stuff(self):
        self.Get_Spread_Results()
        return self.Week_DF

    def Do_Game_Stuff(self):
        #Get the information about this week's games
        # self.Get_Spread_Results()
        self.Get_Game_Location()
        self.Get_Game_Day()
        self.Get_Bye_Week()
        self.Get_Opponents()
        return self.Week_DF

class Betting_Parsing():
    def __init__(self, week_teams, project_path):
        self.project_path = project_path
        self.week_teams = week_teams
        self.source = f'https://www.covers.com/sport/football/NFL/odds'

    def Get_Bet_Stats(self):
        #Query the Data from Online
        # URL = f'https://www.donbest.com/nfl/odds/'
        # print(URL)
        found_data = False
        attempts = 0
        print('Retrieving Latest NFL Odds ...')
        while not found_data:
            time.sleep(1)
            attempts+=1
            html = requests.get(self.source).content
            try:
                df_list = pd.read_html(html, header=0, index_col=None)
                print("Found Data!")
                found_data = True
            except:
                print("Couldn't Retrieve Data")
                print('Attempting Again')
                found_data = False
            if attempts > 10:
                found_data = True
                print('Too many failed attempts. Aborting ...')
        
        print('Collecting Data ...')
        time.sleep(1)
        #Put the Data into Dataframe format
        Week_DFs = []
        matchup_df = df_list[2] #3rd table
        spreads_df = df_list[3] #4th table
        betting_spreads = spreads_df.iloc[:,-1] #Get the last column
        betting_spreads2 = list(spreads_df.iloc[:,-1]) #Get the last column
        matchup_df['Betting Spread'] = betting_spreads2
    
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
        print('Formatting Data ...')
        time.sleep(1)
        game_col = df['Game'].tolist()
        newCols = ['Game Time', 'Team 1', 'Team 2']
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
        # print(df)
        return df

    #Format the Spread Values
    def format_spreads(self, df):
        print('Saving Data ...')
        time.sleep(1)
        open_spreads = list(df['Open'])
        bet_spreads = list(df['Betting Spread'])
        for game in range(len(open_spreads)):
            open_spread_parts = str(open_spreads[game]).split('  ')
            open_spread = f'{open_spread_parts[0]} / {open_spread_parts[1]}'
            open_spreads[game] = open_spread
            bet_spread_parts = str(bet_spreads[game]).split(' ')
            # print(bet_spread_parts)
            bet_spread = f'{bet_spread_parts[0]} / {bet_spread_parts[3]}' 
            bet_spreads[game] = bet_spread
        df['Open'] = open_spreads
        df['Betting Spread'] = bet_spreads
        # print(df)
        return df

    def match_to_week_data(self, df):
        # week_teams = self.Week_DF['Team'].tolist()
        spread_teams = df['Team 1'].tolist() + df['Team 2'].tolist()
        formated_spreads = df['Betting Spread'].tolist()
        self.week_spreads = ['' for i in range(len(self.week_teams))]
        spreads = []
        for t in range(0,2):
            for s in formated_spreads:
                spread_parts = s.split(' / ')
                spread = spread_parts[t]
                if '+' in spread:
                    spread = spread.replace("+", "")
                spreads.append(spread)
        print(spreads)
        for st in range(0,len(spread_teams)): #For each team in the spread data
            Scores = [0 for i in range(len(self.week_teams))]
            for wt in range(0, len(self.week_teams)): #For each team in the original data
                for sl in range(1, len(spread_teams[st])): #For each letter of the team in that spread data
                    #Check if first letter matches. This is a must.
                    if spread_teams[st][0] == self.week_teams[wt][0]:
                        Scores[wt]+=1
                        for wl in range(1, len(self.week_teams[wt])): #For each letter of the team in that original dat
                            if spread_teams[st][sl] == self.week_teams[wt][wl]:
                                Scores[wt]+=1

            last_hs = 0
            for score in range(0, len(Scores)):
                if Scores[score] > last_hs:
                    last_hs = Scores[score]
                    hs_pos = score
            self.week_spreads[hs_pos] = spreads[st]
        print(self.week_spreads)
        # time.sleep(10)

    def save_spreads(self, df):
        #rename columns to standard format
        df = df.rename(columns={"Open": "Opening Spread"})
        final_cols = ['Opening Spread', 'Team 1', 'Team 2', 'Game Time', 'Betting Spread']
        df = df.reindex(columns=final_cols)
        print(df)
        df.to_csv(f'{self.project_path}/Raw Data/latest_spreads.xlsx', index=False)

    def Get_Current_Spreads(self):
        raw_df = self.Get_Bet_Stats()
        game_df = self.format_game_data(raw_df)
        formatted_df = self.format_spreads(game_df)
        # self.match_to_week_data(formatted_df)
        self.save_spreads(formatted_df)
        
        return self.week_spreads

class This_Week_Parsing():
    def __init__(self, season, week, min_week, All_Season_Teams, season_path):
        self.time = time
        self.season = season
        self.week = week
        self.min_week = min_week
        self.All_Season_Teams = All_Season_Teams
        self.season_path = season_path

    def Change_Team_Names(self):
        Naming_Reference = pd.read_csv(f'{self.season_path}/Names.csv')
        # Sched_Team_Names = list(self.Team_Name_Dict.values())
        # New_Team_Names = list(self.Team_Name_Dict.keys())
        Sched_Team_Names = Naming_Reference['Long_Names'].tolist()
        New_Team_Names = Naming_Reference['Short_Names'].tolist()
        print(New_Team_Names)
        for col in ["Winner/tie", "Loser/tie"]:
            Col_Teams = self.Week_Sched_DF[col].tolist()
            for tm in range(0, len(Col_Teams)):
                for val in range(0, len(Sched_Team_Names)):
                    if Sched_Team_Names[val] == Col_Teams[tm]:
                        new_name = New_Team_Names[val]
                        print(new_name)
                        # time.sleep(1)
                        Col_Teams[tm] = new_name
                        self.Week_Sched_DF[col] = Col_Teams
                        # self.Week_Sched_DF[tm, col] = new_name
        print(self.Week_Sched_DF)
        # time.sleep(10)

    def Get_Team_Names(self):
        print("a")
        self.Win_Teams = self.Week_Sched_DF['Winner/tie'].tolist()
        self.Lose_Teams = self.Week_Sched_DF["Loser/tie"].tolist()
        self.All_Week_Teams = self.Win_Teams + self.Lose_Teams
        self.All_Week_Teams.sort()# = sort(self.All_Week_Teams)
        print(self.All_Week_Teams)

    def Check_Team_Letters(self, tm_num):
        match = True
        long_name = self.All_Week_Teams[tm_num].lower()
        long_name = long_name.split(' ')
        del long_name[-1] #Only the city name
        long_letters = []
        for i in range (0, len(long_name)):
            long_letters += list(long_name[i])
        print(long_letters)
        print(self.Short_name.lower())
        for letter in self.Short_name.lower():
            if letter not in long_letters:
                match = False
        return match

    def Make_Naming_Reference(self):
        print(self.All_Season_Teams)
        print(len(self.All_Season_Teams))
        print(self.All_Week_Teams)
        print(len(self.All_Week_Teams))
        # while len(self.All_)
        #Make sure for each short name the long name has all the letters in it and if not swap the long name with name before or after if they have all the same letters
        for tm_num in range(0, len(self.All_Season_Teams)): #Short Names
            self.Short_name = self.All_Season_Teams[tm_num].lower()
            # for tm_name in range(0, len(self.All_Week_Teams)): #All long names
            match = self.Check_Team_Letters(tm_num)
            if match == False: #All Letters were same so action needed
                print(self.All_Season_Teams[tm_num])
                time.sleep(1)
                for pos in [-2, -1, 1, 2]:
                    if tm_num+pos < len(self.All_Week_Teams):
                        match = self.Check_Team_Letters(tm_num+pos)
                        if match == True:
                            #Swap list positions
                            print("Swap")
                            self.All_Week_Teams[tm_num], self.All_Week_Teams[tm_num+pos] = self.All_Week_Teams[tm_num+pos], self.All_Week_Teams[tm_num]
                            break
                    
        week_teams_df = pd.DataFrame()
        week_teams_df['Short_Names'] = self.All_Season_Teams
        week_teams_df['Long_Names'] = self.All_Week_Teams
        week_teams_df.to_csv(f'{self.season_path}/Names.csv', index=False)

    def Get_Week_Schedule(self):
        URL = f'https://www.pro-football-reference.com/years/{self.season}/games.htm'
        html = requests.get(URL).content
        df_list = pd.read_html(html, header=0, index_col=None)
        self.schedule_df = df_list[0]
        self.schedule_headers = list(self.schedule_df)
        print(self.schedule_headers)
        #Make the Week # the index
        self.schedule_df.set_index(keys=['Week'], drop=False,inplace=True)
        # now we can perform a lookup on a 'view' of the dataframe
        self.Week_Sched_DF = self.schedule_df.loc[self.schedule_df.Week==str(self.week)]
        # self.Change_Team_Names()
        self.Get_Team_Names()
        print(self.Week_Sched_DF)
        # return self.All_Week_Teams
    
    def Setup_This_Week_Data(self):
        #Get This Week DF and Team List for schedule
        self.Get_Week_Schedule()
        if self.week == self.min_week:
            self.Get_Team_Names()
            self.Make_Naming_Reference() #- Not same alphabetical order
        self.Change_Team_Names()
        self.Get_Team_Names()
        return self.All_Week_Teams, self.Week_Sched_DF
        
class This_Week_Stats():
    def __init__(self, last_week_teams, Week_DF, Week_Sched_DF):
        self.time = time
        self.last_week_teams = last_week_teams
        self.Week_DF = Week_DF
        self.Week_Sched_DF = Week_Sched_DF

    def Get_Data(self):
        self.this_week_teams = self.Week_DF["Team"].tolist()
        Opps = ["" for i in range(len(self.this_week_teams))]
        Home_Team = ["" for i in range(len(self.this_week_teams))]
        Short_Week = ["" for i in range(len(self.this_week_teams))]
        Off_Bye = ["" for i in range(len(self.this_week_teams))]
        cols = ["Winner/tie", "Loser/tie"]
        for tm in range(0, len(self.this_week_teams)):
            team = self.this_week_teams[tm]
            for col in range(0, len(cols)):
                Col_Teams = self.Week_Sched_DF[cols[col]].tolist()    
                Opp_Teams = self.Week_Sched_DF[cols[(col*-1)+1]].tolist()
                Game_Days = self.Week_Sched_DF["Day"].tolist() 
                Home_Away = self.Week_Sched_DF["Unnamed: 5"].tolist() 
                for name in range(0, len(Col_Teams)):
                    if team == Col_Teams[name]: #Team name of this week is in the DF col
                        opponent = Opp_Teams[name]
                        print(team)
                        print(opponent)
                        Opps[tm] = opponent
                        gameday = Game_Days[name]
                        if gameday == "Sun" or gameday == "Mon": #Not short week
                            Short_Week[tm] = 0
                        else:
                            Short_Week[tm] = 1
                        location = Home_Away[name]
                        if location == "@" and cols[col] == "Loser/tie": #First team is @
                            Home_Team[tm] = 1
                        elif location != "@" and cols[col] == "Winner/tie": #Second team is not @
                            Home_Team[tm] = 1
                        else:
                            Home_Team[tm] = 0
                        if team in self.last_week_teams: #Plaued last week
                            Off_Bye[tm] = 0
                        else:
                            Off_Bye[tm] = 1

        self.Week_DF["Opponent"] = Opps
        self.Week_DF["Home Team"] = Home_Team
        self.Week_DF["Short Week"] = Short_Week
        self.Week_DF["Off Bye Week"] = Off_Bye
        print(self.Week_DF)
                

    def Get_Info(self):
        #Add Teams, Week, Home, Teams, Opponents, Location, Short Week to df
        self.Get_Data()
        return self.Week_DF
