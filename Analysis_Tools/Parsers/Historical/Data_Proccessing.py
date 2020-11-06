import pandas as pd
import time
import os


class PROCCESS_DATA():
    def __init__(self, week, season, df, week_path, all_cleaned_weeks_list, all_weeks_df_list):
        self.time = time
        self.df = df
        self.season = season
        self.week = week        
        self.week_path = week_path
        self.All_Cleaned_Weeks_DF_List = all_cleaned_weeks_list
        self.All_Weeks_DF_List = all_weeks_df_list

        #CleanUp Stuff
        self.Cleanup_Units_Stats = ["Time of Poss", "3rd Down %_For", "3rd Down % Against"]
        self.Per_Game_Stats = ["Point Differential", "Sacks Against", "Sacks For", "Passing Yards For", "Passing Yards Against", "Rushing Yards For", "Rushing Yards Against", "Time of Poss", "Penalty Yards Committed", "Penalty Yards Received", "Total Yard Margin", "Turnover Margin"]
        self.Margin_Stats = ["Penalty Yards"]

        #Ranking Stuff
        self.UnRanked_Stats = ["Team", "Week", "Year", "Games Played", "Spread", "Over/Under", "Game Scoring Margin", "Spread to Result Difference", "Spread Result Class", "Home Team", "Short Week", "Off Bye Week", "Opponent"]
        self.Discount_Rank_Stats = ["Sacks For Per Game", "Sacks Against Per Game", "Yards per Rush For", "Yards per Rush Against", "Rushing Yards For Per Game", "Rushing Yards Against Per Game", "Passer Rating For", "Passer Rating Against", "Passing Yards For Per Game", "Passing Yards Against Per Game"]
        self.Opp_Discount_Rank_Stats = ["Sacks Against Per Game", "Sacks For Per Game", "Yards per Rush Against", "Yards per Rush For", "Passer Rating Against", "Passer Rating For"]
        self.Combined_Stats = [["Sacks For Per Game"], ["Sacks Against Per Game"], ["Yards per Rush For", "Rushing Yards For Per Game"], ["Yards per Rush Against", "Rushing Yards Against Per Game"], ["Passer Rating For", "Passing Yards For Per Game"], ["Passer Rating Against", "Passing Yards Against Per Game"]]
        self.Combined_New_Names = ["Line For", "Line Against", "Rushing For", "Rushing Against", "Passing For", "Passing Against"]

        #Opponent Proccessing Stuff
        # self.Drop_Columns = ["Year", "Week", "Games Played", "Opp Year", "Opp Games Played", "Opp Team", "Opp Week", "Opp Game Scoring Margin", "Opp Spread to Result Difference", "Opp Opponent", "Opp Spread Result Class", "Opp Spread", "Opp Over/Under"]
        self.Drop_Columns = [f'{name} Rating' for name in self.Discount_Rank_Stats] + ["Games Played", "Opp Year", "Opp Games Played", "Opp Team", "Opp Week", "Opp Game Scoring Margin", "Opp Spread to Result Difference", "Opp Opponent", "Opp Spread Result Class", "Opp Spread", "Opp Over/Under"]
        self.Game_Location_Columns = ["Home Team", "Short Week", "Off Bye Week"]
        self.Non_Diff_Columns = ["Team", "Year", "Week", "Opponent", "Spread", "Over/Under", "Game Scoring Margin", "Spread to Result Difference", "Spread Result Class"]
        self.Team_Diff_Columns = ["Home Team", "Off Bye Week"] + ["W-L%", "Point Differential Per Game", 'Discounted Passer Rating For', 'Discounted Passer Rating Against', "Discounted Sacks Against Per Game", "Discounted Sacks For Per Game", "Discounted Yards per Rush For", "Discounted Yards per Rush Against", "Time of Poss Per Game", "Yards For per Play", "Yards Against per Play", "Penalty Yards Margin Per Game", "3rd Down %_For", "3rd Down % Against", "Total Yard Margin Per Game", "Turnover Margin Per Game"]
        self.Opp_Diff_Columns = ["Home Team", "Off Bye Week"] + ["W-L%", "Point Differential Per Game", 'Discounted Passer Rating Against', 'Discounted Passer Rating For', "Discounted Sacks For Per Game", "Discounted Sacks Against Per Game", "Discounted Yards per Rush Against", "Discounted Yards per Rush For", "Time of Poss Per Game", "Yards Against per Play", "Yards For per Play", "Penalty Yards Margin Per Game", "3rd Down % Against", "3rd Down %_For", "Total Yard Margin Per Game", "Turnover Margin Per Game"]

    def Save_DF(self, df, path):
        df.to_csv(path, index=False)

    def Do_Stuff(self):
        raw_week_excel_name = f'{self.week_path}/Week {self.week} Raw Data.csv'
        self.Save_DF(self.df, raw_week_excel_name)
        # self.Raw_DF.to_csv(raw_week_excel_name, index=False)
        
        #Initial Clean Up Proccessing
        Clean_Up = Cleanup_Proccessing(self.week, self.season, self.df, self.week_path, self.All_Cleaned_Weeks_DF_List, self.All_Weeks_DF_List)
        CleanedUp_DF, Cleaned_Season_DF = Clean_Up.Do_Stuff()
        cleaned_data_path = f'{self.week_path}/Week {self.week} Cleaned Data.csv'
        self.Save_DF(CleanedUp_DF, cleaned_data_path)
        # CleanedUp_DF.to_csv(cleaned_data_path, index=False)

        #Turn Stats into Rankings
        Ranking = Make_Rankings(self.week, self.season, Cleaned_Season_DF, self.week_path, self.All_Cleaned_Weeks_DF_List, self.All_Weeks_DF_List)
        Ranked_DF = Ranking.Do_Stuff()
        discounted_data_path = f'{self.week_path}/Week {self.week} Discounted Ranked Data.csv'
        self.Save_DF(Ranked_DF, discounted_data_path)
        # Ranked_DF.to_csv(discounted_data_path, index=False)

        #Add Opponent Stats
        Combine = Proccess_Opponents(self.week, self.season, Ranked_DF, self.week_path, self.All_Cleaned_Weeks_DF_List, self.All_Weeks_DF_List)
        Diff_DF = Combine.Do_Stuff()
        diff_data_path = f'{self.week_path}/Week {self.week} Advantage Data.csv'
        self.Save_DF(Diff_DF, diff_data_path)
        # Diff_DF.to_csv(diff_data_path, index=False)

        #Make for Model
        # self.Normalize_Data()
        # self.Normalized_DF.to_csv(self.Normalized_Model_Data_Path, index=False)

        self.All_Weeks_DF_List.append(Diff_DF) #Put each weeks df in a list - should get 17 dfs by end of season

        return self.All_Weeks_DF_List, self.All_Cleaned_Weeks_DF_List

class Cleanup_Proccessing(PROCCESS_DATA):

    def find_Opponenet_Row(self, team_row):
        sorted_team_list = self.Week_DF["Team"].tolist()
        opponent_list = self.Week_DF["Opponent"].tolist()
        week_list = self.Week_DF["Week"].tolist()
        team_name = sorted_team_list[team_row]
        week_num = week_list[team_row]
        opponent = opponent_list[team_row]
        for game in range(0,len(self.Week_DF)):
            if opponent_list[game] == team_name: #If opponent = team in question
                if week_list[game] == week_num and sorted_team_list[game] == opponent: #If same week and team = team in questions opponenet
                    break
        return game

    def Remove_No_Data_Teams(self): #Do at end so don't mess up matchup with opponent - could be if game == 1 (don't want first game)
        #Just remove rows of bye week teams so don't get into the model with no result
        #Get column num of "Opponenets"
        Bye_Week_Teams = []
        No_Games_Teams = []
        for col in range(0, len(self.df_headers)):
            if self.df_headers[col] == "Opponent":
                Opp_Col = col
                print(Opp_Col)
        for val in range(0,len(self.Week_DF["Opponent"].tolist())):
            if self.Week_DF.iloc[val,Opp_Col] == "" or pd.isnull(self.Week_DF.iloc[val,Opp_Col]):
            # if pd.isnull(self.Week_DF).any():
            # if pd.isnull(self.Week_DF.iloc[val,Opp_Col]):
                Bye_Week_Teams.append(val)
            elif self.Week_DF.iloc[val,4] == "" or pd.isnull(self.Week_DF.iloc[val,4]): #No games played
                No_Games_Teams.append(val) #Team with no games
                #Find team's opponent
                row = self.find_Opponenet_Row(val)
                No_Games_Teams.append(row)
                print(No_Games_Teams)
        No_Data_Teams = Bye_Week_Teams + No_Games_Teams
        print(No_Data_Teams)
        print(self.Week_DF.tail())
        time.sleep(2)
        for row in No_Data_Teams:
            self.Week_DF.drop([row], inplace=True)
        print(self.Week_DF.tail())

    def Make_Stats_Margin(self): #Do 2nd
        print("Working On It")
        Margin = []
        last_header_pos = 10000
        for stat in self.Margin_Stats:
            Stats = []
            for header_pos in range(0, len(self.df_headers)):
                header = self.df_headers[header_pos]
                if stat in header:
                    Stats.append(self.Week_DF[f'{header} Per Game'].tolist())
                    del self.Week_DF[f'{header} Per Game']
                    if header_pos < last_header_pos:
                        first_header_pos = header_pos
                        last_header_pos = header_pos
                    print(Stats)
            for gm in range(0, len(Stats[0])):
                Margin.append(float(Stats[0][gm]) - float(Stats[1][gm]))
            print(Margin)
        # self.Week_DF.rename(columns={'Penalty Yards Committed Per Game':'Penalty Yards Margin Per Game'})
        self.Week_DF.insert(first_header_pos, 'Penalty Yards Margin Per Game', Margin)
        # del self.Week_DF["Penalty Yards Received Per Game"]

    def Make_Stats_Per_Game(self): #Do 2nd
        Games = self.Week_DF["Games Played"].tolist()
        for game in range(0, len(Games)):
            for stat in self.Per_Game_Stats:
                try:
                    self.Week_DF.iloc[game, self.Week_DF.columns.get_loc(stat)] = round(float(self.Week_DF[stat][game])/int(Games[game]),2)
                    #self.Week_DF[stat][game] = float(self.Week_DF[stat][game])/int(Games[game])
                except ValueError:
                    pass
        for stat_type in range(0, len(self.Per_Game_Stats)):
            self.Week_DF.rename(columns={f'{self.Per_Game_Stats[stat_type]}': f'{self.Per_Game_Stats[stat_type]} Per Game'}, inplace = True)
            print(self.Week_DF[f'{self.Per_Game_Stats[stat_type]} Per Game'].head())
    
    def Cleanup_Units(self): #Do 1st
        for unit in self.Cleanup_Units_Stats:
            data = self.Week_DF[unit].tolist()
            print(unit)
            if "Time" in unit:
                split_symbol = ':'
                for point in range(0,len(data)):
                    point_parts = str(data[point]).split(split_symbol)
                    #print(point_parts)
                    try:
                        mins = int(str.strip(point_parts[0]))*60 + int(point_parts[1]) #Don't care about seconds
                    except:
                        mins = "" #No game that week
                    data[point] = mins
            elif "%" in unit:
                split_symbol = '%'
                for point in range(0,len(data)):
                    point_parts = str(data[point]).split(split_symbol)
                    #print(point_parts)
                    if len(point_parts) > 1:
                        try:
                            percent = point_parts[0]
                        except IndexError:
                            percent = 0 #No game that week
                    else:
                        percent = ""
                    data[point] = percent
            #print(data)
            self.Week_DF[unit] = data

    def Make_To_Covered_Not(self):
        temp_list = []
        five_class_src = self.Week_DF["Spread Result Class"].tolist()
        print(five_class_src)
        spreads = self.Week_DF["Spread"].tolist()
        for val in range(0,len(five_class_src)):
            print('val')
            print(five_class_src[val])
            print('val')
            src = float(five_class_src[val])
            spread = float(spreads[val])
            if src == 0: #Must have been a push
                temp_list.append(0)
            elif spread < 0 and src < 0: #Upset so favorite team didn't cocver
                temp_list.append(-1)
            elif spread < 0 and src > 0: #Favorited team did cover
                temp_list.append(1)
            elif spread > 0 and src < 0: #Underdog team was part of upset so underdog team did cocver
                temp_list.append(1)
            elif spread > 0 and src > 0: #Underdog team lost to spread so didn't cocver
                temp_list.append(-1)
            elif spread == 0 and src > 0: #Even spread and team won thus covered
                temp_list.append(1)
            elif spread == 0 and src < 0: #Even spread and team lost thus didn't cover
                temp_list.append(-1)
            else:
                temp_list.append(0) #Backup incase weird thing
        self.Week_DF["Spread Result Class"] = temp_list

    def Combine_Weeks(self):
        #Adding Week DF to list, combine all weeks into a Season DF and save it
        self.All_Cleaned_Weeks_DF_List.append(self.Week_DF) #Put each weeks df in a list - should get 17 dfs by end of season
        self.Cleaned_Season_DF = pd.concat(self.All_Cleaned_Weeks_DF_List) #Concat the list of week dfs into a season df after each season
        print(self.Cleaned_Season_DF.head())

    def Do_Stuff(self):
        #Get Dataframe
        self.Week_DF = self.df
        print(self.Week_DF)
        print('Week DF')
        self.df_headers = list(self.Week_DF)
        #Make pretty
        self.Cleanup_Units()
        self.Make_Stats_Per_Game()
        self.Remove_No_Data_Teams()
        #Make usable by model
        self.Make_Stats_Margin()
        self.Make_To_Covered_Not()
        self.Combine_Weeks()
        # self.Proccessed_DF = self.Week_DF
        # self.Proccessed_DF.to_csv(self.Cleanedup_Data_Path, index=False)
        return self.Week_DF, self.Cleaned_Season_DF


class Make_Rankings(PROCCESS_DATA):

    def Take_Unranked_Cols(self):
        for col in self.UnRanked_Stats:
            self.Ranked_DF[col] = self.Week_DF[col].tolist()

    def Add_DFs(self, new_df):
        for col in list(new_df):
            self.Ranked_DF[col] = new_df[col].tolist()

    def Get_Week_Data(self, week):
        Week_DF = self.Season_DF.loc[self.Season_DF['Week'] == week]
        return Week_DF

    def Normalize_Data(self, data, scale_range):
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=scale_range)
        print(data)
        data_scaled = min_max_scaler.fit_transform(data)
        print(data_scaled)
        return data_scaled

    def Make_Rated(self, data, columns):
        ranked_data = pd.DataFrame()
        ranked_data["Team"] = data["Team"].tolist()
        for col in columns:
            num_data=[]
            ranking = [1]
            rating = []
            ties = 1
            col_data = data[col].tolist()
            for d in col_data:
                num_data.append(float(d))
            ranked_data[col] = num_data
            if "Against" in col or "Turnover" in col: #Lowest Ranked 1st
                ascending_logic = True
                scale_range = (-100, 0)
            else: #Highest ranked 1st
                ascending_logic = False
                scale_range = (0, 100)
            normalize_data = pd.DataFrame()
            normalize_data[col] = data[col].tolist()
            rated_data = self.Normalize_Data(normalize_data, scale_range)
            rated_col = []
            for val in range (0, len(rated_data)):
                rated_col.append(abs(round(rated_data[val][0],0)))
            print(rated_col)

            # ranked_data = ranked_data.sort_values(by=[col], ascending=ascending_logic)
            # ranked_col = ranked_data[col].tolist()
            # for val in range(1, len(ranked_col)):
            #     if ranked_col[val] == ranked_col[val-1]:
            #         ranking.append(ranking[val-1])
            #         ties+=1
            #     else:
            #         ranking.append(ranking[val-1]+ties)
            #         ties = 1
            ranked_data[f'{col} Rating'] = rated_col# list(range(1, len(col_data)+1))
            ranked_data = ranked_data.sort_values(by=["Team"], ascending=True)#, inplace=True)
            del ranked_data[col]
            print(ranked_data)
        del ranked_data["Team"]
        return ranked_data
    
    def Make_Ranked(self, data, columns):
        ranked_data = pd.DataFrame()
        ranked_data["Team"] = data["Team"].tolist()
        for col in columns:
            num_data=[]
            ranking = [1]
            ties = 1
            col_data = data[col].tolist()
            for d in col_data:
                num_data.append(float(d))
            ranked_data[col] = num_data
            if "Against" in col or "Turnover" in col: #Lowest Ranked 1st
                ascending_logic = True
            else: #Highest ranked 1st
                ascending_logic = False
            ranked_data = ranked_data.sort_values(by=[col], ascending=ascending_logic)
            ranked_col = ranked_data[col].tolist()
            for val in range(1, len(ranked_col)):
                if ranked_col[val] == ranked_col[val-1]:
                    ranking.append(ranking[val-1])
                    ties+=1
                else:
                    ranking.append(ranking[val-1]+ties)
                    ties = 1
            ranked_data[f'{col} Rank'] = ranking# list(range(1, len(col_data)+1))
            ranked_data = ranked_data.sort_values(by=["Team"], ascending=True)#, inplace=True)
            del ranked_data[col]
            print(ranked_data)
        del ranked_data["Team"]
        return ranked_data

    def Extract_Stats(self, Discount_Stats, DF, pos):
        #Add each stat value for each week
        for stat in range(0, len(Discount_Stats)):
            print(Discount_Stats[stat])
            self.Discounted_Totals[stat+pos] = (self.Discounted_Totals[stat+pos] + float(DF[Discount_Stats[stat]].tolist()[0]))
            print(self.Discounted_Totals)

    def Discount_Rankings(self):
        #For each week up to current need to get opponent and ranking as of that week from season DF and take average ranking
        self.Discounted_DF = pd.DataFrame()
        All_Discounted_Totals = [[] for i in range(len(self.Discount_Rank_Stats))]
        week_teams = self.Ranked_DF["Team"].tolist() #All teams this season
        self.Discounted_DF["Team"] = week_teams
        print(len(week_teams))
        for team_num in range(0, len(week_teams)):
            team = week_teams[team_num]
            print(team)
            self.Discounted_Totals = [0 for i in range(len(self.Discount_Rank_Stats)*2)] #*2 for Team + Opponent
            #Get Teams' this week data:
            week_df = self.Get_Week_Data(self.week)
            Team_DF = week_df.loc[week_df['Team'] == team]
            print(f'Team: {team}')
            print(Team_DF)
            #Add each stat value for each week
            self.Extract_Stats(self.Discount_Rank_Stats, Team_DF, 0)
            #Get Opp prev week data:
            week_count = 0
            for week_num in range(self.min_week, self.week):
                print(week_num)
                week_df = self.Get_Week_Data(week_num) #Use this week's data unranked
                this_week_df = self.Get_Week_Data(self.week-1) #Use current week's data unranked
                last_week_df = self.Get_Week_Data(self.week-2) #Use last week's data unranked
                Team_DF = week_df.loc[week_df['Team'] == team]
                print(Team_DF)
                if len(Team_DF["Team"].tolist()) > 0:
                    opponent = Team_DF["Opponent"].tolist()[0]
                    # Opp_DF = week_df.loc[week_df['Team'] == opponent] #Use this week's data unranked
                    Opp_DF = this_week_df.loc[this_week_df['Team'] == opponent] #Use current week's data unranked
                    print(f'Opponent: {opponent}')
                    print(Opp_DF)
                    #Add each stat value for each week
                    try:
                        self.Extract_Stats(self.Opp_Discount_Rank_Stats, Opp_DF, len(self.Discount_Rank_Stats))
                    except:
                        Opp_DF = last_week_df.loc[last_week_df['Team'] == opponent] #Use last week's data unranked
                        self.Extract_Stats(self.Opp_Discount_Rank_Stats, Opp_DF, len(self.Discount_Rank_Stats))
                    week_count+=1
                else: #No game that week
                    pass
                # time.sleep(1)
            #Add Team and Opponents Stats together
            for stat in range(0, len(self.Discount_Rank_Stats)):
                if week_count == 0:
                    week_count = 1
                avg_opp_stat = self.Discounted_Totals[stat+len(self.Discount_Rank_Stats)]/week_count
                All_Discounted_Totals[stat].append(2*self.Discounted_Totals[stat]-avg_opp_stat)
            # print(All_Discounted_Totals)
            # time.sleep(2)
        #Make DF of the stats
        for stat in range(0, len(self.Discount_Rank_Stats)):
            self.Discounted_DF[f'Discounted {self.Discount_Rank_Stats[stat]}'] = All_Discounted_Totals[stat]     
        self.Discounted_Columns = [f'Discounted {name}' for name in self.Discount_Rank_Stats]    
        print(self.Discounted_DF)        
        print(self.Discounted_Columns)   

    def Week2_Discount(self):
        self.Discounted_DF = pd.DataFrame()
        week_teams = self.Ranked_DF["Team"].tolist() #All teams this season
        self.Discounted_DF["Team"] = week_teams
        for stat in self.Discount_Rank_Stats:
            un_discounted = self.Week_DF[f'{stat}'].tolist()
            self.Discounted_DF[f'Discounted {stat}'] = un_discounted   
        self.Discounted_Columns = [f'Discounted {name}' for name in self.Discount_Rank_Stats]    
        print(self.Discounted_DF)        
        print(self.Discounted_Columns)   

    def Combine_Stats(self):
        combined_df = pd.DataFrame()
        combined_df["Team"] = self.Week_DF["Team"].tolist()
        for col in range(0, len(self.Combined_Stats)):
            for stat in self.Combined_Stats[col]:
                this_stat = self.Week_DF[stat].tolist()
                total_ranking = [0 for i in range(len(this_stat))]
                for tm in range(0, len(this_stat)):
                    total_ranking[tm] += float(this_stat[tm])
            combined_df[self.Combined_New_Names[col]] = total_ranking
        return combined_df

    def Remove_Columns(self): #Do last
        for col_name in self.Discount_Rank_Stats:
            self.Ranked_DF.drop(f'{col_name} Rank', axis=1, inplace=True)

    def Do_Stuff(self):
        #Get Dataframes
        self.min_week = 2
        self.Season_DF = self.df
        self.Ranked_DF = pd.DataFrame()
        #Get Stats to Rank
        self.df_headers = list(self.Season_DF)
        self.Rank_Stats = []
        for header in range(0,len(self.df_headers)):
            if self.df_headers[header] not in self.UnRanked_Stats and "Adv" not in self.df_headers[header] and "Rank" not in self.df_headers[header]:
                self.Rank_Stats.append(self.df_headers[header])
        print(self.Rank_Stats)
        # time.sleep(20)
        #Get Current Week
        self.Week_DF = self.Get_Week_Data(self.week) #Get this week's data
        print(self.Week_DF)
        print('HERE')
        self.Take_Unranked_Cols() #Put Non-Ranked Stats into Dataframe
        ranked_df = self.Make_Rated(self.Week_DF, self.Rank_Stats) #Make Ranked Stats Ranked
        self.Add_DFs(ranked_df) #Add the ranked columns to the Ranked DF
        print(self.Ranked_DF.head())
        combined_df = self.Combine_Stats()
        combined_rated_df = self.Make_Rated(combined_df, self.Combined_New_Names)
        self.Add_DFs(combined_rated_df) #Add the ranked columns to the Ranked DF
        print(self.Ranked_DF.head())
        #Discount The Rankings
        # if self.week > self.min_week:
        #     self.Discount_Rankings()
        # else:
        #     self.Week2_Discount()
        # ranked_discount_df = self.Make_Ranked(self.Discounted_DF, self.Discounted_Columns) #Make Discounted Stats Ranked
        # self.Add_DFs(ranked_discount_df) #Add the ranked columns to the Ranked DF    
        print(self.Ranked_DF.head())
        ranked_excel_name = f'{self.week_path}/Week {self.week} Rated Data.csv'
        self.Ranked_DF.to_csv(ranked_excel_name, index=False)

        # self.Remove_Columns()
        # self.Ranked_DF.to_csv(self.Discounted_Data_Path, index=False)
        return self.Ranked_DF

class Proccess_Opponents(PROCCESS_DATA):

    def find_Opponenet_Row(self, team_row):
        sorted_team_list = self.Week_DF["Team"].tolist()
        opponent_list = self.Week_DF["Opponent"].tolist()
        week_list = self.Week_DF["Week"].tolist()
        team_name = sorted_team_list[team_row]
        week_num = week_list[team_row]
        opponent = opponent_list[team_row]
        for game in range(0,len(self.Week_DF)):
            if opponent_list[game] == team_name: #If opponent = tam in question
                if week_list[game] == week_num and sorted_team_list[game] == opponent: #If same week and team = team in questions opponenet
                    break
        return game

    def Append_Opponent_Stats(self): #Make one team "Team 1 point diff" and other "Team 2 point diff"
        #For each row, need to find the corresponding opponent row, take the values and append to end of first row with "Opp" before each header
        #Can store each row as a DF and then concat them sideways and then concat each new row to form a full DF
        #OR can store each element from the row into a list, then each from matching row to a list, add the lists and then store as a df/array
        num_rows = len(self.Week_DF)
        #Get the headers of DF, make combined list of these plus opponenet ones
        df_headers = list(self.Week_DF)
        print(df_headers)
        opp_headers = []
        for header in df_headers:
            print(header)
            opp_headers.append(f'Opp {header}')
            Model_DF_Headers = df_headers + opp_headers
        print(Model_DF_Headers)
        num_columns = len(df_headers)

        Temp_DF_List = []
        #Put entire DF into a list
        DF_As_List = [[] for i in range(num_columns)]
        #print(self.Proccessed_DF[1])
        for col in range(0, num_columns):
            DF_As_List[col] = self.Week_DF[df_headers[col]].tolist()
        for team_row in range(num_rows):
            Matchup_Data = [[] for i in range(2*num_columns)]
            #team = self.sorted_team_list[team_row]
            opp_row = self.find_Opponenet_Row(team_row)
            #Put data from team and opponent into a list
            for col in range(0, num_columns):
                Matchup_Data[col] = [(DF_As_List[col][team_row])]
            for new_col in range(0, num_columns):
                Matchup_Data[new_col+num_columns] = [(DF_As_List[new_col][opp_row])]
            #Store list into a df and store df into a list
            print(Matchup_Data)
            temp_df = pd.DataFrame(Matchup_Data)#, columns=Model_DF_Headers)
            temp_df = temp_df.transpose()
            temp_df.columns = Model_DF_Headers
            print(temp_df)
            Temp_DF_List.append(temp_df)
        #Put each row df into final Master DF
        self.Combined_DF = pd.concat(Temp_DF_List)

    def Differentiate_Columns(self):
        self.Diff_DF = pd.DataFrame()
        #Add the non-diff columns first
        for col in range(0, len(self.Non_Diff_Columns)):
            temp_list = self.Combined_DF[self.Non_Diff_Columns[col]].tolist()
            self.Diff_DF[self.Non_Diff_Columns[col]] = temp_list
        for col in range(0, len(self.Team_Diff_Columns)):
            if self.Team_Diff_Columns[col] in self.Game_Location_Columns:
                extra = ""
                direction = -1 #Makes it Team - Opp
            else:
                extra = " Rank"
                direction = 1
            team_data = self.Combined_DF[f'{self.Team_Diff_Columns[col]}{extra}'].tolist()
            opp_data = self.Combined_DF[f'Opp {self.Opp_Diff_Columns[col]}{extra}'].tolist()
            temp_list = []
            for game in range(0,len(team_data)):
                team_val = float(team_data[game])
                opp_val = float(opp_data[game])
                temp_list.append(direction*(opp_val-team_val))
            self.Diff_DF[f'{self.Team_Diff_Columns[col]} Adv'] = temp_list
            print(self.Diff_DF.head())


    def Remove_Columns(self): #Do last
        print(self.Drop_Columns)
        for col_name in self.Drop_Columns:
            self.Combined_DF.drop(col_name, axis=1, inplace=True)

    def Do_Stuff(self):
        #Get Dataframe
        self.Week_DF = self.df
        self.df_headers = list(self.Week_DF)
        #Do Proccessing
        self.Append_Opponent_Stats()
        self.Remove_Columns()
        opponent_excel_name = f'{self.week_path}/Week {self.week} Opponent Combined Data.csv'
        self.Combined_DF.to_csv(opponent_excel_name, index=False)
        return self.Combined_DF
        # self.Differentiate_Columns()
        # self.Diff_DF.to_csv(self.Diff_Data_Path, index=False)
        # return self.Diff_DF


class Modelize(PROCCESS_DATA):
    
    def Normalize_Data(self):
        self.Model_DF = self.Model_DF[[c for c in self.Model_DF if c not in ['Spread Result Class']] + ['Spread Result Class']]
        print(self.Model_DF.head())
        print(self.Model_DF.shape)
        df_headers = list(self.Model_DF)

        self.Training_Data = self.Model_DF[[c for c in self.Model_DF if c not in ['Spread Result Class']]]
        df_headers = list(self.Training_Data)
        self.Training_Data = self.Training_Data.values
        print(self.Training_Data)
        print(self.Training_Data.shape)

        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        print(self.Training_Data)
        self.Training_Data_Scaled = min_max_scaler.fit_transform(self.Training_Data)
        print(self.Training_Data_Scaled)
        self.Normalized_DF = pd.DataFrame(self.Training_Data_Scaled, columns=df_headers)

    # def Remove_Columns(self):
    #     for col_name in self.Model_Drop_Cols:
    #         self.Model_DF.drop(f'{col_name}', axis=1, inplace=True)

# Proccessing_Data = NFL_DATA_PROCCESSING()
# Proccessing_Data.Do_Stuff()