import pandas as pd
import numpy as np
import time
import os
import re
import requests
from statistics import mean, stdev
import plotly
import plotly.graph_objs as go
'''
Will rely on:
1.The Team and Opponents' difference in DVOA 
2. Their trends vs opponents' ratings in Passing For/Aginast, Line For/Against, Rushing For/Against
We will come up with an expected game outcome (EGO) based on DVOA ratings and then bias this prediction with the Matchup Trend Correlation Data

1. For EGO based on DVOA - DVOA-EGO:
    We will look at all the game scoring margins over the past 20 seasons and the teams' difference in DVOA. 
    We will group all the EGO difference in groups (100-60% - Pats vs Fins in 2019 for ex, 60%-40%, 40%-30%, 30-25%, 25-20% etc...)
    We will take the average game scoring margin for games in each of the groups
    This will give us a table of DVOA-Diff vs Average Game Margin
    Now for every game we want to predict, we can look up te DVOA difference and get an expcted game scoring margin
    This is our DVOA-EGO. This may be enough to go on to predict game results, compare to spread, pick against the spread and pick correctly > 55% of times

    Will Run Data Collection for up to newest week
    Will Run DVOA analysis which will take All Season Stats and:
        Make a seperate csv for each 2019 week DVOA Raw Data - will take user input for what week is the latest
        Make a csv for each 2019 week of team, spread, ego, ego-spread diff, right/wrong - will take the saved week DVOA data if there's a matching one, else will take the earliest saved
        Need to make a csv for all Data that has team, ego, and then the 6 correlation data and use to predict games. For 2019 do the same thing where we take the earliest week we can

2. Adjust EGO based on Matchup Correlation
    For Matchup trends, will look at trend between teams' Opponents Passing/Line/Rush and the result of the game relative to our DVOA-EGO.
        *Note - The Matchup trends can be used when there are trends to go on. In the case where there is little trend, the model of course will be going off only #1(DVOA-EGO)
    From this, we get the Game Scoring Margin to DVOA-EGO difference vs the opponents rating and the correlation of this relationship
    Then for the game at hand, we can fit a prediction of Game margin to DVOA-EGO difference for each of the Pass/Line/Rush stats
    So for each stat we'll have (for example):
        Opp {Stat} Based Game Margin-DVOA-EGO Diff Pred    Correlation  ...... X6 ...
                            -3.72                              0.65
    This is what I've already done but I used the Spread in place of the DVOA-EGO

Then our model will take the #1 DVOA-EGO and, for each stat, the adjustment/correlation and use them to assign a Total EGO
We will compare to the spread of the game and for each game since 2001 we will get our pick from this and see the % we are correct

Primary problem - our historical DVOA Data is only for end of season, so not accurate representation of teams at the given week of the season
Secondary Note - It is likely that we can only make predictions and train the model for Weeks > ~6, as before then all the in season DVOA and Matchup Trends are too unreliable
                -This may help with the primary problem, since by Week 6, the differnce in DVOA then vs the end of season could be ignored

'''

'''
Starting with the All Seasons Results Data
Split into season by season DF
For each season get the season DVOA DF
For every game of season, need to get teams and opponents DVOA from DVOA DF
    Then take differnce and store in Overall list to add to main df
    Store the actual game scoring margin in lists of each DVOA Diff range
Then add column of DVOA Diff to main df
Make CSV of cols DVOA Diff, avg game margin

'''
class DVOA_DATA():
    def __init__(self, latest_week, latest_season, DVOA_Type):#, home_val, season_both, total_path, Home_Proccessed_Path, Total_Proccessed_Path):
        # self.DVOA_Type = input(f'Enter The DVOA Type to use (DVOA or WDVOA): ')
        # self.DVOA_Type = "WDVOA"
        # self.DVOA_Type = "DVOA"
        self.DVOA_Type = DVOA_Type
        if self.DVOA_Type != "WDVOA" and self.DVOA_Type != "DVOA":
            print("BAD DVOA TYPE ENTERED")
        self.time = time
        # self.home_val = home_val
        # self.Home_Proccessed_Path = Home_Proccessed_Path
        self.project_path = 'E:/Project Stuff/Data Analysis Stuff/V5/NFL Data V1'
        self.Raw_Data_Path = f'E:/Project Stuff/Data Analysis Stuff/V5/NFL Data V1/Raw Data/All Seasons Results.csv'
        self.DVOA_Raw_Path = f'{self.project_path}/Raw Data/WDVOA DATA'
        # self.BySeason_DVOA_Stat_save_path = f'{self.project_path}/Team Scores Grouped By {self.DVOA_Type} Diff and Season.csv'
        # self.Training_DVOA_Stat_save_path = f'{self.project_path}/All Team Scores Grouped By {self.DVOA_Type} Diff.csv'
        self.Training_DVOA_Stat_save_path = f'{self.project_path}/Train_Test_Data'
        # self.DVOA_DF_Save_Path = f'{self.project_path}/Testing Seasons {self.DVOA_Type} Results.csv'
        self.DVOA_DF_Save_Path = f'{self.project_path}/Total Data/All Seasons {self.DVOA_Type} Picking Results.csv'
        self.Final_DVOA_DF_Save_Path = f'{self.project_path}/Total Data/All Seasons {self.DVOA_Type} Results.csv'

        self.Game_Locations = ["Away", "Home"]
        self.Possible_WeightedDVOA_Names = ["WEI.DVOA", "WEIGHTED DVOA", "WEIGHTEDDVOA", "WEIGHTEDVOA", "DAVE", "TOTAL DAVE", "TOTALDAVE", "TOTAL  DAVE", 'WEI.  DVOA']
        self.DVOA_Stat_Headers = ["Season", "DVOA Diff Range", "Count", "Avg Spread", "Avg Game Margin", "EGO", "Max", "Min", "Std Dev"]
        self.EGO_Stat_Headers = ["Team DVOA", "DVOA Diff", "EGO", "EGO to Spread Diff", "EGO to Score Diff", "EGO Pick", "Right Pick", "Wrong Pick", "SRD", "Spread", "Score Margin"]
        self.Stat_Cols = ["Line For Rating", "Line Against Rating", "Rushing For Rating", "Rushing Against Rating", "Passing For Rating", "Passing Against Rating"]
        self.Stat_Cols = self.Stat_Cols + [f'Opp {name}' for name in self.Stat_Cols]
        # self.Final_Drop_Cols = ["EGO to Score Diff", "EGO Pick", "Right Pick", "Wrong Pick"]
        self.Final_Drop_Cols = ["EGO to Score Diff", "Wrong Pick"]

        self.min_week = 6
        self.max_ls_week = latest_week
        self.max_season = latest_season

        dvoa_ranges = [50, 40, 35, 30, 25, 20, 17.5, 15, 12.5, 10, 7.5, 5, 2.5, 0]
        neg_dvoa_ranges = [i*-1 for i in dvoa_ranges if i != 0]
        self.DVOA_Diff_Groups = dvoa_ranges + neg_dvoa_ranges + [100]
        self.DVOA_Diff_Groups.sort()
        print(self.DVOA_Diff_Groups)
        
        self.Train_Test_Seasons = [[2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019], []]
        # self.Train_Test_Seasons = [[2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]]

    def Initialize_Data(self):
        self.Scores_byDVOA = [[[] for i in range(len(self.DVOA_Diff_Groups))] for i in range(len(self.Game_Locations))]
        self.Spreads_byDVOA = [[[] for i in range(len(self.DVOA_Diff_Groups))] for i in range(len(self.Game_Locations))]

        self.DVOA_Diffs = [[] for i in range(len(self.Game_Locations))]
        self.Scores_byLoc = [[] for i in range(len(self.Game_Locations))]
        self.DVOA_Stats = [[] for i in range(len(self.DVOA_Stat_Headers))] #Year, Bin, Count, Average, Max, Min, Std Dev
        self.EGO_Data = [[] for i in range(len(self.EGO_Stat_Headers))]

        self.no_dvoa_weeks = []
        self.no_weighted_dfs = []

    def Setup_Map(self):
        #Need to take all the DVOA_Diffs and map them to an EGO via the total home or away map
        #Read the Map DF
        self.Map_DF = pd.read_csv(f'{self.Training_DVOA_Stat_save_path}/{self.Training_DVOA_Stat_save_name}')
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

    def Get_EGO_Data(self, game):
        Loc = self.Locs[game]
        teamDVOA = float(self.All_Team_DVOAs[game])
        DVOA_Diff = float(self.All_DVOA_Diffs[game])
        EGO = round(np.interp(DVOA_Diff, list(self.Maps[Loc].keys()), list(self.Maps[Loc].values())),2)
        # EGO = np.poly1d(np.polyfit(self.All_Training_DVOA_Diffs, self.All_Training_Game_Margins, 4))(DVOA_Diff)# (np.unique(self.prior_feature_data))
        Spread = float(self.Spreads[game])
        Score = float(self.Game_Margins[game])
        Spr_EGO_Diff = EGO+Spread
        Scr_EGO_Diff = Score-EGO
        Scr_Spr_Diff = Score+Spread
        if Spr_EGO_Diff < 0:
            Pick = -1
            if Scr_Spr_Diff < 0:
                Pick_Right = 1
                Pick_Wrong = -1
            elif Scr_Spr_Diff == 0:
                Pick_Right = 0
                Pick_Wrong = 0
            else:
                Pick_Wrong = 1
                Pick_Right = -1
        elif Spr_EGO_Diff == 0:
            Pick = 0
            Pick_Right = 0
            Pick_Wrong = 0
        else:
            Pick = 1
            if Scr_Spr_Diff > 0:
                Pick_Right = 1
                Pick_Wrong = -1
            elif Scr_Spr_Diff == 0:
                Pick_Right = 0
                Pick_Wrong = 0
            else:
                Pick_Wrong = 1
                Pick_Right = -1
        self.EGO_Data[0].append(teamDVOA)
        self.EGO_Data[1].append(DVOA_Diff)
        self.EGO_Data[2].append(EGO)
        self.EGO_Data[3].append(Spr_EGO_Diff)
        self.EGO_Data[4].append(Scr_EGO_Diff)
        self.EGO_Data[5].append(Pick)
        self.EGO_Data[6].append(Pick_Right)
        self.EGO_Data[7].append(Pick_Wrong)
        self.EGO_Data[8].append(Scr_Spr_Diff)
        self.EGO_Data[9].append(Spread)
        self.EGO_Data[10].append(Score)

    def Add_DVOA_Data(self):
        #Add DVOA Diff To main DF
        # print(len(self.All_DVOA_Diffs))
        # self.dataset[self.EGO_Stat_Headers[0]] = self.All_DVOA_Diffs
        # self.dataset.to_csv(self.Total_Proccessed_Path, index=False)

        #Make Map
        self.Setup_Map()

        #Get Required Data for Making New DF
        self.semi_dataset = self.dataset.loc[(self.dataset["Week"] >= self.min_week) & (self.dataset["Year"].isin(self.Seasons))] #Only
        self.Game_Margins = self.semi_dataset["Game Scoring Margin"].tolist()
        self.Spreads = self.semi_dataset["Spread"].tolist()
        self.Locs = self.semi_dataset["Home Team"].tolist()

        #Get Special DVOA Stats into new DF
        self.dvoa_dataset = pd.DataFrame()
        # self.ending_dataset = self.dataset.loc[(self.dataset["Year"].isin(self.Seasons))] #Only
        self.early_dataset = self.dataset.loc[(self.dataset["Week"] < self.min_week) & (self.dataset["Year"].isin(self.Seasons))] #Only
        self.ending_dataset = self.dataset.loc[(self.dataset["Week"] >= self.min_week) & (self.dataset["Year"].isin(self.Seasons))] #Only

        Data_Cols = ["Team", "Opponent", "Year", "Week", "Home Team", "Spread", "Game Scoring Margin"] + self.Stat_Cols + ["Spread to Result Difference", "Spread Result Class"]
        Data = []
        for col in range(len(Data_Cols)):
            Data.append(self.early_dataset[Data_Cols[col]].tolist() + self.ending_dataset[Data_Cols[col]].tolist())
            self.dvoa_dataset[Data_Cols[col]] = Data[col]
            DF_Length = len(Data[col])
        for game in range(0, len(self.Locs)):
            self.Get_EGO_Data(game)
        for stat in range(0, len(self.EGO_Stat_Headers)-2): #Not Spread and Scoring Margin
            print(len(self.EGO_Data[stat]))
            Stat_Length = len(self.EGO_Data[stat])
            self.dvoa_dataset[self.EGO_Stat_Headers[stat]] = [0 for i in range(DF_Length-Stat_Length)] + self.EGO_Data[stat]
        headers = Data_Cols[:-2] + self.EGO_Stat_Headers[:-3] + Data_Cols[-2:]
        self.dvoa_dataset = self.dvoa_dataset[headers]
       
        if self.iteration == 0:
            self.total_dvoa_dataset = self.dvoa_dataset
        else:
            self.total_dvoa_dataset = self.total_dvoa_dataset.append(self.dvoa_dataset, ignore_index = True) 
            print(self.total_dvoa_dataset.shape)
        self.total_dvoa_dataset = self.total_dvoa_dataset[headers]
        # self.total_dvoa_dataset.to_csv(self.DVOA_DF_Save_Path, index=False) #Not Working properly
        self.dvoa_dataset.to_csv(self.DVOA_DF_Save_Path, index=False)
        
        try:
            seasonName = self.Train_Test_Seasons[1][0]
        except IndexError:
            seasonName = "All Seasons"
        self.dvoa_dataset.to_csv(f'{self.project_path}/Train_Test_Data/{seasonName} {self.DVOA_Type} Picking Results.csv', index=False)

        #Evaluate Picks
        Right_Picks = self.EGO_Data[6]
        Wrong_Picks = self.EGO_Data[7]
        for pick in range(0,len(Right_Picks)):
            if Right_Picks[pick] == -1:
                Right_Picks[pick] = 0
            if Wrong_Picks[pick] == -1:
                Wrong_Picks[pick] = 0
        num_right_picks = sum(Right_Picks)
        num_wrong_picks = sum(Wrong_Picks)
        pick_acc = round(num_right_picks/(num_right_picks+num_wrong_picks),4)
        print(f'EGO Based Predictions gave a {pick_acc}% Betting Accuracy')
        # self.time.sleep(10)

        #Drop the prediction/result columns for the final DF to be used for matchup and model analysis
        #Add the 6 stat columns we're interested in now (for team + opp)
        for col in self.Final_Drop_Cols:
            self.dvoa_dataset.drop(col, axis=1, inplace=True)
        self.dvoa_dataset = self.dvoa_dataset.sort_values(by=["Year", "Week", "Team"])
        self.dvoa_dataset.to_csv(self.Final_DVOA_DF_Save_Path, index=False)


    def Get_DVOA_DF(self):
        # URL_BASE = f'https://www.footballoutsiders.com/stats/teameff/{self.season}'
        have_df = False
        try:
            URL_BASE = f'https://www.footballoutsiders.com/dvoa-ratings/{self.season}/week-{self.week-1}-dvoa-ratings'
            html = requests.get(URL_BASE).content
            for head_row in range(0,2):
                df_list = pd.read_html(html, header=head_row, index_col=0)
                # print(df_list)
                for i in range(0, len(df_list)):
                    Pot_DVOA_DF = df_list[i]
                    pot_dvoaHeaders = list(Pot_DVOA_DF)
                    # print(pot_dvoaHeaders)
                    if len(pot_dvoaHeaders) >= 10 and "TEAM" in pot_dvoaHeaders: #The right df
                        for header in self.Possible_WeightedDVOA_Names:
                            if header in pot_dvoaHeaders: #Have the weighted df
                                self.DVOA_DF = Pot_DVOA_DF
                                have_df = True
                                break
                if have_df == True:
                    break
        except ValueError:
            pass
        if have_df == False: #Never found a good df
            self.no_dvoa_weeks.append(f'No Good DF: {self.season}, week {self.week}')
        #IF NO DF FOUND (2015 Week 10) then will use previous weeks DVOA DF
        
        #Drop any header rows
        self.DVOA_DF.drop(self.DVOA_DF.loc[self.DVOA_DF['TEAM']=="TEAM"].index, inplace=True)
        print(self.DVOA_DF)
        self.dvoaHeaders = list(self.DVOA_DF)
        print(self.dvoaHeaders)
        self.DVOA_DF = self.DVOA_DF.sort_values(by=["TEAM"])

    def Get_Team_Lists(self):
        #Get list of dvoa teams for the season
        self.Season_DVOA_Teams = self.DVOA_DF['TEAM'].tolist()
        self.Season_DVOA_Teams.sort()
        print(self.Season_DVOA_Teams)
        #Get list of data teams for the season
        self.Season_Stat_Teams = self.Season_DF.Team.unique()
        self.Season_Stat_Teams.sort()
        print(self.Season_Stat_Teams)
        # self.time.sleep(1)

    def Change_Names(self):
        self.DVOA_New_Names = []
        for name in range(0, len(self.Season_DVOA_Teams)):
            dvoaName = self.Season_DVOA_Teams[name]
            highest_match = 0
            for offset in [0, 1, -1, 2, -2]:
                if name+offset < len(self.Season_Stat_Teams):
                    match_letters = 0
                    teamName = self.Season_Stat_Teams[name+offset]
                    for letter in dvoaName:
                        if letter in teamName:
                            match_letters += 1
                    if match_letters > highest_match: #New most matching name letters
                        stat_names_pos = name+offset
                        highest_match = match_letters
            self.DVOA_New_Names.append(self.Season_Stat_Teams[stat_names_pos])
        print(self.DVOA_New_Names)
        self.DVOA_DF["TEAM"] = self.DVOA_New_Names
        self.DVOA_DF = self.DVOA_DF.sort_index()
        print(self.DVOA_DF)

    def Make_Folder(self, new_path):
        try:
            os.mkdir(new_path)
        except:
            pass

    def Save_DVOA_DF(self):
        self.Make_Folder(self.DVOA_Raw_Path)
        season_path = f'{self.DVOA_Raw_Path}/{self.season}'
        self.Make_Folder(season_path)
        # if self.season != 2019:
        #     save_week = 17
        # else:
        save_week = self.week
        week_path = f'{season_path}/Week {save_week}'
        self.Make_Folder(week_path)
        save_path = f'{week_path}/DVOA Data.csv'
        self.DVOA_DF.to_csv(save_path, index=False)


    def Get_DVOA_Vals(self):
        self.Team_DVOA_DF = self.DVOA_DF.loc[(self.DVOA_DF["TEAM"] == self.team)]
        self.Opp_DVOA_DF = self.DVOA_DF.loc[(self.DVOA_DF["TEAM"] == self.opp)]
        for header in self.Possible_WeightedDVOA_Names:
            try:
                if self.DVOA_Type == "DVOA":
                    header = self.dvoaHeaders[1]
                # print(header)
                # self.team_dvoa = self.Team_DVOA_DF[self.dvoaHeaders[3]].tolist()[-1]
                self.team_dvoa = self.Team_DVOA_DF[header].tolist()[-1]
                # print(self.team_dvoa)
                self.team_dvoa = float(self.team_dvoa.split('%')[0])

                # self.opp_dvoa = self.Opp_DVOA_DF[self.dvoaHeaders[3]].tolist()[-1]
                self.opp_dvoa = self.Opp_DVOA_DF[header].tolist()[-1]
                self.opp_dvoa = float(self.opp_dvoa.split('%')[0])
                self.head+=1
                break
            except:
                pass

        self.dvoa_diff = self.team_dvoa-self.opp_dvoa
        self.DVOA_Diffs[self.location].append(self.dvoa_diff)
        self.Scores_byLoc[self.location].append(self.game_margin)
        self.All_DVOA_Diffs.append(self.dvoa_diff)
        self.All_Team_DVOAs.append(self.team_dvoa)
        # print(self.Team_DVOA_DF)
        # print(self.Opp_DVOA_DF)
        # print(self.team_dvoa)
        # print(self.opp_dvoa)
        # print(self.dvoa_diff)

    def Bin_DVOA(self):
        for group in range(0, len(self.DVOA_Diff_Groups)):
            if self.dvoa_diff >= self.DVOA_Diff_Groups[group] and self.dvoa_diff < 0:
                continue
            elif self.dvoa_diff > self.DVOA_Diff_Groups[group] and self.dvoa_diff >= 0:
                continue
            else:
                # print(self.dvoa_diff)
                # print(self.DVOA_Diff_Groups[group])
                # print(self.game_margin)
                self.Scores_byDVOA[self.location][group].append(self.game_margin)
                self.Season_Scores_byDVOA[self.location][group].append(self.game_margin)
                self.Spreads_byDVOA[self.location][group].append(self.spread)
                self.Season_Spreads_byDVOA[self.location][group].append(self.spread)
                break
        # self.time.sleep(2)

    def plotly_graph(self, x_data, y_data, save_path, file_name, legend, x_axis_name, y_axis_name):
        plt_title = ' '.join([file_name, y_axis_name, "vs", x_axis_name])
        plt_title = file_name
        plt_save_name = f'{save_path}/{plt_title}.html'
        # print(len(x_data))
        # print(len(y_data))
        data= []
        line_Train_Test_Seasons = ['markers', 'lines+markers', 'lines+markers', 'markers', 'lines+markers']
        # hovertexts= [self.Hover_Data[:-1], ["lobf" for i in range(len(x_data[1]))], [self.Hover_Data[-1]]]
        for i in range(0,len(legend)):
            # print(i)
            # print(legend[i])
            trace = go.Scatter(
            x=x_data[i],
            y=y_data[i],
            mode = line_Train_Test_Seasons[i],
            name = legend[i],
            # hovertext=hovertexts[i],
            )
            data.append(trace)
        x_axis_label = f'{x_axis_name}'
        y_axis_label = f'{y_axis_name}'
        plotly.offline.plot({
        "data": data,
        "layout": go.Layout(title=go.layout.Title(text=plt_title, xref="paper", x=0.5), showlegend=True, xaxis=dict(title = x_axis_label, color="#000", gridcolor="rgb(232, 232, 232)"), yaxis = dict(title = y_axis_label, color="#000", gridcolor="rgb(232, 232, 232)"),  plot_bgcolor='rgba(0,0,0,0)')
        }, filename = plt_save_name, auto_open=False)

    def Interpolate_EGOs(self, loc):
        self.DVOA_Ranges = []
        for diff in range(0, len(self.DVOA_Diff_Groups)):
            UL = self.DVOA_Diff_Groups[diff]
            if diff == 0:
                LL = -100
            else:
                LL = self.DVOA_Diff_Groups[diff-1]
            if LL == -100:
                avg = -75
            elif UL == 100:
                avg = 75
            else:
                avg = round(((LL + UL)/2),2)
            self.DVOA_Ranges.append(avg)
        print(self.DVOA_Ranges)

        self.Best_Fit_Scores = np.poly1d(np.polyfit(self.DVOA_Diffs[loc], self.Scores_byLoc[loc], 2))(np.unique(self.DVOA_Ranges))
        X_Vals = []
        Y_Vals = []
        headers = []
        Y_Vals.append(self.Scores_byLoc[loc])
        X_Vals.append(self.DVOA_Diffs[loc])
        headers.append("All points")
        Y_Vals.append(self.Best_Fit_Scores)
        X_Vals.append(np.unique(self.DVOA_Ranges))
        headers.append("Grouped 2nd order LOBF")
        print(self.Best_Fit_Scores)
        # self.plotly_graph(X_Vals, Y_Vals, self.DVOA_Raw_Path, f'{self.Game_Locations[loc]}', headers, 'DVOA Diff', 'Score Margin')


    def Get_DVOA_Stats(self, Score_Data, Spread_Data, season, savePath):
        self.DVOA_Stat_DF = pd.DataFrame()
        # self.Interpolate_EGOs(Score_Data[0]) #Get the smoothed version of the Avg Game MArgins
        for loc in range(0, len(self.Game_Locations)):
            self.Interpolate_EGOs(loc) #Get the smoothed version of the Avg Game MArgins
            for group in range(0, len(self.DVOA_Diff_Groups)):
                Group_Margins = Score_Data[loc][group]
                Group_Spreads = Spread_Data[loc][group]
                UL = self.DVOA_Diff_Groups[group]
                if group == 0:
                    LL = -100
                else:
                    LL = self.DVOA_Diff_Groups[group-1]
                print(Group_Margins)
                if len(Group_Margins) >= 1:
                    self.DVOA_Stats[0].append(season)
                    self.DVOA_Stats[1].append(f'{LL} to {UL}')
                    self.DVOA_Stats[2].append(len(Group_Margins))
                    if len(Group_Margins) < 2:
                        Group_Margins += Group_Margins
                    self.DVOA_Stats[3].append(mean(Group_Spreads))
                    self.DVOA_Stats[4].append(mean(Group_Margins))
                    self.DVOA_Stats[5].append(self.Best_Fit_Scores[group])
                    self.DVOA_Stats[6].append(max(Group_Margins))
                    self.DVOA_Stats[7].append(min(Group_Margins))
                    self.DVOA_Stats[8].append(stdev(Group_Margins))
            for stat in range(0,len(self.DVOA_Stat_Headers)):
                print(len(self.DVOA_Stats[stat]))
                self.DVOA_Stat_DF[f'{self.Game_Locations[loc]} {self.DVOA_Stat_Headers[stat]}'] = self.DVOA_Stats[stat]
            self.DVOA_Stats = [[] for i in range(len(self.DVOA_Stat_Headers))] #Year, Bin, Count, Average, Max, Min, Std Dev

        self.DVOA_Stat_DF.to_csv(f'{savePath}/{self.Training_DVOA_Stat_save_name}', index=False)
        print(self.DVOA_Stat_DF)
        # self.time.sleep(2)

    def Do_DVOA_Analysis(self):
        Teams = self.Week_DF["Team"].tolist()
        Opponents = self.Week_DF["Opponent"].tolist()
        Game_Margins = self.Week_DF["Game Scoring Margin"].tolist()
        Spreads = self.Week_DF["Spread"].tolist()
        Locations = self.Week_DF["Home Team"].tolist()
        self.head = 0
        for self.game_num in range(0,len(Teams)):
            self.location = Locations[self.game_num]
            self.team = Teams[self.game_num]
            self.opp = Opponents[self.game_num]
            self.game_margin = Game_Margins[self.game_num]
            self.spread = Spreads[self.game_num]
            # self.All_Game_Margins.append(self.game_margin)
            self.Get_DVOA_Vals()
            self.Bin_DVOA()
        if self.head == 0: #No col found
            self.no_weighted_dfs.append(f'No Good Col: {self.season}, week{self.week}')
        print(self.no_weighted_dfs)
        #Add Season's DVOA Stats to DF
        # self.Get_DVOA_Stats(self.Season_Scores_byDVOA, self.Season_Spreads_byDVOA, self.season, self.BySeason_DVOA_Stat_save_path)

    def Setup_Data(self):
        self.dataset = pd.read_csv(self.Raw_Data_Path)
        self.dataset = self.dataset.loc[(self.dataset["Year"] >= 2006)]
        # self.dataset = self.dataset.loc[(self.dataset["Home Team"] == self.home_val)]
        self.Seasons = self.dataset['Year'].unique()
        print(self.Seasons)
       
    def Do_Stuff(self):
        for self.iteration in range(0, len(self.Train_Test_Seasons[0])+1):
            self.Train_Test_Seasons = [[2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019], []]
            if self.iteration == len(self.Train_Test_Seasons[0]):
                del self.Train_Test_Seasons[-1]
                self.Training_DVOA_Stat_save_path = f'{self.project_path}/Total Data'
                self.Training_DVOA_Stat_save_name = f'All Seasons Scores Grouped By {self.DVOA_Type} Diff.csv'
            else:
                self.Train_Test_Seasons[1].append(self.Train_Test_Seasons[0][self.iteration]) #Add the season to test seasons
                del self.Train_Test_Seasons[0][self.iteration] #Delete the season from the training seasons
                self.Training_DVOA_Stat_save_name = f'Excluding {self.Train_Test_Seasons[1][0]} Scores Grouped By {self.DVOA_Type} Diff.csv'
            print(self.Train_Test_Seasons)
            self.Initialize_Data()
            max_week = 16
            # max_ls_week = int(input(f'Enter The most recent week of the latest season: '))
            # self.max_ls_week = 14
            self.Setup_Data()
            for mode in range(0,len(self.Train_Test_Seasons)):
            # for mode in range(0,len(self.Train_Test_Seasons)-1):
                self.Seasons = self.Train_Test_Seasons[mode]
                # self.Seasons = self.Train_Test_Seasons[mode] + self.Train_Test_Seasons[1]
                self.Seasons.sort()
                self.All_Team_DVOAs = []
                self.All_DVOA_Diffs = []
                self.All_Game_Margins = []
                for self.season in self.Seasons:
                    print(self.season)
                    self.Season_DF = self.dataset.loc[(self.dataset["Year"] == self.season) & (self.dataset["Week"] >= self.min_week)]
                    print(self.Season_DF)
                    self.Season_Scores_byDVOA = [[[] for i in range(len(self.DVOA_Diff_Groups))] for i in range(len(self.Game_Locations))]
                    self.Season_Spreads_byDVOA = [[[] for i in range(len(self.DVOA_Diff_Groups))] for i in range(len(self.Game_Locations))]
                    
                    if self.season == self.max_season:
                        max_week = self.max_ls_week
                    for self.week in range(self.min_week, max_week+1):
                        print(f'WEEK {self.week}')
                        # if self.season == self.max_season and self.week == self.max_ls_week:
                        #     #USE IF NEED TO GET NEW DVOA DATA
                        #     self.Get_DVOA_DF()
                        #     self.Get_Team_Lists()
                        #     self.Change_Names()
                        #     self.Save_DVOA_DF()
                        # else:
                        #USE IF CAN READ PREVIOUSLY SAVED DVOA DATA
                        self.DVOA_DF = pd.read_csv(f'{self.DVOA_Raw_Path}/{self.season}/Week {self.week}/DVOA Data.csv')
                        self.dvoaHeaders = list(self.DVOA_DF)
                        print(self.dvoaHeaders)

                        self.Week_DF = self.Season_DF.loc[(self.Season_DF["Week"] == self.week)]
                        self.Do_DVOA_Analysis()
                        print(self.season)
                        print(self.no_dvoa_weeks)
                if mode == 0: #Training
                    #Save Total DVOA Stats
                    self.Get_DVOA_Stats(self.Scores_byDVOA, self.Spreads_byDVOA, "All", self.Training_DVOA_Stat_save_path)
                    # self.All_Training_DVOA_Diffs = self.All_DVOA_Diffs
                    # self.All_Training_Game_Margins = self.All_Game_Margins
                if mode == 1 or len(self.Train_Test_Seasons) == 1: #Testing Mode Or if Training/Testing on same data (Final season - 2020)
                    # Add data to the overall DF
                    self.Add_DVOA_Data()
                print(self.no_dvoa_weeks)

# DVOA_Analysis = DVOA_DATA(16)
# DVOA_Analysis.Do_Stuff()
