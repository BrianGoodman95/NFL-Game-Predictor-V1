import pandas as pd
import numpy as np
import random
import time
import os
from scipy.stats import pearsonr
import plotly
import plotly.graph_objs as go
from itertools import combinations 
from sklearn.preprocessing import MinMaxScaler

#For each team of each season get all games up to that week (only week 5 and later)
#For each of the Big 6 columns, get correlation factor between spread/result difference and the advantage
#For the top column, interpolate the spread result based on the column adv and then turn into covered/not
#Check predictions from above with actual result


class NFL_DATA_MODEL():
    def __init__(self, project_path, min_week, min_year, DVOA_Type):
        #Data is in the form of team stats up to week #, opp stats up to week #, data about game (home team, day of week, bye week, spread) and result class of game - what I want to predict
        #Multi class classification model needed to predict big upset, upset, expected, blowout, big blowout
        # self.DVOA_Type = input(f'Enter The DVOA Type to use (DVOA or WDVOA): ')
        # self.DVOA_Type = "WDVOA"
        self.DVOA_Type = DVOA_Type
        if self.DVOA_Type != "WDVOA" and self.DVOA_Type != "DVOA":
            print("BAD DVOA TYPE ENTERED")
        self.time = time
        self.min_year = min_year
        self.project_path = project_path
        self.Model_Data_Path = f'{self.project_path}/Total Data/All Seasons {self.DVOA_Type} Results.csv'
        # self.Training_DVOA_Stat_save_name = f'Excluding {self.Train_Test_Seasons[1][0]} Scores Grouped By {self.DVOA_Type} Diff.csv'
        # self.Training_DVOA_Stat_save_path = f'{self.project_path}/Train_Test_Data/{self.Training_DVOA_Stat_save_name}'
        self.Training_DVOA_Stat_save_path = f'{self.project_path}/Total Data/All Seasons Scores Grouped By {self.DVOA_Type} Diff.csv'
        self.Game_Prediction_Data_Path = f'{self.project_path}/Total Data/All {self.DVOA_Type} Model Data.csv'
        self.Total_Prediction_Data_Path = f'{self.project_path}/Total Data/Total {self.DVOA_Type} Prediction Data.csv'
        self.Regression_Model_Data_Path = f'{self.project_path}/ML Data/Model Data/Regression {self.DVOA_Type} Model Data.csv'
        self.Classification_Model_Data_Path = f'{self.project_path}/ML Data/Model Data/Classification {self.DVOA_Type} Model Data.csv'

        self.Game_Locations = ["Away", "Home"]
        # self.Game_Cols = ["Team", "Year", "Week", "Opponent", "Spread", "Game Scoring Margin", "SRD"]
        self.Team_Cols = ["Line For Rating", "Line Against Rating", "Rushing For Rating", "Rushing Against Rating", "Passing For Rating", "Passing Against Rating"]
        self.Correlation_Cols = [f'Opp {name}' for name in self.Team_Cols]
        self.Matchup_EGOs = [[] for i in range(len(self.Correlation_Cols))]
        self.Matchup_Correlations = [[] for i in range(len(self.Correlation_Cols))]
        self.All_MT_EGO_Diffs = []
        self.All_EGOs = []

        self.regressionNN_Drop_Cols = ["Team", "Opponent", "Year", "Home Team", "Team DVOA", "DVOA Diff", "DVOA EGO to Spread Diff", "Game Scoring Margin", "Spread to Result Difference"]#, "Spread Result Class"]
        self.regressionNN_New_Cols = ["EGO To Result Diff"]
        self.classificationNN_Drop_Cols = ["Team", "Opponent", "Year", "Home Team", "Spread", "Team DVOA", "DVOA Diff", "Game Scoring Margin", "Spread to Result Difference"]#, "Spread Result Class"]
        self.classificationNN_New_Cols = ["EGO Pick Correct"]

        self.min_week = min_week

    def Make_Folder(self, new_path):
        path_exists = False
        try:
            os.mkdir(new_path)
        except:
            print('folder exists')
            path_exists = True
        return path_exists

##### HERE IS WHERE WE HAVE OUR GRAPHING PART FOR THE MATCHUP CORRELATIONS ####

    def plotly_graph(self, x_data, y_data, save_path, file_name, legend, x_axis_name, y_axis_name):
        # plt_title = ' '.join([file_name, y_axis_name, "vs", x_axis_name])
        plt_title = file_name
        plt_save_name = f'{save_path}/{plt_title}.html'
        # print(len(x_data))
        # print(len(y_data))
        data= []
        line_modes = ['markers', 'lines+markers', 'lines+markers']
        hovertexts= [self.Hover_Data[:-1], ["lobf" for i in range(len(x_data[1]))], [self.Hover_Data[-1]]]
        for i in range(0,len(legend)):
            # print(i)
            # print(legend[i])
            trace = go.Scatter(
            x=x_data[i],
            y=y_data[i],
            mode = line_modes[i],
            name = legend[i],
            hovertext=hovertexts[i],
            )
            data.append(trace)
        x_axis_label = f'{x_axis_name}'
        y_axis_label = f'{y_axis_name}'
        plotly.offline.plot({
        "data": data,
        "layout": go.Layout(title=go.layout.Title(text=plt_title, xref="paper", x=0.5), showlegend=True, xaxis=dict(title = x_axis_label, autorange='reversed', color="#000", gridcolor="rgb(232, 232, 232)"), yaxis = dict(title = y_axis_label, color="#000", gridcolor="rgb(232, 232, 232)"),  plot_bgcolor='rgba(0,0,0,0)')
        }, filename = plt_save_name, auto_open=False)

    def Do_Graphing(self):
        #Graph Stuff
        Best_Feature_Data = [self.prior_feature_data]
        Y_Vals = [self.prior_label_data]
        leg_header = self.Correlation_Cols[self.num]
        headers = [f'{leg_header}-{self.corr}']
        #LOBF Graph Stuff
        headers.append("LOBF")
        Best_Feature_Data.append(self.lobf_Xvals)
        Y_Vals.append(self.lobf_Yvals)
        #SRD Plot
        Best_Feature_Data.append([self.feature_data[-1]])
        Y_Vals.append([self.ego_score_prediction])
        headers.append("PREDICTION")
        #Plot
        self.plotly_graph(Best_Feature_Data, Y_Vals, self.plt_save_path, f'{self.team} {leg_header} Correlation of {round(-1*self.corr,2)}', headers, 'Opp Rating', 'Score to EGO Difference')

    def Setup_Graphing(self):
        #Make Plotting Hover Data and save path
        self.Hover_Data = []
        for game in range(0, len(self.Opps)):
            self.Hover_Data.append(f'{self.Opps[game]}-Week{self.Weeks[game]}')
        self.prior_label_data = self.label_data[:-1]
        # del self.prior_label_data[-1]
        self.plt_save_path = f'{self.project_path}/Raw Data/{self.year}/Week {self.week}/Matchup Correlation Graphs'
        self.Make_Folder(self.plt_save_path)


##### HERE IS WHERE WE FIND THE MATCHUP CORRELATIONS #####

    def Get_Opponent_Ratings(self, col):
        #For the col in question, get the opponents ranking in the col for the latest week
        self.feature_data = []
        print(col)
        column = col.split('Opp ')[1]
        print(column)
        for opp in self.Opps:
            Opp_DF = self.dataset.loc[(self.dataset["Team"] == opp) & (self.dataset['Year'] == self.year) & (self.dataset['Week'].isin(self.weeks))]
            opp_data = Opp_DF[column].tolist()[-1]
            self.feature_data.append(opp_data)
            print(self.feature_data)

    def Get_EGOScore_Diff_Data(self, game_score_data):
        #for each opponent, get their most recent DVOA Diff
        #Use Map to assign an EGO and then take difference from label data (game margin)
        label_data = []
        print(self.team)
        print(self.Opps)
        # Team_DF = self.dataset.loc[(self.dataset["Team"] == self.team) & (self.dataset['Year'] == self.year) & (self.dataset['Week'].isin(self.weeks))]
        #Using the team's most recent WDVOA rating
        teamDVOA = self.Team_DF["Team DVOA"].tolist()[-1]    
        for game in range(0, len(self.Opps)):
            #Get each opponents most recent team DVOA rating
            Opp_DF = self.dataset.loc[(self.dataset["Team"] == self.Opps[game]) & (self.dataset['Year'] == self.year) & (self.dataset['Week'].isin(self.weeks))]
            print(Opp_DF)
            oppDVOA = Opp_DF["Team DVOA"].tolist()[-1]        
            #Get location of teh game (from Team's perspective)
            Loc = self.Team_DF["Home Team"].tolist()[game]
            DVOA_Diff = teamDVOA - oppDVOA
            #Get the EGO
            EGO = round(np.interp(DVOA_Diff, list(self.Maps[Loc].keys()), list(self.Maps[Loc].values())),2)
            #Get the EGO - Score Difference
            Score_EGO_Diff = game_score_data[game] - EGO
            label_data.append(Score_EGO_Diff)
        print(label_data)
        return label_data

    def Get_EGO_Result_Vals(self): 
        #Define data header
        label_header = 'Game Scoring Margin'
        game_score_data = self.Team_DF[label_header].tolist() #Get the game scoring margins for each teams game up to the current week
        self.Opps = self.Team_DF["Opponent"].tolist() #Get all opponents of that tema up to the current week
        self.Weeks = self.Team_DF["Week"].tolist()  #Get the week number of each game played
        EGOs = self.Team_DF["EGO"].tolist() #Get the EGO assined from DVOA for each teams game up to the current week
        self.All_EGOs.append(EGOs[-1])
        self.label_data = self.Get_EGOScore_Diff_Data(game_score_data) #Get a list of the EGO-Game Result differences for each game up to the current week ***BASED ON EGO Calculated as of the Teams' WDVOA as of the current week
       
    def Make_Prediction(self):
        # feature_val = self.feature_data[-1]
        # ego_score_prediction = int(round(np.interp(self.feature_data[-1], self.prior_feature_data, self.prior_label_data)))
        self.ego_score_prediction = np.poly1d(np.polyfit(self.prior_feature_data, self.prior_label_data, 1))(self.feature_data[-1])# (np.unique(self.prior_feature_data))
        self.lobf_Yvals = np.poly1d(np.polyfit(self.prior_feature_data, self.prior_label_data, 1))(np.unique(self.prior_feature_data))
        self.lobf_Xvals = np.unique(self.prior_feature_data)
        print(self.feature_data[-1])
        print(self.ego_score_prediction)
        print(self.label_data[-1])
        # time.sleep(5)

    def Filter_Trends(self):
        #Filter #1 - Teams' opponents must have a spread of rankings of at least 50 - can't have played only teams with rankings between 40-60 in the stat - this is not a clear sample size
        Opp_Stat_Spread = max(self.prior_feature_data) - min(self.prior_feature_data)
        print(Opp_Stat_Spread)
        if Opp_Stat_Spread >= 50: #If the team has faced oppenets with a range of at least 50 in their stat ratings (30-80 at least)
            Good_Matchup_Spread = True
        else:
            Good_Matchup_Spread = False
        
        # #Filter #2 - Teams' Opponent must have a rating no more or less than 20 of the next highest or lowest previous opponent rating - if more, than this is too much a difference and will cause a skewed/unreliable prediction
        thisWeek_oppRating = self.feature_data[-1]
        if thisWeek_oppRating > max(self.prior_feature_data)+20 or thisWeek_oppRating < min(self.prior_feature_data)-20: #If 20 more or less than previous highest/lowest opponent
            Good_Opponent_Rating = False
        else:
            Good_Opponent_Rating = True

        #Filter #3 - The correlation for this stat must have a certain minimum value. Don't want tiny correlations with relatively large EGO predictions (maybe because all values on the graph are above or below zero) to skew the result
        self.min_cor = 0.1 #Minimum
        if abs(self.corr) >= self.min_cor:
            Good_Correlation = True
        else:
            Good_Correlation = False

        if Good_Matchup_Spread == True and Good_Opponent_Rating == True and Good_Correlation == True:
            #Make Matchup Based EGO Prediction
            self.Make_Prediction()
        else:
            #Assign a zero for ego adjustment
            self.ego_score_prediction = 0

    def Get_Stat_Correlation(self):
        self.Get_Opponent_Ratings(self.Correlation_Cols[self.num])
        self.prior_feature_data = self.feature_data[:-1]
        self.corr, p_value = pearsonr(self.prior_feature_data, self.prior_label_data)
        print(self.Correlation_Cols[self.num])
        print(self.corr)
        self.Filter_Trends()
        #Make Plots for the stat
        if self.year == 2019 and self.week >= 15 and self.ego_score_prediction!= 0 and self.DVOA_Type == 'WDVOA':
            self.Do_Graphing()      
          
    def Track_Correlations(self):
        self.weighted_ego_diff += abs(self.corr)*self.ego_score_prediction
        self.Matchup_EGOs[self.num].append(self.ego_score_prediction)
        self.Matchup_Correlations[self.num].append(self.corr*-1)


    def Correlation_Collector(self):
        self.weighted_ego_diff = 0

        self.Get_EGO_Result_Vals()
        self.Setup_Graphing()
        for self.num in range(0, len(self.Correlation_Cols)):   
            self.Get_Stat_Correlation()     
            self.Track_Correlations()
        self.All_MT_EGO_Diffs.append(round(self.weighted_ego_diff,2))

##### ABOVE HERE IS WHERE WE READ THE INPUT DATA AND FILTER FOR YEARS/WEEKS WE CARE ABOUT ######

    def Get_Rows(self):
        self.Team_List = self.correlation_dataset["Team"].tolist()
        self.Year_List = self.correlation_dataset["Year"].tolist()
        self.Week_List = self.correlation_dataset["Week"].tolist()
        self.Spread_List = self.correlation_dataset["Spread"].tolist()
        self.SRC_List = self.correlation_dataset["Spread Result Class"].tolist()
        for game in range(0, len(self.Team_List)):
            self.team = self.Team_List[game]
            self.year = self.Year_List[game]
            self.week = self.Week_List[game]
            self.weeks = [num for num in range(1, self.week+1)]
            # if self.week < 6:
            #     self.weeks = [num for num in range(1,7)] #Up to Week 6
            if self.week >= self.min_week:# and self.year >= self.min_year: #Only Do This for Week 6 and later data
                print(self.team)
                print(self.week)
                self.Team_DF = self.dataset.loc[(self.dataset["Team"] == self.team) & (self.dataset['Year'] == self.year) & (self.dataset['Week'].isin(self.weeks))]
                print(self.Team_DF)
                # Analyze the Matchup Correlations
                self.Correlation_Collector()

    def Setup_Map(self):
        #Need to take all the DVOA_Diffs and map them to an EGO via the total home or away map
        #Read the Map DF
        self.Map_DF = pd.read_csv(self.Training_DVOA_Stat_save_path)
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

    def Read_Data(self):
        self.dataset = pd.read_csv(self.Model_Data_Path)
        print(self.dataset.head())
        print(self.dataset.shape)
        print(self.dataset.tail())  
        self.correlation_dataset = self.dataset.loc[(self.dataset['Week'] >= self.min_week) & (self.dataset['Year'] >= self.min_year)]


##### HERE IS WHERE WE SAVE OUR NEW DATAFRAMES FOR THE USE OF MORE SOPHISTICATED MODELS #####

    def ReOrder_Cols(self):
        #Get the column order
        template_df = pd.read_csv(f'{self.project_path}/Templates/All_Model_Data_Template.csv')
        cols = list(template_df)
        #Rename necessary columns
        self.correlation_dataset = self.correlation_dataset.rename(columns={"EGO": "DVOA EGO", "EGO to Spread Diff": "DVOA EGO to Spread Diff", "EGO Pick": "DVOA EGO Pick", "Right Pick": "DVOA Pick Right"})
        #Re-Order Columns
        self.correlation_dataset = self.correlation_dataset[cols]

    def Save_Correlation_Data(self):
        for stat in range(0,len(self.Correlation_Cols)):
            # stat_name_parts = self.Correlation_Cols[stat].split(' ')
            ego_stat_name = f'{self.Correlation_Cols[stat]} EGO Adj'
            cor_stat_name = f'{self.Correlation_Cols[stat]} Correlation'
            self.correlation_dataset[ego_stat_name] = self.Matchup_EGOs[stat]
            self.correlation_dataset[cor_stat_name] = self.Matchup_Correlations[stat]
        # self.correlation_dataset["Matchup EGO"] = self.All_MT_EGO_Diffs
        # self.correlation_dataset["Total EGO"] = self.Total_Score_Predictions
        # self.correlation_dataset["Game Pick"] = self.Picks
        self.ReOrder_Cols()
        self.correlation_dataset.to_csv(self.Game_Prediction_Data_Path, index=False)


### HERE IS WHERE WE MAKE THE VARIOUS MODEL DATASETS ###

    def Get_EGO_Score_Diff(self):
        dvoaEGOs = self.regression_dataset["DVOA EGO"].tolist()
        gameScores = self.regression_dataset["Game Scoring Margin"].tolist()
        egoScore_Diffs = []
        for game in range(0, len(dvoaEGOs)):
            egoScroe_Diff = gameScores[game] - dvoaEGOs[game]
            egoScore_Diffs.append(egoScroe_Diff)
        self.regression_dataset[self.regressionNN_New_Cols[0]] = egoScore_Diffs

    def Get_EGO_Prediction_Class(self):
        dvoaEGOs = self.classification_dataset["DVOA EGO"].tolist()
        spreads = self.classification_dataset["Spread"].tolist()
        gameScores = self.classification_dataset["Game Scoring Margin"].tolist()
        EGO_Pick_Class = []
        for game in range(0, len(dvoaEGOs)):
            egoScore = dvoaEGOs[game]
            spreadScore = spreads[game]*-1
            score = gameScores[game]
            if (score >= spreadScore and egoScore >= spreadScore) or (score <= spreadScore and egoScore <= spreadScore):
                EGO_Pick_Class.append(1)
            else:
                EGO_Pick_Class.append(-1)
        self.classification_dataset[self.classificationNN_New_Cols[0]] = EGO_Pick_Class

    def Save_Regression_Model_Data(self):
        print(list(self.regression_dataset))
        for col_name in self.regressionNN_Drop_Cols:
            self.regression_dataset.drop(f'{col_name}', axis=1, inplace=True)
        self.regression_dataset.to_csv(self.Regression_Model_Data_Path, index=False)
    
    def Save_Classification_Model_Data(self):
        print(list(self.classification_dataset))
        for col_name in self.classificationNN_Drop_Cols:
            self.classification_dataset.drop(f'{col_name}', axis=1, inplace=True)
        self.classification_dataset.to_csv(self.Classification_Model_Data_Path, index=False)

    def Save_Model_Data(self):
        self.model_dataset = pd.read_csv(self.Game_Prediction_Data_Path)
        self.regression_dataset = self.model_dataset.copy()
        self.Get_EGO_Score_Diff()     
        self.classification_dataset = self.model_dataset.copy()
        self.Get_EGO_Prediction_Class()        

        self.Save_Regression_Model_Data()
        self.Save_Classification_Model_Data()


##### HERE IS WHERE WE MAKE MATCHUP BASED BASIC PREDICTIONS ####

    def Save_New_Data(self):
        for stat in range(0,len(self.New_Matchup_Cols)):
            self.matchup_dataset[self.New_Matchup_Cols[stat]] = self.New_Matchup_Data[stat]
        template_df = pd.read_csv(f'{self.project_path}/Templates/Total_Prediction_Data_Template.csv')
        cols = list(template_df)
        print(cols)
        #Re-Order Columns
        self.matchup_dataset = self.matchup_dataset[cols]
        self.matchup_dataset.to_csv(self.Total_Prediction_Data_Path, index=False)

    def Evaluate_Picks(self):
        #Evaluate Picks
        Right_Picks = []
        Wrong_Picks = []
        for pick in self.New_Matchup_Data[5]:
            if pick == 1:
                Right_Picks.append(1)
            elif pick == -1:
                Wrong_Picks.append(1)
        num_right_picks = sum(Right_Picks)
        num_wrong_picks = sum(Wrong_Picks)
        pick_acc = round(num_right_picks/(num_right_picks+num_wrong_picks),4)
        print(f'EGO Based Predictions gave a {pick_acc}% Betting Accuracy')
        # self.All_accs.append(pick_acc)

    def Scale_Matchup_Strengths(self):
        #Sort all the Matchup Strengths and normalize them with the highest/lowest being +/- 6 and the other being scaled from that
        #Get the min and max to scale to
        min_strength = min(self.New_Matchup_Data[0])
        max_strength = max(self.New_Matchup_Data[0])
        print(min_strength)
        print(max_strength)
        if max_strength >= abs(min_strength): #Max is bigger so this will scale to be + 6 and min scales accordingly
            max_scale = self.largest_EGO_change
            min_scale = -1*self.largest_EGO_change*(abs(min_strength/max_strength)) #Something like -5
        else:
            min_scale = -1*self.largest_EGO_change
            max_scale = self.largest_EGO_change*(abs(max_strength/max_strength)) #Something like 5
        scaler = MinMaxScaler(feature_range=(min_scale, max_scale))
        array = np.array(self.New_Matchup_Data[0])
        Data = array.reshape(-1,1)
        scaled_strengths = scaler.fit_transform(Data)
        self.New_Matchup_Data[1] = scaled_strengths
    
    def Adjust_EGO_Make_Picks(self):
        self.EGOs = self.matchup_dataset["DVOA EGO"].tolist()
        self.Spreads = self.matchup_dataset["Spread"].tolist()
        self.Spread_Results = self.matchup_dataset["Spread Result Class"].tolist()
        print(len(self.EGOs))
        print(len(self.New_Matchup_Data[0]))
        for game in range(0, len(self.EGOs)):
            new_ego = self.EGOs[game] + self.New_Matchup_Data[1][game]
            self.New_Matchup_Data[2].append(new_ego[0])
            new_ego_diff = new_ego - (-1*self.Spreads[game])
            self.New_Matchup_Data[3].append(new_ego_diff[0])
            if new_ego_diff < 0:
                new_pick = -1
            else:
                new_pick = 1
            self.New_Matchup_Data[4].append(new_pick)
            if new_pick == self.Spread_Results[game]:
                pick_right = 1
            elif self.Spread_Results[game] == 0:
                pick_right = 0
            else:
                pick_right = -1
            self.New_Matchup_Data[5].append(pick_right)

    def Determine_Total_Matchup_Strength(self, Team_Matchup_Strength, Opp_Matchup_Strength):
        #Check the direction of the matchup strengths
        if Team_Matchup_Strength*Opp_Matchup_Strength < 0: #Opposite sign - so good for opponent and bad for team or vice versa
            #Add the matchup strengths
            total_strength = Team_Matchup_Strength + (-1*Opp_Matchup_Strength)
        elif Team_Matchup_Strength*Opp_Matchup_Strength > 0: #Same sign so matchup is good for both teams so no matchup strength then
            total_strength = 0
        else: #Opponent has 0 so take teams or team has 0 so take teams (0)
            total_strength = Team_Matchup_Strength

        return total_strength

    def Determine_Individual_Matchup_Strength(self, Matchup_Strengths):
        Matchup_Strength = 0 #Assume matchup total of 0
        Bad_Matchups = []
        Good_Matchups = []
        for strength in Matchup_Strengths:
            if strength <= -1*self.min_matchup_strength:
                Bad_Matchups.append(strength)
            elif strength >= self.min_matchup_strength:
                Good_Matchups.append(strength)
        if len(Bad_Matchups) > 0 and len(Good_Matchups) > 0: #Could switch it to needing 2 more in Good vs Bad (3 Good vs 1 Bad still = Good)
            Matchup_Strength = 0
        elif len(Bad_Matchups) >= 2:
            Matchup_Strength = sum(Bad_Matchups)
        elif len(Good_Matchups) >= 2:
            Matchup_Strength = sum(Good_Matchups)

        return Matchup_Strength

    def Asses_Team_Matchups(self):
        #Here we need to get the value of MatchupStat Adj Factor * Correlation for each of the 6 matchups for team and for opponent
        #Then we see for team if at least 2 show strong value in same direction AND if no others show strong value in opposite direction
        #Repeat for Opponent
        #Then see if Teams and Opponents are: Showing same trend, 1 shows trend and other does not, showing opposite trends, neither shows trend
            #If in same direction, we add the totals, if in oppostie direction, subtract totals, if only 1 then keep the team total, if none then nothing
        #Minimum value of matchup strength will be say 4 (10*0.4)
        self.Matchup_Cols = self.Correlation_Cols
        Team_Stat_Strengths = []
        Opponent_Stat_Strengths = []
        for col in self.Matchup_Cols:
            adj_stat_name = f'{col} EGO Adj'
            cor_stat_name = f'{col} Correlation'
            team_cor = self.Team_Data.loc[f'{cor_stat_name}']
            team_adj = self.Team_Data.loc[f'{adj_stat_name}']
            team_stat_strength = team_adj*abs(team_cor)
            Team_Stat_Strengths.append(team_stat_strength)
            opp_cor = self.Opponent_Data.loc[f'{cor_stat_name}']
            opp_adj = self.Opponent_Data.loc[f'{adj_stat_name}']
            opp_stat_strength = opp_adj*abs(opp_cor)
            Opponent_Stat_Strengths.append(opp_stat_strength)
        print(Team_Stat_Strengths)
        print(Opponent_Stat_Strengths)
        Team_Matchup_Strength = self.Determine_Individual_Matchup_Strength(Team_Stat_Strengths)
        Opp_Matchup_Strength = self.Determine_Individual_Matchup_Strength(Opponent_Stat_Strengths)
        print(Team_Matchup_Strength)
        print(Opp_Matchup_Strength)
        self.Total_Matchup_Strength = self.Determine_Total_Matchup_Strength(Team_Matchup_Strength, Opp_Matchup_Strength)
        self.New_Matchup_Data[0].append(self.Total_Matchup_Strength)
        print(self.Total_Matchup_Strength)
    def Get_Opp_Data(self):
        self.Opp = self.Team_Data["Opponent"]
        self.Opponent_dataset = self.matchup_dataset.loc[(self.matchup_dataset['Team'] == self.Team_Data["Opponent"]) & (self.matchup_dataset['Week'] == self.Team_Data["Week"]) & (self.matchup_dataset['Year'] == self.Team_Data["Year"])]
        opp_index = self.Opponent_dataset.index.values[0]
        self.Opponent_Data = self.matchup_dataset.loc[opp_index]

    def Read_Matchup_Data(self):
        self.matchup_dataset = pd.read_csv(self.Game_Prediction_Data_Path)
        self.Rows = len(self.matchup_dataset.index)
        self.Cols = list(self.matchup_dataset)
        print(self.matchup_dataset.head())
        print(self.matchup_dataset.tail())
        print(self.Rows)
        print(self.Cols)
        self.time.sleep(5)

    def Make_Matchup_Picks(self):
        self.Read_Matchup_Data()
        self.min_matchup_strength = 4
        self.largest_EGO_change = 6 #Most we can change an EGO for the highest matchup strength
        self.New_Matchup_Cols = ["Total Matchup Strength", "Scaled Matchup Strength", "Matchup Adj EGO", "Matchup Adj EGO to Spread Diff", "Matchup Adj Game Pick", "Matchup Adj Pick Rigt"]
        self.New_Matchup_Data = [[] for i in range(len(self.New_Matchup_Cols))]
        for team_index in range(0,self.Rows):
            self.Team_Data = self.matchup_dataset.loc[team_index]
            self.Get_Opp_Data()
            print(self.Team_Data['Team'])
            print(self.Team_Data['Year'])
            self.Asses_Team_Matchups()
            # self.time.sleep(1)
        self.Scale_Matchup_Strengths()
        self.Adjust_EGO_Make_Picks()
        self.Evaluate_Picks()
        self.Save_New_Data()


    ##### HERE IS WHERE WE DO EVERYTHING #####

    def Analyze(self):
        #Get the Matchup Stats
        self.Read_Data()
        self.Setup_Map()
        self.Get_Rows()
        # #Save the dataset with the matchup stats
        self.Save_Correlation_Data()

        #Use the matchup stats to make new EGO/Predictions
        self.Make_Matchup_Picks()
        # self.All_accs = []
        # Cutoffs = []
        # Thresholds = []
        # for threshold in range(8, 40):
        #     self.largest_EGO_change = threshold/4
        #     for cutoff in range(8, 40):
        #         self.min_matchup_strength = cutoff/4
        #         Cutoffs.append(cutoff)
        #         Thresholds.append(threshold)

        #         self.Make_Matchup_Picks()
                
        #         trialdf = pd.DataFrame()
        #         trialdf['Min Matchup Strength'] = Cutoffs
        #         trialdf['Max EGO Adj'] = Thresholds
        #         trialdf['Picking Accuracy'] = self.All_accs
        #         trialdf.to_csv('Matchup Strength Parameters Accuracy', index=False)
        #         print(self.All_accs)

        #Make Data for the NN Models
        self.Save_Model_Data()


# min_week = 6
# min_season = 2006
# MatchupCollector = NFL_DATA_MODEL(min_week, min_season, 'DVOA')
# MatchupCollector.Analyze()