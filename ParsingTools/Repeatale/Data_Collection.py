import pandas as pd
import time
import os
import requests
from ParsingTools.Repeatale import Data_Proccessing
from ParsingTools.Repeatale import Game_Collection


'''
Each Week                                                        /    
    -Get the data just as before excluding the unecessary data \/                                                 /
    -Need to track all the opponents each team plays and the week they play them in a dataframe for each season \/
    -Can take the existing Get Opponent Part already in use and add the opponent list to this new DF each week \/
    Team    Opp W1  Opp W2 ... 
    ARZ     Det     Bal
                                                                                                                               /
    -Proccess just as before up to the differing part and don't add opponent stats to dataframe yet (just the opponent name) \/
    -Make stats per game excluding (Basic Season, Betting and Result Stats) \/                                       /
    -Change every stat excluding (Year, Games, Spread and Result Stats) to a ranking from 1-#teams up to that week \/
    
    -For the big 6 stats:
        Need Teams' For and Against Rank as of the week
        Need Teams' Opponents up to the week and the week # they played
            Need these teams' for/agaisnt ranks as of the week they played - then take average
                From the Matchup DataFrame (above) determine the teams and order (week number) - skip blanks
                From the Curent Data frame (up to current week) locate each team and their rank for the given week
        Discount Teams For and Against Rank as per the Template Doc
        Need Opponents' For and Against Rank as of the week
        Need Opponents' Opponents up to the week and the week # they played
            Need these teams' for/agaisnt ranks as of the week they played - then take average
                From the Matchup DataFrame (above) determine the teams and order (week number) - skip blanks
                From the Curent Data frame (up to current week) locate each team and their rank for the given week
        Discount Opponents For and Against Rank as per the Template Doc
    -For all ranked stats:
        Take difference between Teams' for and Opponents Against (if for/against applies - else do Teams Rank - Opponents Rank) as described in template

'''

class NFL_DATA_COLLECTER():

    def __init__(self, project_path, latest_week):
        self.time = time
        #Dictionary of stat names to use in search on database, corresponding name in html table and name to be used in dataframe for the collumn
        #Some search names have muliple table collumns to be collected
        #What is important is the order of html table name and dataframe names matches so will rename the collumn to proper name
        self.Search_Stat_Dict = {
            #Only collecting Up-To-week data for Result Stats (week 1-5, week 1-6 etc)
            "Basic Season Stats":{'Search Name': ['points_diff'],
                            'Number of Headers': [4],
                            'Table Headers': ['Year', 'G', 'W-L%', 'PD'],
                            'My Headers': ['Year', 'Games Played', 'W-L%', 'Point Differential']
            },
            "Betting Stats":{'Search Name': ['vegas_line'],
                            'Number of Headers': [2],
                            'Table Headers': ['Spread', 'Over/Under'],
                            'My Headers': ['Spread', 'Over/Under']
            },
            "Basic Off/Def Stats":{'Search Name': ['pass_rating', 'pass_sacked_opp', 'rush_yds_per_att', 'rush_yds_per_att_opp', 'pass_yds', 'pass_yds_opp'],
                            'Number of Headers': [1, 1, 2, 2, 2, 2],
                            'Table Headers': ['Sk', 'Sk', 'Y/A', 'Yds', 'Y/A', 'Yds', 'Rate', 'Yds', 'Rate', 'Yds'],
                            'My Headers': ['Sacks Against', 'Sacks For', 'Yards per Rush For', 'Rushing Yards For', 'Yards per Rush Against', 'Rushing Yards Against', 'Passer Rating For', 'Passing Yards For', 'Passer Rating Against', 'Passing Yards Against']
            },
            "Advanced Stats":{'Search Name': ['time_of_poss', 'comb_penalties', 'third_down_pct', 'third_down_pct_opp', 'tot_yds_diff'],
                            'Number of Headers': [3, 2, 1, 1, 2],
                            'Table Headers': ['ToP', "Y/P", 'DY/P', 'Yds', 'OppYds', '3D%', 'Opp3D%', 'Tot', 'TO'],
                            'My Headers': ['Time of Poss', 'Yards For per Play', 'Yards Against per Play', 'Penalty Yards Committed', 'Penalty Yards Received', '3rd Down %_For', '3rd Down % Against', 'Total Yard Margin', 'Turnover Margin']
            },
            #Only collecting by-week data for Result Stats (week 1, week 2 etc)
            "Results Stats":{'Search Name': ['points_diff'],
                            'Number of Headers': [1],
                            'Table Headers': ['PD'],
                            'My Headers': ['Game Scoring Margin']
            }
        }
        self.min_week = 2
        self.max_week = 16
        self.last_season_max_week = latest_week
        self.min_season = 2006
        self.max_season = 2019
        self.Read_Previous_Data = True
        #Define Years which Data is being collected for
        self.All_Search_Seasons = [n for n in range(self.min_season,self.max_season+1)]
        #Defien Weeks which Data is being collected for
        self.All_Search_Weeks = [n for n in range(self.min_week,self.max_week+1)] #Stats heading into Weeks 2 through 17 games

        #Lists and DFs
        self.All_Seasons_DF = pd.DataFrame()
        self.Season_DF = pd.DataFrame()
        self.Week_DF = pd.DataFrame()
        self.Opponent_Tracking_DF = pd.DataFrame()
        self.All_Seasons_DF_List = []
        self.All_Weeks_DF_List = []
        self.All_Cleaned_Weeks_DF_List = []
        self.Blank_Positions = []
        
        #Make save locations
        # self.project_path = 'E:/Project Stuff/Data Analysis Stuff/V5/NFL Data V1/Raw Data'
        self.project_path = project_path
        self.data_path = f'{project_path}/Raw Data'
        # try:
        #     os.mkdir(self.data_path)
        # except FileExistsError:
        #     print('Directory already made')

    def Make_Folder(self, new_path):
        path_exists = False
        try:
            os.mkdir(new_path)
        except:
            print('folder exists')
            path_exists = True
        return path_exists
    
    def Concat_and_Save(self, df_list, path):
        df = pd.concat(df_list) #Concat the list of dfs into a season df
        df.to_csv(path, index=False)
        return df


    def Track_All_Oppononents(self):
        self.Opponent_Tracking_DF[f'Week {self.week} Opp'] = self.Week_DF['Opponent'].tolist()


    def Get_DF(self):
        #Get the df from the website
        html = requests.get(self.URL_BASE).content
        df_list = pd.read_html(html, header=1, index_col=None)
        df = df_list[-1]
        #Sort the df by team name (so lines up all stats (since team names the same each season))
        df = df.sort_values(by=['Year', 'Tm'])
        #Cut off last row which is the headers again
        df = df.head(len(df['Tm'])-1)
        lastrow_df = df.tail(1)
        
        return df

    def Get_Teams_List(self, season):
        #Get teams in alphabetical order for the season
        df = self.Get_DF()
        team_list = df['Tm'].tolist()
        return team_list

    def Match_By_Teams(self, These_Teams):
        #Match team position to be the same for each week of the season
        self.Blank_Positions = []
        #print("starting matching")
        #print(These_Teams)
        for team_pos in range(0, len(self.All_Season_Teams)):
            try:
                if These_Teams[team_pos] != self.All_Season_Teams[team_pos]:
                    These_Teams.insert(team_pos, self.All_Season_Teams[team_pos])
                    self.Blank_Positions.append(team_pos)
            except IndexError:
                These_Teams.append(self.All_Season_Teams[team_pos])
                self.Blank_Positions.append(team_pos)
        # print(self.All_Season_Teams)
        # print(These_Teams)

    def Setup_This_Week_Stats(self):
        if self.season == 2019:
            This_Week_Game_Info = Game_Collection.This_Week_Parsing(self.season, self.week, self.min_week, self.All_Season_Teams, self.season_path)
            This_Week_Teams, Sched_Week_DF = This_Week_Game_Info.Setup_This_Week_Data()
            if self.week == self.min_week:
                self.last_week_teams = This_Week_Teams
            #Get the location of missing teams relative to the list of all teams for the season
            self.Match_By_Teams(This_Week_Teams)
        else:
            This_Week_Teams = self.All_Season_Teams
        #Add the team list for this week (should be same for every week of the season)
        self.Week_DF['Team'] = This_Week_Teams
        self.Week_DF['Week'] = [self.week for i in range(len(This_Week_Teams))]
        #Get the information about this week's games
        if self.season == 2019:
            This_Week_Game_Info = Game_Collection.This_Week_Stats(self.last_week_teams, self.Week_DF, Sched_Week_DF)
            self.Week_DF = This_Week_Game_Info.Get_Info()
            self.last_week_teams = This_Week_Teams
        print(self.Week_DF)

    def Collect_Data(self, stat_type):
        #Get the df from the website
        if self.season == 2019 and self.week == self.max_week and ("Results Stats" in stat_type or "Betting Stats" in stat_type):
            print("Dont Get DF")
        else:
            sorted_df = self.Get_DF()
            #Get df headers to looks for
            df_headers = list(sorted_df)
        #print(df_headers)
        next_df_header_pos = 0
        for header_pos in range(self.first_available_stat_header, self.first_available_stat_header + self.Search_Stat_Dict[stat_type]['Number of Headers'][self.search_stat_pos]):
            print(self.Search_Stat_Dict[stat_type]['Table Headers'][header_pos])
            if self.season == 2019 and self.week == self.max_week and ("Results Stats" in stat_type or "Betting Stats" in stat_type):
                if "Spread" in self.Search_Stat_Dict[stat_type]['My Headers'][header_pos]:
                    latest_spreads_path = f'{self.project_path}/Templates/Latest Week Spreads.csv'
                    latest_spreads_df = pd.read_csv(latest_spreads_path)
                    self.Week_DF[self.Search_Stat_Dict[stat_type]['My Headers'][header_pos]] = latest_spreads_df["Spread"].tolist()
                else:
                    self.Week_DF[self.Search_Stat_Dict[stat_type]['My Headers'][header_pos]] = [0 for i in range(len(self.All_Season_Teams))]
            else:
                for df_header_pos in range(next_df_header_pos, len(df_headers)):
                    if self.Search_Stat_Dict[stat_type]['Table Headers'][header_pos] == df_headers[df_header_pos]:
                        self.listed_stat = sorted_df[self.Search_Stat_Dict[stat_type]['Table Headers'][header_pos]].tolist()
                        #print(len(self.listed_stat))
                        #Normalize the team list to all teams for the season
                        for pos in self.Blank_Positions:
                            self.listed_stat.insert(pos, "")
                        #print(len(self.listed_stat))
                        self.Week_DF[self.Search_Stat_Dict[stat_type]['My Headers'][header_pos]] = self.listed_stat
                        print(self.Week_DF.head())
                        #next_df_header_pos = df_header_pos
                        break
        #self.time.sleep(3)

    def Setup_Data_Search(self):
        for self.season in self.All_Search_Seasons:
            print(f'Season: {self.season}')
            if self.season == self.max_season:
                # self.max_week = self.last_season_max_week ##IF WANT THE LAST WEEK NOT TOO LOOK FOR SCORES (SINCE DON"T EXIST YET)
                self.max_week = self.last_season_max_week+1 ##IF WANT THE LAST WEEK TO LOOK FOR SCORES
                self.All_Search_Weeks = [n for n in range(self.min_week,self.max_week+1)] #Stats heading into Weeks 2 through 10 games
                if self.max_week == self.last_season_max_week+1: #Get rid of the last week in this case where we added it to the max week
                    del self.All_Search_Weeks[-1] 
            #Make the folder for data
            self.season_path = f'{self.data_path}/{self.season}'
            self.Make_Folder(self.season_path)

            #Clear the season df and list of all weeks
            self.Season_DF = pd.DataFrame()# self.Season_DF.iloc[0:0]
            self.Opponent_Tracking_DF = pd.DataFrame()
            self.All_Weeks_DF_List = []
            self.All_Cleaned_Weeks_DF_List = []

            #Get all teams list for that season
            self.URL_BASE = f'https://www.pro-football-reference.com/play-index/tgl_finder.cgi?request=1&match=single&year_min={self.season}&year_max={self.season}&game_type=R&game_num_min=0&game_num_max=99&week_num_min=1&week_num_max=17&temperature_gtlt=lt&league_id=NFL&c5val=1.0&order_by=points_diff'
            self.All_Season_Teams = self.Get_Teams_List(self.season)

            #Add the Season Teams to the Opponent Tracker
            self.Opponent_Tracking_DF["Team"] = self.All_Season_Teams

            for self.week in self.All_Search_Weeks:
                print(f'Week: {self.week}')
                #Make the folder for data
                self.week_path = f'{self.season_path}/Week {self.week}'
                Path_Exists = self.Make_Folder(self.week_path)
                if Path_Exists and self.season != self.max_season and self.Read_Previous_Data == True:
                    #Read csv that already exists
                    self.Week_DF = pd.read_csv(f'{self.week_path}/Week {self.week} Opponent Combined Data.csv')
                    self.All_Weeks_DF_List.append(self.Week_DF)
                else:
                    #Get Game Info for this week
                    #Clear the week_df
                    self.Week_DF = pd.DataFrame()
                    self.Setup_This_Week_Stats()
                    #Get Stat Info for this week
                    for stat_type in self.Search_Stat_Dict:
                        print(stat_type)
                        #Reset the first header to 0
                        self.search_stat_pos = 0
                        
                        for search_name in range(0, len(self.Search_Stat_Dict[stat_type]['Search Name'])):
                            stat = self.Search_Stat_Dict[stat_type]['Search Name'][search_name]
                            print(stat)
                            #Define the headers to look at for this stat
                            if self.search_stat_pos == 0:
                                self.first_available_stat_header = 0
                            else:
                                #for pos in range(1, self.search_stat_pos):
                                self.first_available_stat_header += self.Search_Stat_Dict[stat_type]['Number of Headers'][self.search_stat_pos-1]

                            if "Results Stats" in stat_type or "Betting Stats" in stat_type: #Only that week
                                self.URL_BASE = f'https://www.pro-football-reference.com/play-index/tgl_finder.cgi?request=1&match=single&year_min={self.season}&year_max={self.season}&game_type=R&game_num_min=0&game_num_max=99&week_num_min={self.week}&week_num_max={self.week}&temperature_gtlt=lt&league_id=NFL&c5val=1.0&order_by={stat}'
                                week_using = self.week
                                # if self.season == 2019 and self.week == self.max_week:
                                #     week_using = self.week-1 #Not doing betting stats for last week yet
                                #     self.URL_BASE = f'https://www.pro-football-reference.com/play-index/tgl_finder.cgi?request=1&match=single&year_min={self.season}&year_max={self.season}&game_type=R&game_num_min=0&game_num_max=99&week_num_min={self.week-1}&week_num_max={self.week-1}&temperature_gtlt=lt&league_id=NFL&c5val=1.0&order_by={stat}'
                            else: #Week 1 to that Week
                                self.URL_BASE = f'https://www.pro-football-reference.com/play-index/tgl_finder.cgi?request=1&match=single&year_min={self.season}&year_max={self.season}&game_type=R&game_num_min=0&game_num_max=99&week_num_min=1&week_num_max={self.week-1}&temperature_gtlt=lt&league_id=NFL&c5val=1.0&order_by={stat}'
                                week_using = self.week-1
                            #Get the list of teams playing this week
                            # if self.season != 2019 or self.week != self.max_week:
                            if self.season == 2019 and self.week == self.max_week and ("Results Stats" in stat_type or "Betting Stats" in stat_type):
                                print("Don't Match Teams") 
                            else:
                                self.This_Stat_Teams = self.Get_Teams_List(self.season)
                                #Get the location of missing teams relative to the list of all teams for the season
                                self.Match_By_Teams(self.This_Stat_Teams)
                            #Add the team list for this week (should be same for every week of the season)
                            # self.Week_DF['Team'] = self.This_Stat_Teams
                            # self.Week_DF['Week'] = [self.week for i in range(len(self.This_Stat_Teams))]
                            
                            #Collect the data for this week
                            self.Collect_Data(stat_type)
                            self.search_stat_pos+=1

                    # #Get the information about this week's games ###To Do: MIGHT NEED TO ADD THE self.Week_DF RETURN FOR IT TO ACTUALLY UPDATE THE DATA
                    if self.season != 2019:
                        #Get the information about this week's games
                        This_Week_Game_Info = Game_Collection.HISTORICAL_PARSING(self.Week_DF, self.This_Stat_Teams, self.All_Season_Teams, self.season, self.week)
                        This_Week_Game_Info.Do_Game_Stuff()
                    This_Week_Game_Info = Game_Collection.HISTORICAL_PARSING(self.Week_DF, self.This_Stat_Teams, self.All_Season_Teams, self.season, self.week)
                    This_Week_Game_Info.Do_Betting_Stuff()
                    #NEW GAME STATS HERE
                    # This_Week_Game_Info = Game_Collection.This_Week_Parsing(self.Week_DF, self.This_Stat_Teams, self.All_Season_Teams, self.season, self.week)
                    # This_Week_Game_Info.Do_Stuff()
                    #Proccess the data
                    Proccess_Stats = Data_Proccessing.PROCCESS_DATA(self.week, self.season, self.Week_DF, self.week_path, self.All_Cleaned_Weeks_DF_List, self.All_Weeks_DF_List)
                    self.All_Weeks_DF_List, self.All_Cleaned_Weeks_DF_List = Proccess_Stats.Do_Stuff()

                    # self.Track_All_Oppononents()
                    #To Do Still
                    # 3. Use this version to make model verison (remove even more columns)
                    # 4. Find new way to get Game Stats (Home Team, Opponent) For Current Week
              
            #Combine all weeks into a Season DF and save it
            season_excel_name = f'{self.season_path}/{self.season} Results.csv'
            self.Season_DF = self.Concat_and_Save(self.All_Weeks_DF_List, season_excel_name)
            self.All_Seasons_DF_List.append(self.Season_DF) #Put each season df in a list
            #Save the Season Opponents DF
            season_opp_excel_name = f'{self.season_path}/{self.season} Opponents.csv'
            self.Opponent_Tracking_DF.to_csv(season_opp_excel_name, index=False)

        project_excel_name = f'{self.data_path}/All Seasons Results.csv'
        self.Concat_and_Save(self.All_Seasons_DF_List, project_excel_name)

    def Do_Stuff(self):
        self.Setup_Data_Search()


# NFL_DATA = NFL_DATA_COLLECTER(latest_week)
# NFL_DATA.Do_Stuff()