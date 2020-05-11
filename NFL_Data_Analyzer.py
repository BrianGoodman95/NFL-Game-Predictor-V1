from ParsingTools.Repeatale import Data_Collection
from ParsingTools.Repeatale import DVOA_Collector
from ParsingTools.Repeatale import Matchup_Analyzer
from ParsingTools.Repeatale import Data_Evaluation
from ParsingTools.Setup import Directory_setup

min_season = 2019
latest_season = 2019
min_week = 14
latest_week_of_lastest_season = 16
dvoaTypes = ["DVOA", "WDVOA"]


# Set up all the working directories
setup = Directory_setup.Create_Directories()
project_path = setup.project_path

# Acquire all the Raw Statistical Data for teams historically and process it into a standard format
# Create set of standard Stats for each team and opponent for every game
DataCollector = Data_Collection.NFL_DATA_COLLECTER(project_path, latest_week_of_lastest_season)
DataCollector.Do_Stuff()

for DVOA_Type in dvoaTypes:
    # Acquire all the DVOA rating data for teams historically, build a model with it, 
    # Use model to assign an Expected Game Outcome (EGO) and add to the standard format of Data
    DVOACollector = DVOA_Collector.DVOA_DATA(project_path, latest_week_of_lastest_season, latest_season, DVOA_Type)
    DVOACollector.Do_Stuff()

    #Analyze the processed statistical data to look at the matchups between the teams of each game for 6 Main Stats:
        # Passing Offense vs Opp Passing Defense, Passing Defense vs Opp Passing Offense, Rushing Offense vs Opp Rushing Defense, Rushing Defense vs Opp Rushing Offense, Oline vs Opps DLine, Dline vs Opp Oline
    #Assign a game outcome adjustment based on the combined statistical matchup strengths/weakness for each category
    MatchupCollector = Matchup_Analyzer.NFL_DATA_MODEL(project_path, min_week, min_season, DVOA_Type)
    MatchupCollector.Analyze()

    #Visually evaluate the results of the predictions for each game and the accuracy of the Model
    Evalauator = Data_Evaluation.PRREDICTION_DATA_EVALUATOR(project_path, DVOA_Type)
    Evalauator.Do_Stuff()