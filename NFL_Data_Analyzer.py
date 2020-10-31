from Analysis_Tools.Parsers import Data_Collection
from Analysis_Tools.Parsers import DVOA_Collector
from Analysis_Tools.Parsers import Matchup_Analyzer
from Analysis_Tools.Parsers import Data_Evaluation
from Analysis_Tools.Parsers import Game_Predictor
from Analysis_Tools.Setup import Directory_setup

# Set up all the working directories
setup = Directory_setup.Create_Directories()
project_path = setup.project_path


'''
Want 2 modes
    Historical Model - DONE (Use as is)
    1. Collect the Raw Game and DVOA Data up until 2020, build model, make predictions and evaluate historically
        Need to modify such that the parsers can accept any changes necessary for the new "This Week" Part
    
    This Week Predictions
    2. Collect ame Info Data (Team, Opponent, Week, Year, Spread, Home/Away, Off bye, short week) and DVOA Data for the week and use model to make EGO predictions for this week. Output in nice format
        Need to get current spreads - Done
        Need to modify collectors such that only get Game Info Data for un-saved weeks
        Need to get 2020 DVOA data working
        Need to get good output format
        Need to get way to evaluate how its going

'''

Predict_Current_Games = True
Update_HistoricalData_Model = False

if Predict_Current_Games:
    '''
    MODE 2 - GET CURRENT DATA, MAKE PREDICTION WITH MODEL, EVALUATE
    '''
    season = 2020
    week = 8

    GP = Game_Predictor.NFL_Game_Predictor(project_path, week, season)
    GP.Do_Stuff()

    # for DVOA_Type in dvoaTypes:
    #     # Acquire all the DVOA rating data for teams historically, build a model with it, 
    #     # Use model to assign an Expected Game Outcome (EGO) and add to the standard format of Data
    #     DVOACollector = DVOA_Collector.DVOA_DATA(project_path, current_week, current_season, DVOA_Type)
    #     DVOACollector.Do_Stuff()


if Update_HistoricalData_Model:
    '''
    MODE 1 - GET HISTORICAL DATA, BUILD/EVALUATE MODEL
    '''
    mode = 'Historical'
    min_season = 2006
    min_week = 6
    latest_week_of_lastest_season = 16
    dvoaTypes = ["DVOA", "WDVOA"]

    # Acquire all the Raw Statistical Data for teams historically and process it into a standard format
    # Create set of standard Stats for each team and opponent for every game
    DataCollector = Data_Collection.NFL_DATA_COLLECTER(project_path, mode)#, latest_week_of_lastest_season, latest_season)
    DataCollector.Do_Stuff()

    for DVOA_Type in dvoaTypes:
        # Acquire all the DVOA rating data for teams historically, build a model with it, 
        # Use model to assign an Expected Game Outcome (EGO) and add to the standard format of Data
        DVOACollector = DVOA_Collector.DVOA_DATA(project_path, latest_week_of_lastest_season, 2019, DVOA_Type)
        DVOACollector.Do_Stuff()

        #Analyze the processed statistical data to look at the matchups between the teams of each game for 6 Main Stats:
            #Passing Offense vs Opp Passing Defense, Passing Defense vs Opp Passing Offense, Rushing Offense vs Opp Rushing Defense, Rushing Defense vs Opp Rushing Offense, Oline vs Opps DLine, Dline vs Opp Oline
        #Assign a game outcome adjustment based on the combined statistical matchup strengths/weakness for each category
        MatchupCollector = Matchup_Analyzer.NFL_DATA_MODEL(project_path, min_week, min_season, DVOA_Type)
        MatchupCollector.Analyze()

        #Visually evaluate the results of the predictions for each game and the accuracy of the Model
        Evalauator = Data_Evaluation.PRREDICTION_DATA_EVALUATOR(project_path, DVOA_Type)
        Evalauator.Do_Stuff()

