import Data_Collection
import DVOA_Collector
import Matchup_Analyzer
import Data_Evaluation

min_week = 6
latest_week = 16
min_season = 2006
latest_season = 2019
dvoaTypes = ["DVOA", "WDVOA"]

# DataCollector = Data_Collection.NFL_DATA_COLLECTER(latest_week)
# DataCollector.Do_Stuff()

for DVOA_Type in dvoaTypes:
    DVOACollector = DVOA_Collector.DVOA_DATA(latest_week, latest_season, DVOA_Type)
    DVOACollector.Do_Stuff()

    MatchupCollector = Matchup_Analyzer.NFL_DATA_MODEL(min_week, min_season, DVOA_Type)
    MatchupCollector.Analyze()

    Evalauator = Data_Evaluation.PRREDICTION_DATA_EVALUATOR(min_week, min_season, DVOA_Type)
    Evalauator.Do_Stuff()