import pathlib
import os

class Create_Directories():
    def __init__(self):
        dir_path = pathlib.Path().absolute()
        Data_Path = "Data"
        self.project_path = f'{str(dir_path)}/{Data_Path}'
        try:
            os.mkdir(Data_Path)
        except FileExistsError:
            pass
        self.Data_folders = ['Raw Data', 'Raw Data/Stat_Based', 'Raw Data/DVOA_Based', 'Total Data', 'Total Data/Evaluations', 'ML Data', 'ML Data/Model Data',  'ML Data/Prediction Data',  'ML Data/Train_Test_Data']
        for folder_path in self.Data_folders:
            try:
                os.mkdir(f'{Data_Path}/{folder_path}')
                print(f'Created Directory {folder_path}')
            except FileExistsError:
                pass
                # print(f'Directory {folder_path} already exists')

# /Users/briangoodman/Documents/GitHub/NFL-Game-Predictor-V1/Data/Raw Data/All Seasons Results.csv