import pathlib
import os

class Create_Directories():
    def __init__(self):
        self.project_path = pathlib.Path().absolute()
        self.Data_folders = ['Raw Data', 'Raw Data/WDVOA DATA', 'Total Data', 'Total Data/Evaluations', 'ML Data', 'ML Data/Model Data',  'ML Data/Prediction Data',  'ML Data/Train_Test_Data']
        for folder_path in self.Data_folders:
            try:
                os.mkdir(folder_path)
                print(f'Created Directory {folder_path}')
            except FileExistsError:
                print(f'Directory {folder_path} already exists')