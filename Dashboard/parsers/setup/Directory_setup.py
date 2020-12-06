import pathlib
import os

class Create_Directories():
    def __init__(self):
        dir_path = pathlib.Path().absolute()
        Data_Path = "data"
        self.project_path = f'{str(dir_path)}/{Data_Path}'
        try:
            os.mkdir(Data_Path)
        except FileExistsError:
            pass
        self.Data_folders = ['raw data', 'compiled', 'compiled/evaluations']
        for folder_path in self.Data_folders:
            try:
                os.mkdir(f'{Data_Path}/{folder_path}')
                print(f'Created Directory {folder_path}')
            except FileExistsError:
                pass
                # print(f'Directory {folder_path} already exists')
