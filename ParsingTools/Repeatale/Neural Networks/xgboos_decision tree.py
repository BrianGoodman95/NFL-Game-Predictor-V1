import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

project_path = 'E:/Project Stuff/Data Analysis Stuff/V5/NFL Data V1'
Model_Data_Path = f'{project_path}/Model Data/Regression WDVOA Model Data.csv'
Save_Picks_Path = f'{project_path}/Prediction Data/Regression Prediction Data.csv'
df = pd.read_csv(Model_Data_Path)
df.drop(['Spread Result Class'], axis=1, inplace=True)
used_columns = [c for c in df.columns if c != 'EGO To Result Diff']

# Split the data into train and test dataframes.
X_train, X_test, y_train, y_test = train_test_split(
    df[used_columns],
    df['EGO To Result Diff'],
    test_size=0.2, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

param = dict(
            eta= 0.3,
            silent=1,
            scale_pos_weight=0.1,
            learning_rate=0.2,
            colsample_bytree=0.4,
            subsample=0.8,
            objective='reg:squarederror',
            n_estimators=2000,
            reg_alpha=0.3,
            max_depth=4,
            gamma=3)

num_round = 2000  # the number of training iterations

bst = xgb.train(param, dtrain, num_round)

# Generate predictions vs real values dataframe and save it to CSV.
preds = bst.predict(dtest)
predictions = []
for i in range(len(preds)):
    predictions.append({'Predicted value': preds[i], 'EGO To Result Diff': y_test.iloc[i]})

df = pd.DataFrame(predictions)
df.to_csv(Save_Picks_Path, index=False)
