import tensorflow as tf
import keras
# from tensorflow import keras.layers
from sklearn.metrics import precision_score
from keras.models import model_from_yaml
from keras.models import model_from_json
from keras.models import load_model
import pandas as pd
import numpy as np
import random
import pickle
import time

def NN(best_acc, best_seed):
    # Load the data and drop the answer column. We will keep this column in a variable
    # because we want to add it back to the outpit csv file.
    project_path = 'E:/Project Stuff/Data Analysis Stuff/V5/NFL Data V1'
    Model_Data_Path = f'{project_path}/Model Data/Test Classification WDVOA Model Data.csv'
    Save_Picks_Path = f'{project_path}/Prediction Data/Classification Prediction Data.csv'
    df = pd.read_csv(Model_Data_Path)
    answer_column = df['Spread Result Class']
    pick_column = df['Game Pick']
    df.drop(['Spread Result Class'], axis=1, inplace=True)
    # df.drop(['Game Pick'], axis=1, inplace=True)
    # Define the label column.
    label_column = 'EGO Pick Correct'

    # Split the dataset into test and train parts. In this case we will test on the
    # first 500 rows of data.
    # train_test_split = 500
    # df_train = df.iloc[train_test_split:]
    # df_test = df.iloc[:train_test_split]

    #Get a random 80% of the data for training
    seed = random.randrange(0, 100)
    df_train = df.sample(frac=0,random_state=seed)
    df_test = df.drop(df_train.index)
    # df_train.to_csv(f'{project_path}/Model Data/Train Classification WDVOA Model Data.csv', index=False)
    # df_test.to_csv(f'{project_path}/Model Data/Test Classification WDVOA Model Data.csv', index=False)
    df_test_indicies = df_test.index.values.tolist()
    print(df_test_indicies)
    print(len(df_test_indicies))
    # time.sleep(5)

    # Create the model building function.
    def build_model():
        model = keras.Sequential([
            keras.layers.Dense(100, activation='relu', input_shape=[len(df.columns)-1]),
            keras.layers.Dropout(0.01, noise_shape=None, seed=None),
            keras.layers.Dense(100, activation='relu'),
            keras.layers.Dropout(0.01, noise_shape=None, seed=None),
            keras.layers.Dense(1)
        ])

        optimizer = keras.optimizers.RMSprop(0.001)
        loss = 'mse'
        model.compile(loss=loss,
                    optimizer=optimizer,
                    metrics=['mae', 'mse'])
        return model

    model = load_model('model2.h5')

    # # Make New Model
    # model = build_model()
    # # Train the model for given number of epochs.
    # EPOCHS = 15
    # X = df_train[[c for c in df_train.columns if c != label_column]].to_numpy()
    # y = df_train[label_column].to_numpy()
    # model.fit(X, y, epochs=EPOCHS, validation_split=0.2, verbose=1)

    # Print predictions compared to targets.
    number_of_predictions = 10

    # print('{} randomly picked predictions:'.format(number_of_predictions))
    # for _ in range(number_of_predictions):
    #     # index = random.randint(0, train_test_split)
    #     index = random.randint(0, len(df_test_indicies))
    #     predicion_x = df_test[[c for c in df_test.columns if c != label_column]].iloc[index].to_numpy().reshape(1, len(df.columns)-1)
    #     prediction = model.predict(predicion_x)
    #     target = df_test[label_column].iloc[index]
    #     print('Prediction: {} | Target: {}'.format(prediction[0][0], target))

    # Calculate the models score.
    y_pred = model.predict(df_test[[c for c in df_test.columns if c != label_column]].to_numpy())
    binary_y_pred = list(map(lambda x: 1 if x > 0 else 0, y_pred))
    binary_y_test = list(map(lambda x: 1 if x > 0 else 0,df_test[label_column].to_numpy()))
    score = precision_score(binary_y_pred, binary_y_test)
    print('Model precision score: {}'.format(score))

    # Generate output.
    df = df_test
    output_filename = 'classification_results_01.csv'
    y_pred = model.predict(df[[c for c in df.columns if c != label_column]].to_numpy())
    pred_confidence = -0.4
    y_pred = list(map(lambda x: 1 if x > pred_confidence else -1, y_pred))

    Results = [0 for i in range(2)]
    y_test = df[label_column].tolist()
    for game in range(0, len(y_pred)):
        if y_pred[game]*y_test[game] >= 0:
            Results[0] += 1
        else:
            Results[1] += 1
    accuracy = Results[0]/(Results[0]+Results[1])
    print('Model Accuracy: {}'.format(accuracy))

    df['Model Output'] = y_pred
    # df['EGO Pick'] = pick_column
    df['Spread Result Class'] = answer_column
    # df['EGO Prediction Right'] = df.apply(lambda row: 1 if row['Game Pick']*row['Spread Result Class'] >= 0 else -1, axis=1) #If the pick and the result are not opposite signs
    df['New Prediction'] = pick_column * df['Model Output']
    df['New Prediction Right'] = df.apply(lambda row: 1 if row['New Prediction']*row['Spread Result Class'] >= 0 else -1, axis=1)
    df.to_csv(Save_Picks_Path, index=False)
    accuracy_cols = ['EGO Pick Correct', 'New Prediction Right']
    old_acc = sum(df[accuracy_cols[0]])
    print(old_acc)
    new_acc = sum(df[accuracy_cols[1]])
    print(new_acc)
    if accuracy > 0.53 and accuracy > best_acc:
        # save model and architecture to single file
        model.save("model2.h5")
        print("Saved model to disk")
        best_acc = accuracy
        best_seed = seed
    print(best_seed)
    time.sleep(2)
    return best_acc, best_seed

start_time = time.time()
best_acc = 0
best_seed = 0
while time.time() - start_time < 60*60: #1 hour
    best_acc, best_seed = NN(best_acc, best_seed)

