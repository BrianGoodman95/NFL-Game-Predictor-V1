import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

class Plots():
    def __init__(self):
        self.something = 1

    def Season_Results(self, Prediction_Stats, ycols):
        fig = go.Figure()
        for col in ycols:
            if 'Season' in col: 
                fig.add_trace(go.Scatter(x=list(Prediction_Stats['Week']), y=list(Prediction_Stats[col]),
                    # text=list(Prediction_Stats[col]),
                    # textposition='auto',
                    name=col))
            else:
                fig.add_trace(go.Bar(x=list(Prediction_Stats['Week']), y=list(Prediction_Stats[col]),
                    text=list(Prediction_Stats[col]),
                    textposition='auto',
                    name=col))
        fig.update_layout(dict(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'), title={'text': "2020 Betting Results",'x':0.45,'xanchor': 'center','yanchor': 'top'}, xaxis=dict(title="Week"), yaxis=dict(title="Result"))
        fig.update_xaxes(dtick=1)
        return fig

    def Historical_Results(self, Data, xcol):
        # Want moving average of model accuracy to be on the yaxis for xaxis of EGO, Spread and EGO/Spread Diff
        # Want just model accuracy to be on y-axis for Season, Week and Teams on the xaxis.
        # Want filter for each of season, week and team as well (only when that variable isn't selected as the xaxis ideally)
        moving_average_xcols = ['Spread to EGO Diff', 'WDVOA Delta', 'EGO', 'Betting Spread', 'Scoring Margin']
        if xcol in moving_average_xcols: #Get the moving average DF first
            results_df = self.Moving_Average_Accuracy(Data, xcol)
        else:
            results_df = self.Discrete_Accuracy(Data, xcol)
        fig = (px.scatter(results_df, x=xcol, y="Betting Accuracy", color="Analysis Name").update_traces(mode='lines+markers', line=dict(width=1), marker=dict(size=3),))
        fig.update_layout(dict(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'))
        return fig

    def Discrete_Accuracy(self, Data, xcol):
        df = pd.DataFrame()
        xvals = list(set(Data[xcol].tolist()))
        xvals.sort()
        accs = []
        legend_data = []
        for val in xvals:
            val_data = Data.loc[(Data[xcol] == val) & (Data['Make Pick'] == 1)] #Get the Picks Made for that Season/Week/Team
            picks = len(val_data['Make Pick']) #Total Number of Picks Made
            picks_right = sum(list(val_data['Pick Right'])) #All the 1s for Pick being right
            accs.append(round((picks_right/picks),2))
            legend_data.append(f'Model Accuracy by {xcol}')
        df[xcol] = xvals
        df['Betting Accuracy']= accs
        df['Analysis Name'] = legend_data
        print(df)
        return df

    def Moving_Average_Accuracy(self, Data, xcol):
        #Defien the columns needed for this evaluation
        evaluation_df = Data[['Pick Right', xcol]].sort_values(by=xcol, ascending=False)
        #Initialize Dataframe and Lists to Store REsults
        results_df = pd.DataFrame()
        accuracy_data = []
        avg_ego_data = []
        legend_data = []
        moving_avg = int(len(Data[list(Data)[0]])/10) #For full dataset of 4200 values measn 420 moving average

        #Get Betting accuracy by EGO to Spread Difference for each moving average window
        legend = f'Model Accuracy by {xcol}'
        ego_data = list(evaluation_df[xcol])
        prediction_data = list(evaluation_df['Pick Right'])
        for pos, pred in enumerate(prediction_data):
            if pred == 0:
                prediction_data[pos]=-1
            elif pred == 1:
                prediction_data[pos] = 1
            else:
                prediction_data[pos] = 0
        counter=0
        for dp in range(0,len(ego_data)-moving_avg):
            avg_egoDiff = sum(ego_data[dp:(dp+moving_avg)])/moving_avg
            avg_acc = ((sum(prediction_data[dp:(dp+moving_avg)])/2)+(moving_avg/2))/moving_avg
            avg_ego_data.append(avg_egoDiff)
            accuracy_data.append(avg_acc)
            counter+=1
        legend_data += [legend for i in range(counter)]
            
        #Store results into final dataframe
        results_df['Betting Accuracy'] = accuracy_data
        results_df[xcol] = avg_ego_data
        results_df['Analysis Name'] = legend_data
        print(results_df)
        return results_df