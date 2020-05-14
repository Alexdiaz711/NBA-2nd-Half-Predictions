# This script runs a betting simulation on the last 2 seasons, based on the cost/benefit 
# analysis performed on the previous three seasons
# Simulation was performed using both the "bet every home team" stategy, and the strategy 
# utilizing the algorithm.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

plt.style.use('default')
font = {'weight': 'bold',
        'size':   12}
plt.rc('font', **font)



# Defining a funtion to be used in this script.
def run_sim(class_threshold, betting_threshold):
    '''
    Uses the provided classification and betting thresholds to generate the
    results of a betting simulation for the last two seasons of NBA games.

    Parameters
    ----------
    class_threshold : Float between 0.0 and 1.0 (both inclusive)
    Threshold for which targets will be classified as positive
    when the prediction probability is higher.

    betting_threshold : Non-Negative Integer
    Threshold for which all bets will be ignored when the winning
    payouts is lower.

    Returns
    -------
    games : numpy array of shape (N,)
        1-D numpy array containing and index of games 0 - N
    cum_profit : List of length N
        List containing the cumulative profit for the simulation over the two 
        seasons 
    '''
    sim = predict_df.copy()
    sim['sim predict H>A'] = sim['sim predict proba H>A'].apply(lambda x: 1 if x > class_threshold else 0)
    sim['sim push'] = (sim['2-Half H-A'] == 0).astype(int)
    sim['sim tp'] = ((sim['sim predict H>A'] == sim['2-Half H>A']) & (sim['sim predict H>A'] == 1)).astype(int)
    sim['sim tn'] = ((sim['sim predict H>A'] == sim['2-Half H>A']) & (sim['sim predict H>A'] == 0)).astype(int)
    sim['sim fp'] = ((sim['sim predict H>A'] != sim['2-Half H>A']) & (sim['sim predict H>A'] == 1)).astype(int)
    sim['sim fn'] = ((sim['sim predict H>A'] != sim['2-Half H>A']) & (sim['sim predict H>A'] == 0)).astype(int)
    conditions = [(sim['sim push'] == 1), (sim['sim tp'] == 1), (sim['sim tn'] == 1), (sim['sim push'] == 0)]
    choices = [0.00, sim['tp wins'], sim['tn wins'], -100.00]
    sim['profit'] = np.select(conditions, choices).round(2)
    temp_df1 = sim[sim['sim predict H>A']==1][sim['tp wins'] >= betting_threshold]
    temp_df2 = sim[sim['sim predict H>A']==0][sim['tn wins'] >= betting_threshold]
    result = temp_df1.append(temp_df2)
    total_profit=0
    cum_profit = [0]
    for profit in result.profit:
        total_profit += profit
        cum_profit.append(total_profit)
    games = np.arange(len(cum_profit))
    return games, cum_profit


# Importing data from CSV, declaring features and target
data = pd.read_csv('data/data.csv')
data.drop(['Unnamed: 0'], axis=1, inplace=True)
features = ['S-Min H-A', 'FG H-A', 'FGA H-A', '3P H-A', '3PA H-A', 'FT H-A',
           'FTA H-A', 'ORB H-A', 'DRB H-A', 'TRB H-A', 'AST H-A', 'STL H-A',
           'BLK H-A', 'TOV H-A', 'PF H-A', 'PTS H-A', 'FG% H-A', '3P% H-A',
           'FT% H-A',  'AST/FG H-A', 'TOV/AST H-A', 'home_b2b', 'away_b2b', 
           'Favored Ahead By', 'Home Spread']
X = data[features].values
y = data['2-Half H>A'].values
df_ML = data.dropna()



# The best algorithm developed in the Model Selection
RF_best = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=5, max_features=None,
                       max_leaf_nodes=None, max_samples=0.5,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=3, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)



# Predictions for the simulation. Each season is predicted individually
# with the model trained on only the previous seasons.
predict_df = df_ML[df_ML['season'].isin([2018, 2019])]
predict_proba = []
for i in range(2018, 2020):
    X_train = data[data['season']<i][features]
    y_train = data[data['season']<i]['2-Half H>A']
    X_test = df_ML[df_ML['season']==i][features]
    y_test = df_ML[df_ML['season']==i]['2-Half H>A']
    RF_best.fit(X_train, y_train)
    y_prob = RF_best.predict_proba(X_test)[:,1]
    predict_proba += list(y_prob)
predict_df['sim predict proba H>A'] = predict_proba


# Getting results of the simulation and plotting them
games1, cum_profit1 = run_sim(0.242, 104)
games2, cum_profit2 = run_sim(0.605, 87)
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(games1, cum_profit1, color='royalblue', label='Home Team Always, Bet thresh. = 104')
ax.plot(games2, cum_profit2, color='firebrick', label='Class thresh. = 0.605, Bet thresh. = 87')
ax.set_xlabel('Games Bet On', labelpad=10)
ax.set_ylabel('Total Profit ($)')
ax.grid()
ax.legend(loc='lower_right')
ax.set_title('Betting Simulation for Last 2 Seasons ($100 Bets)')
plt.savefig('images/BettingSim.png')
plt.show()