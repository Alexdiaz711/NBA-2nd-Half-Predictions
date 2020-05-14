# This script produces data visualizations to perform a cost/benefit analysis based on 
# historical betting odds and develop a betting strategy to be implemented

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('default')
font = {'weight': 'bold',
        'size':   12}
plt.rc('font', **font)



# Defining a function to be used in this script
def get_stats(class_threshold, betting_threshold, exp_profit=True):
    '''
    Uses the provided classification and betting thresholds to generate the
    profit to be used in the cost/benefit analysis at each combination of 
    classification threshold and betting threshold.

    Parameters
    ----------
    class_threshold : Float between 0.0 and 1.0 (both inclusive)
    Threshold for which targets will be classified as positive
    when the prediction probability is higher.

    betting_threshold : Non-Negative Integer
    Threshold for which all bets will be ignored when the winning
    payouts is lower.

    exp_profit : Boolean, default=True
        When True, function returns profit as an average profit per bet,
        the "expected profit." When False, return total profit over the two
        seasons.

    Returns
    -------
    profit : Float
        Expected or Total Profit
    num : Integer
        Number of bets made over twhe two seasons
    cb: numpy array of shape (2,2)
        Cost/Benefit matrix at the given parameters
    accuracy: Float
        Accuracy of predictions at the given classification threshold
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
    temp_df3 = temp_df1.append(temp_df2)
    cb_tp = round(temp_df3[temp_df3['sim tp']==1].profit.mean(), 2)
    cb_fp = round(temp_df3[temp_df3['sim fp']==1].profit.mean(), 2)
    cb_tn = round(temp_df3[temp_df3['sim tn']==1].profit.mean(), 2)
    cb_fn = round(temp_df3[temp_df3['sim fn']==1].profit.mean(), 2)
    cb = np.array(([cb_tp, cb_fn], [cb_fp, cb_tn]))
    accuracy = np.mean(temp_df3['sim predict H>A'] == temp_df3['2-Half H>A']) 
    if exp_profit==True:
        return temp_df3.profit.mean(), temp_df3.profit.count(), cb, accuracy
    else: 
        return temp_df3.profit.sum(), temp_df3.profit.count(), cb, accuracy
  



# Importing the game data for model training and prediciton
data = pd.read_csv('data/data.csv')
data.drop(['Unnamed: 0'], axis=1, inplace=True)
features = ['S-Min H-A', 'FG H-A', 'FGA H-A', '3P H-A', '3PA H-A', 'FT H-A',
           'FTA H-A', 'ORB H-A', 'DRB H-A', 'TRB H-A', 'AST H-A', 'STL H-A',
           'BLK H-A', 'TOV H-A', 'PF H-A', 'PTS H-A', 'FG% H-A', '3P% H-A',
           'FT% H-A',  'AST/FG H-A', 'TOV/AST H-A', 'home_b2b', 'away_b2b', 
           'Favored Ahead By', 'Home Spread']
X = data[features].values
y = data['2-Half H>A'].values

# Creating a DataFrame with only games with historical half-time betting odds to be
# used in cost/benefit
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

# Prediction probabilities to be used throughout analysis. Each season is predicted individually
# with the model trained on the other seasons.
predict_df = df_ML[df_ML['season'].isin([2015, 2016, 2017])]
predict_proba = []
for i in range(2015, 2018):
    X_train = data[data['season']!=i][features]
    y_train = data[data['season']!=i]['2-Half H>A']
    X_test = df_ML[df_ML['season']==i][features]
    y_test = df_ML[df_ML['season']==i]['2-Half H>A']
    RF_best.fit(X_train, y_train)
    y_prob = RF_best.predict_proba(X_test)[:,1]
    predict_proba += list(y_prob)
predict_df['sim predict proba H>A'] = predict_proba

# Creating 2-D Profit Curve for Expected Profit per Bet
sorted_probs = sorted(predict_proba)
profits = []
accuracies = []
for prob in sorted_probs:
    p, num, cb, a = get_stats(prob, 0)
    accuracies.append(a)
    profits.append(p)
best_ind = np.argmax(profits)
fig, ax = plt.subplots()
ax.plot(sorted_probs, profits, color='r', alpha=0.5)
ax2 = ax.twinx()
ax2.plot(sorted_probs, accuracies, color='b', alpha=0.5)
ax.set_xlabel('Positive Classification Threshold')
ax.set_ylabel('Expected Profit From $100 Bet', color='r')
ax2.set_ylabel('Accuracy', color='b')
ax.grid(True, alpha=0.4)
ax.scatter([sorted_probs[best_ind], 0.5], [profits[best_ind], get_stats(0.5,0.0)[0]], color='g')
text = "({:0.3f}, {:2.2f})".format(sorted_probs[best_ind], profits[best_ind])
ax.set_title('Profit Curve (w/Accuracies)')
plt.savefig('images/InitialProfitCurve.png')
plt.show()

# Creating CB Matrix, with Accuracy, and Expected Profit for illustrative purposes
p, num, cb, a = get_stats(0.5, 0)
print(
    '''
    Cost/Benefit Matrix at Classification Threshold of 0.5:
    {}
    Expected Profit (per bet): {:.2f}
    Accuracy: {:.3f}
    '''
    .format(cb, p, a))
print('\n')
p, num, cb, a = get_stats(0, 0)
print(
    '''
    Cost/Benefit Matrix at Classification Threshold of 0.0:
    {}
    Expected Profit (per bet): {:.2f}
    Accuracy: {:.3f}
    '''
    .format(cb, p, a))

# Creating 3d Expected profit surface, implementing betting threshold
profit_coord = []
bet_coord = []
prob_coord = []
for bet_thresh in np.arange(0, 201):
    for prob in np.arange(.2075, .85, 0.0025):
        p, num, cb, a = get_stats(prob, bet_thresh)
        profit_coord.append(p)
        bet_coord.append(bet_thresh)
        prob_coord.append(prob)
xmin = np.min(prob_coord)
xmax = np.max(prob_coord)
ymin = np.min(bet_coord)
ymax = np.max(bet_coord)
zmin = np.min(profit_coord)
zmax = np.max(profit_coord)
zero_loc = -zmin/(zmax - zmin)
c = ["darkred","red","green","darkgreen"]
v = [0., zero_loc - 0.05, zero_loc + 0.05, 1.]
l = list(zip(v,c))
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(np.array(prob_coord), np.array(bet_coord), np.array(profit_coord),
                cmap=LinearSegmentedColormap.from_list('rg',l, N=256))
# fig.colorbar(surf, shrink=0.5) # to evaluate color scale
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_zlim(zmin, zmax)
ax.tick_params(axis='z', which='major', pad=10)
ax.set_xlabel('Positive Classification Threshold', labelpad=10)
ax.set_ylabel('Betting Payout Threshold', labelpad=10)
ax.set_zlabel('Expected Profit Per $100 Bet', labelpad=20)
# plt.savefig('images/3dExpProfit.png')
plt.show()

# Printing max exp profit point
best_ind = np.array(profit_coord).argmax()
p, num, cb, a = get_stats(np.array(prob_coord)[best_ind], np.array(bet_coord)[best_ind])
print(
    '''
    Max Expected Profit is {:.2f} per bet,
    at betting threshold  of {},
    and classification threshold of {:.4f},
    placing {} bets in 3 seasons.
    '''
    .format(p, 
            np.array(bet_coord)[best_ind], 
            np.array(prob_coord)[best_ind],
            num))

# Creating 3d Total Profit Surface
profit_coord = []
bet_coord = []
prob_coord = []
for bet_thresh in np.arange(0, 201):
    for prob in np.arange(.2075, .85, 0.0025):
        p, num, cb, a = get_stats(prob, bet_thresh, exp_profit=False)
        profit_coord.append(p)
        bet_coord.append(bet_thresh)
        prob_coord.append(prob)
xmin = np.min(prob_coord)
xmax = np.max(prob_coord)
ymin = np.min(bet_coord)
ymax = np.max(bet_coord)
zmin = np.min(profit_coord)
zmax = np.max(profit_coord)
zero_loc = -zmin/(zmax - zmin)
c = ["darkred","red","green","darkgreen"]
v = [0., zero_loc - 0.05, zero_loc + 0.05, 1.]
l = list(zip(v,c))
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(np.array(prob_coord), np.array(bet_coord), np.array(profit_coord),
                cmap=LinearSegmentedColormap.from_list('rg',l, N=256))
# fig.colorbar(surf, shrink=0.5) # to evaluate color scale
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_zlim(zmin, zmax)
ax.tick_params(axis='z', which='major', pad=10)
ax.set_xlabel('Positive Classification Threshold', labelpad=10)
ax.set_ylabel('Betting Payout Threshold', labelpad=10)
ax.set_zlabel('Total Profit ($100 Bets)', labelpad=20)
plt.savefig('images/3dTotalProfit.png')
# plt.show()

# Printing max total profit point
best_ind = np.array(profit_coord).argmax()
p, num, cb, a = get_stats(np.array(prob_coord)[best_ind], np.array(bet_coord)[best_ind], exp_profit=False)
print(
    '''
    Max Total Profit is {:.2f} for 3 seasons,
    at betting threshold  of {},
    and classification threshold of {:.4f},
    placing {} bets in 3 seasons.
    '''
    .format(p, 
            np.array(bet_coord)[best_ind], 
            np.array(prob_coord)[best_ind],
            num))
