# This script will use GridSerach with 10-Fold Cross Validation on train data to build the best of each model type.
# Next, the models are compared against each other using accuracy, precision, and ROC AUC on the test data.

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline

plt.style.use('default')
font = {'weight': 'bold',
        'size':   14}
plt.rc('font', **font)

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
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=999)

# Defining a funtion to help score models
def scores(y_hat, y_test, p=False, model_name=None):
    '''
    Uses the provided target predictions and target values to generate model scores.

    Parameters
    ----------
    y_hat : list or numpy array
        Target predictions
    y_test : list or numpy array
        Actual target values.
    p: Boolean, default=False
        When true, function will print scores.
    model_name, default=None
        Name of model being scored used in the printout of scores.

    Returns
    -------
    a : Float
        Accuracy of model.
    r : Float
        recall of model.
    pr: Float
        Precision of Model.
    np: Float
        Rate of predictions being of the positive class.
    auc_score: Float
        Area under the Receiver Operator Characteristic Curve
    '''
    a = accuracy_score(y_test, y_hat)
    r = recall_score(y_test, y_hat)
    pr = precision_score(y_test, y_hat)
    np = y_hat.mean()
    fpr, tpr, thresholds = roc_curve(y_test, y_hat)
    auc_score = auc(fpr, tpr)
    if p==True:
        print('''
            For {}:
            Accuracy : {:2.3f}
            Recall : {:2.3f}
            Precision : {:2.3f}
            Rate of Predict Pos: {:2.3f}
            ROC AUC: {:0.3f}'''.format(model_name, a, r, pr, np, auc_score))
    return a, r, pr, np, auc_score

# Defining a function to plot Receiver Operator Characteristic Curve
def plot_roc(y_test, y_prob, ax, model_label, color):
    '''
    Uses the provided target predictions and target values to plot the ROC curve.

    Parameters
    ----------
    y_hat : list or numpy array
        Target predictions
    y_test : list or numpy array
        Actual target values.
    ax: AxesSubplot
        MatPlotLib Axes Subplot to plot ROC Curve on.
    model_label: string
        Name of model being plotted used in the legend.
    color: string
        Color to be used in plot for the current model.

    Returns
    -------
    None
    '''
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    ax.plot(fpr, tpr, label=model_label, color=color)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operator Characteristic Curve')
    ax.legend(loc='lower right')



# Define a pipeline to search for the best combination of parameters for a Logistic Regression model
# using 10-fold CV mean accuracy as score. 
scaler = StandardScaler()
logistic = LogisticRegression(max_iter=10000)
pipe = Pipeline(steps=[('scaler', scaler), ('logistic', logistic)])
param_grid = {
    'logistic__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'logistic__penalty': ['l1', 'l2'],
    'logistic__solver' : ['newton-cg', 'lbfgs', 'liblinear']}
search = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=10, 
                      scoring='accuracy', 
                      verbose=1)
search.fit(X_train, y_train)
ind = search.best_index_
mean = search.cv_results_['mean_test_score'][ind]
std = search.cv_results_['std_test_score'][ind]
LR_best = search.best_estimator_[1]
print("Best LR CV accuracy mean: {:.3f}, std:{:.3f}".format(mean, std))
print(LR_best)


# Define a pipeline to search for the best combination of parameters for a Random Forest model
# using 10-fold CV mean accuracy as score. 
RF = RandomForestClassifier()
pipe = Pipeline(steps=[('RF', RF)])
param_grid = {
       'RF__max_depth' : [None, 3, 5, 10],
       'RF__min_samples_leaf' : [1, 3, 5, 10],
       'RF__min_samples_split' : [2, 5, 10],
       'RF__max_features' : [None, 'auto', 'log2'],
       'RF__max_samples' : [None, 0.5]}
search = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=10, 
                      scoring='accuracy', 
                      verbose=1)
search.fit(X_train, y_train)
ind = search.best_index_
mean = search.cv_results_['mean_test_score'][ind]
std = search.cv_results_['std_test_score'][ind]
RF_best = search.best_estimator_[0]
print("Best RF CV accuracy mean: {:.3f}, std:{:.3f}".format(mean, std))
print(RF_best)


# Define a pipeline to search for the best combination of parameters for a Gradient Boosting model
# using 10-fold CV mean accuracy as score. 
scaler = StandardScaler()
GB = GradientBoostingClassifier()
pipe = Pipeline(steps=[('scaler', scaler), ('GB', GB)])
param_grid = {
       'GB__loss' : ['deviance', 'exponential'],
       'GB__learning_rate' : [0.01],
       'GB__n_estimators': [500],
       'GB__max_depth' : [None, 3, 5, 10],
       'GB__min_samples_leaf' : [3, 5, 10],
       'GB__min_samples_split' : [2, 5, 10],
       'GB__subsample' : [0.5]}
search = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=10, 
                      scoring='accuracy', 
                      verbose=1)
search.fit(X_train, y_train)
ind = search.best_index_
mean = search.cv_results_['mean_test_score'][ind]
std = search.cv_results_['std_test_score'][ind]
GB_best = search.best_estimator_[1]
print("Best GB CV accuracy mean: {:.3f}, std:{:.3f}".format(mean, std))
print(GB_best)


# Due to the complex nature of tuning and building a Neural Network, this was done manually 
# through trail and error. The best NN classifier built is created below:

# Defining a function to build Neural Network
def create_NN():
    '''
    Function to create a Neural Network based on previously configured parameters.

    Parameters
    ----------
    None

    Returns
    ----------
    None
    '''
    model = Sequential()
    model.add(Dense(10, input_dim=25, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    return model

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_NN, epochs=20, batch_size=32, verbose=1)))
pipeline_NN = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline_NN, X_train, y_train, cv=kfold, n_jobs=-1)
print("Best NN CV accuracy mean: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



# Comparing the best of each model using avg accuracy, precision, and ROC AUC for
# 100 different train/test splits.

# Logistic Regressor Pipeline
estimators_LR = []
estimators_LR.append(('standardize', StandardScaler()))
estimators_LR.append(('LR', LR_best))
pipeline_LR = Pipeline(estimators_LR)

# Random Forest Pipeline
estimators_RF = []
estimators_RF.append(('RF', RF_best))
pipeline_RF = Pipeline(estimators_RF)

# Gradient Boosting Pipeline
estimators_GB = []
estimators_GB.append(('standardize', StandardScaler()))
estimators_GB.append(('GB_best', GB_best))
pipeline_GB = Pipeline(estimators_GB)

# Neural Network Pipeline
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_NN, epochs=20, batch_size=32, verbose=1)))
pipeline_NN = Pipeline(estimators)



# Training Models for comparison and selection
# Baseline: Flip a coin
y_hat_guess = np.random.binomial(1, 0.5, size=len(y_test))
y_prob_guess = np.full(y_test.shape, 0.5)
acc_guess, rec_guess, prec_guess, num_pos_guess, auc_guess = scores(y_hat_guess, y_test, p=True, model_name='Baseline')

# Logistic Regression
pipeline_LR.fit(X_train, y_train)
y_prob_LR = pipeline_LR.predict_proba(X_test)
y_hat_LR = pipeline_LR.predict(X_test)
acc_LR, rec_LR, prec_LR, num_pos_LR, auc_LR = scores(y_hat_LR, y_test, p=True, model_name='Logistic Regression')

# Random Forest
pipeline_RF.fit(X_train, y_train)
y_prob_RF = pipeline_RF.predict_proba(X_test)
y_hat_RF = pipeline_RF.predict(X_test)
acc_RF, rec_RF, prec_RF, num_pos_RF, auc_RF = scores(y_hat_RF, y_test, p=True, model_name='Random Forest')

# Gradient Boosting
pipeline_GB.fit(X_train, y_train)
y_prob_GB = pipeline_GB.predict_proba(X_test)
y_hat_GB = pipeline_GB.predict(X_test)
acc_GB, rec_GB, prec_GB, num_pos_GB, auc_GB = scores(y_hat_GB, y_test, p=True, model_name='Gradient Boosting')

# Neural Network
pipeline_NN.fit(X_train, y_train)
y_prob_NN = pipeline_NN.predict_proba(X_test)
y_hat_NN = pipeline_NN.predict(X_test)
acc_NN, rec_NN, prec_NN, num_pos_NN, auc_NN = scores(y_hat_NN, y_test, p=True, model_name='Neural Network')


# Plotting results
accuracies = [acc_guess, acc_LR, acc_RF, acc_GB, acc_NN]
fig, ax = plt.subplots(1,2, figsize=(17,7))
ax[0].grid(alpha=0.4)
ind = np.arange(1, 6)    # the x locations for the groups
width = 0.5         # the width of the bars
ax[0].bar(ind, accuracies, color=['royalblue', 'orange', 'teal', 'firebrick', 'darkmagenta'])
ax[0].axhline(acc_RF, color='teal', alpha=0.5, ls='--')
ax[0].text(3, 0.618, s='{:0.3f}'.format(acc_RF), fontweight='normal', ha='center')
ax[0].set_title('Test Accuracy By Model')
ax[0].set_xticks(ind)
ax[0].set_xticklabels(('Baseline', 'Logistic\nRegression', 'Random\nForest',
                       'Gradient\nBoosting', 'Neural\nNetwork'))
ax[0].set_ylabel('Test Accuracy')
ax[0].set_ylim(0.475, 0.65)

ax[1].grid(alpha=0.5)
plot_roc(y_test, y_prob_guess, ax[1], "Baseline, AUC={:0.3f}".format(auc_guess), color='royalblue')
plot_roc(y_test, y_prob_LR[:,1], ax[1], 'LR, AUC={:0.3f}'.format(auc_LR), color='orange')
plot_roc(y_test, y_prob_RF[:,1], ax[1], 'RF, AUC={:0.3f}'.format(auc_RF), color='teal')
plot_roc(y_test, y_prob_GB[:,1], ax[1], 'GB, AUC={:0.3f}'.format(auc_GB), color='firebrick')
plot_roc(y_test, y_prob_NN[:,1], ax[1], 'NN, AUC={:0.3f}'.format(auc_NN), color='darkmagenta')
plt.savefig('images/ModelSelection.png')
plt.show()