from __future__ import print_function

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, train_test_split,cross_val_score, cross_validate, ShuffleSplit
import matplotlib.pyplot as plt
from data import EngineeredData

x_train, x_test_kaggle, test_ids = EngineeredData.get_data()

y_train = x_train.pop('Survived')

# splitting complete kaggle training data further to implement early stopping 
x_train_final, x_test_final, y_train_final, y_test_final = train_test_split(x_train, y_train, test_size = 0.35)

tree = DecisionTreeClassifier(max_depth = 4, random_state = 0)
base = cross_val_score(tree, x_train_final, y_train_final)
fit = tree.fit(x_train_final, y_train_final)

preds = tree.predict(x_test_final)

acc = accuracy_score(y_test_final, preds)
print("Accuracy on pre RFE testing data: %.2f%%" % (acc * 100.0))

tree_rfe = RFECV(tree, step = 1, scoring = 'accuracy')
fit = tree_rfe.fit(x_train_final, y_train_final)

#PLot # of features selected vs. Model Score
plt.figure()
plt.title('XGB CV score vs No of Features')
plt.xlabel("Number of features selected")
plt.ylabel("Neg Mean Squared Error")
plt.plot(range(1, len(fit.grid_scores_) + 1), fit.grid_scores_)
plt.show()

#get rank of X model features
print(fit.ranking_)

x_train_rfe = x_train_final[x_train_final.columns.values[tree_rfe.get_support()]]
x_test_rfe = x_test_final[x_test_final.columns.values[tree_rfe.get_support()]]
x_test_kaggle = x_test_kaggle[x_test_kaggle.columns.values[tree_rfe.get_support()]]

redClass = DecisionTreeClassifier(random_state = 0)
redfit = redClass.fit(x_train_rfe, y_train_final)

redpreds = redClass.predict(x_test_rfe)

acc = accuracy_score(y_test_final, redpreds)
print("Accuracy on post RFE pre tune testing data: %.2f%%" % (acc * 100.0))

param_grid = {'criterion': ['gini', 'entropy'],  #scoring methodology; two supported formulas for calculating information gain - default is gini
              #'splitter': ['best', 'random'], #splitting methodology; two supported strategies - default is best
              'max_depth': [2,3,4,6], #max depth tree can grow; default is none
              'min_samples_split': [.03], #minimum subset size BEFORE new split (fraction is % of total); default is 2
              #'min_samples_leaf': [1,5,10,.03,.05], #minimum subset size AFTER new split split (fraction is % of total); default is 1
              #'max_features': [None, 'auto'], #max features to consider when performing split; default none or all
              'random_state': [0] #seed or control random number generator: https://www.quora.com/What-is-seed-in-random-number-generation
             }

tree_tune_rfe = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = 4,  return_train_score=True)
tree_tune_rfe.fit(x_train_rfe, y_train_final)

print(tree_tune_rfe.cv_results_.keys())
print(tree_tune_rfe.best_params_)

print(tree_tune_rfe.cv_results_['mean_train_score'])
print(tree_tune_rfe.cv_results_['mean_test_score'])

tunepreds = tree_tune_rfe.best_estimator_.predict(x_test_rfe)
acc = accuracy_score(y_test_final, tunepreds)
print("Accuracy on tuned test data: %.2f%%" % (acc * 100.0))

kagglepreds = tree_tune_rfe.best_estimator_.predict(x_test_kaggle)
y_pred_test_ids = pd.DataFrame({ 'PassengerId': test_ids, 'Survived': kagglepreds})
y_pred_test_ids.Survived = y_pred_test_ids.Survived.round().astype("int")
y_pred_test_ids.to_csv('submissionskl.csv', sep=',', encoding='utf-8', index=False)