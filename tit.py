from __future__ import print_function

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
from data import EngineeredData

x_train, x_test_kaggle, test_ids = EngineeredData.get_data()

y_train = x_train.pop('Survived')

# splitting complete kaggle training data further to implement early stopping 
x_train_final, x_test_final, y_train_final, y_test_final = train_test_split(x_train, y_train, test_size = 0.35)

clf = xgb.XGBClassifier(base_score=0.35, booster='gbtree',
       nthread=None, objective='binary:logistic')

clfrfe = RFECV(clf, step=1, min_features_to_select=1, cv=4, scoring='neg_mean_absolute_error')
clfrfe.fit(x_train_final, y_train_final)

#PLot # of features selected vs. Model Score
plt.figure()
plt.title('XGB CV score vs No of Features')
plt.xlabel("Number of features selected")
plt.ylabel("Neg Mean Squared Error")
plt.plot(range(1, len(clfrfe.grid_scores_) + 1), clfrfe.grid_scores_)
plt.show()

#get rank of X model features
print(clfrfe.ranking_)

x_train_final = x_train_final[x_train_final.columns.values[clfrfe.get_support()]]
x_test_final = x_test_final[x_test_final.columns.values[clfrfe.get_support()]]
x_test_kaggle = x_test_kaggle[x_test_kaggle.columns.values[clfrfe.get_support()]]

params = {
    'max_depth': [6],
    'min_child_weight': [0.8, 0.9],
    'learning_rate': [0.01],
    'n-estimators': [300]
}

clf = xgb.XGBClassifier(base_score=0.35, booster='gbtree',
       nthread=None, objective='binary:logistic')

cv = GridSearchCV(clf, params, scoring = 'accuracy', cv=4)

print(x_train_final)
print(y_train_final)

fit = cv.fit(x_train_final, y_train_final)
y_pred_train = cv.best_estimator_.predict(x_train_final)
y_pred_test = cv.best_estimator_.predict(x_test_final)

y_pred_test_kaggle = cv.best_estimator_.predict(x_test_kaggle)

print("The best parameters are %s with a score of %0.2f"
      % (cv.best_params_, cv.best_score_))

y_pred_test_ids = pd.DataFrame({ 'PassengerId': test_ids, 'Survived': y_pred_test_kaggle})
y_pred_test_ids.Survived = y_pred_test_ids.Survived.round().astype("int")
y_pred_test_ids.to_csv('submission.csv', sep=',', encoding='utf-8', index=False)


acc = accuracy_score(y_train_final, y_pred_train)
print("Accuracy on training data: %.2f%%" % (acc * 100.0))

acc = accuracy_score(y_test_final, y_pred_test)
print("Accuracy on testing data: %.2f%%" % (acc * 100.0))

# feature_importances = pd.Series(cv.best_estimator_.feature_importances_, index = x_train.columns)
# feature_importances.nlargest(40).plot(kind='barh')

# plt.show(block=True)




