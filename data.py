import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt


class EngineeredData:

    def correlation(x):
        feature_columns = x.columns.pop('Survived', inplace = True)     
        for column in feature_columns:
            print('Survival correllation by ' + column)
            print(x[[column, 'Survived']].groupby(column, as_index=False).mean())
            print('-'*15, '\n')

    def impute_age(row):
        if 'Master.' in row['Name']:
            return 12
        elif 'Mr.' in row['Name']:
            return 35
        elif 'Mrs.' in row['Name']:
            return 35
        elif 'Miss.' in row['Name']:
            return 12
        else:
            print('couldnt determine person type from name')
            return 30

    def determine_title(row):
        if 'Master.' in row['Name']:
            return 'Master'
        elif 'Mr.' in row['Name']:
            return 'Mr'
        elif 'Mrs.' in row['Name']:
            return 'Mrs'
        elif 'Miss.' in row['Name']:
            return 'Ms'
        else:
            return 'Misc'

    def get_data():
        test_file = r'test.csv'
        train_file = r'train.csv'
        test_data = pd.read_csv(test_file, sep=',', header=[0])
        train_data = pd.read_csv(train_file, sep=',', header=[0])

        train_data['FamMem'] = train_data['Parch'] + train_data['SibSp'] + 1
        test_data['FamMem'] = test_data['Parch'] + test_data['SibSp'] + 1

        #impute the one missing piece of fair data in test_set
        test_data['Fare'].fillna(test_data['Fare'].mean(), inplace = True)

        # fill missing emabarked values with the mode
        test_data['Embarked'].fillna(test_data['Embarked'].mode(), inplace= True)
        train_data['Embarked'].fillna(train_data['Embarked'].mode(), inplace= True)

        # impute hmissing age values based on title in name
        train_data['Age'] = train_data.apply(
            lambda row: EngineeredData.impute_age(row) if np.isnan(row['Age']) else row['Age'],
            axis=1
        )

        # impute hmissing age values based on title in name
        test_data['Age'] = test_data.apply(
            lambda row: EngineeredData.impute_age(row) if np.isnan(row['Age']) else row['Age'],
            axis=1
        )

        # creating the bainry feature, is alone ?
        train_data['isAlone'] = 1
        test_data['isAlone'] = 1
        train_data['isAlone'].loc[train_data['FamMem'] > 1] = 0
        test_data['isAlone'].loc[test_data['FamMem'] > 1] = 0

        #creating a title feature
        train_data['Title'] = 'Misc'
        test_data['Title'] = 'Misc'
        train_data['Title'] = train_data.apply( lambda row: EngineeredData.determine_title(row) , axis = 1 )
        test_data['Title'] = test_data.apply( lambda row: EngineeredData.determine_title(row) , axis = 1 )

        #creating age bins
        train_data['AgeBin'] = pd.cut(train_data['Age'].astype(int), 5, labels = ['0', '1', '2', '3', '4'])
        test_data['AgeBin'] = pd.cut(test_data['Age'].astype(int), 5, labels = ['0', '1', '2', '3', '4'])

        #creating fare bins
        train_data['FareBin'] = pd.qcut(train_data['Fare'], 5, labels = ['0', '1', '2', '3', '4'])
        test_data['FareBin'] = pd.qcut(test_data['Fare'], 5, labels = ['0', '1', '2', '3', '4'])

        train_data = train_data.drop(['Name', 'Fare', 'Age', 'Parch', 'SibSp', 'Ticket', 'Cabin'], axis = 1 )
        test_data = test_data.drop(['Name',  'Fare', 'Age', 'Parch', 'SibSp', 'Ticket', 'Cabin'], axis = 1 )

        test_ids = test_data['PassengerId']
        train_data.drop(['PassengerId'], axis = 1, inplace = True)
        test_data.drop(['PassengerId'], axis = 1, inplace = True)

        # y_train = train_data.pop('Survived') need to keep this for training tree
        x_train = train_data
        x_test = test_data

        x_train['Pclass'] = x_train['Pclass'].astype('category')
        x_train['Sex'] = x_train['Sex'].astype('category')
        x_train['Title'] = x_train['Title'].astype('category')
        x_train['FareBin'] = x_train['FareBin'].astype('category')
        x_train['Embarked'] = x_train['Embarked'].astype('category')
        x_train = pd.get_dummies(x_train)

        x_test['Pclass'] =  x_test['Pclass'].astype('category')
        x_test['Sex'] =  x_test['Sex'].astype('category')
        x_test['Title'] = x_test['Title'].astype('category')
        x_test['FareBin'] = x_test['FareBin'].astype('category')
        x_test['Embarked'] = x_test['Embarked'].astype('category')
        x_test = pd.get_dummies(x_test)
        x_test_kaggle = x_test

        return x_train, x_test_kaggle, test_ids





