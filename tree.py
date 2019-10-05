from __future__ import print_function

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt


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
    lambda row: impute_age(row) if np.isnan(row['Age']) else row['Age'],
    axis=1
)

# impute hmissing age values based on title in name
test_data['Age'] = test_data.apply(
    lambda row: impute_age(row) if np.isnan(row['Age']) else row['Age'],
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
train_data['Title'] = train_data.apply( lambda row: determine_title(row) , axis = 1 )
test_data['Title'] = test_data.apply( lambda row: determine_title(row) , axis = 1 )

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

print(x_train)

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

x_train = x_train[['FamMem', 'isAlone', 'Pclass_1', 'Pclass_2', 'Pclass_3',
       'Sex_female', 'Sex_male', 'Title_Master', 'Title_Misc', 'Title_Mr',
       'Title_Mrs', 'Title_Ms', 'AgeBin_0', 'AgeBin_1', 'AgeBin_2', 'AgeBin_3',
       'AgeBin_4', 'FareBin_0', 'FareBin_1', 'FareBin_2', 'FareBin_3',
       'FareBin_4', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Survived']]

training_data = x_train.to_numpy()
testing_data = x_test_kaggle.to_numpy()

header = ['FamMem', 'isAlone', 'Pclass_1', 'Pclass_2', 'Pclass_3',
       'Sex_female', 'Sex_male', 'Title_Master', 'Title_Misc', 'Title_Mr',
       'Title_Mrs', 'Title_Ms', 'AgeBin_0', 'AgeBin_1', 'AgeBin_2', 'AgeBin_3',
       'AgeBin_4', 'FareBin_0', 'FareBin_1', 'FareBin_2', 'FareBin_3',
       'FareBin_4', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Survived']

def unique_vals(rows, col):
    return set([row[col] for row in rows])

def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)

class Question:
    """A Question is used to partition a dataset.

    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.
    """

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))

def partition(rows, question):
    """Partitions a dataset.

    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

def gini(rows):
    """Calculate the Gini Impurity for a list of rows.

    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

def info_gain(left, right, current_uncertainty):
    """Information Gain.

    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

def find_best_split(rows):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            question = Question(col, val)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question

class Leaf:
    """A Leaf node classifies data."""

    def __init__(self, rows):
        self.predictions = class_counts(rows)

class Decision_Node:
    """A Decision Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

def build_tree(rows):
    """Builds the tree.

    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = find_best_split(rows)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf(rows)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return Decision_Node(question, true_branch, false_branch)

def custom_tree(rows):

    is_female_q = Question(5, 1)
    is_master_q = Question(7, 1)
    is_class_1  = Question(2, 1)
    is_class_2 =  Question(3, 1)
    is_embark_s = Question(24, 1)

    true_rows_female, false_rows_female = partition(rows, is_female_q)
    true_rows_master, false_rows_master = partition(false_rows_female, is_master_q)
    true_rows_1, false_rows_1           = partition(true_rows_female, is_class_1)
    true_rows_2, false_rows_2           = partition(false_rows_1, is_class_2)
    true_rows_embarked, false_rows_embarked = partition(false_rows_2, is_embark_s)

    return Decision_Node(is_female_q, Decision_Node(is_class_1, Leaf(true_rows_1), Decision_Node(is_class_2, Leaf(true_rows_2), Decision_Node(is_embark_s, Leaf(true_rows_embarked), Leaf(false_rows_embarked)))), Decision_Node(is_master_q, Leaf(true_rows_master), Leaf(false_rows_master)))

def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")

def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""

    return int(max(counts, key=counts.get))

my_tree = custom_tree(training_data)
print_tree(my_tree)






preds = []

for row in training_data:
    preds.append(print_leaf(classify(row, my_tree)))

actual_surv = train_data.pop('Survived')
print(actual_surv.to_numpy())

acc = accuracy_score(actual_surv.to_numpy(), preds)
print("Accuracy on training data: %.2f%%" % (acc * 100.0))


# y_pred_test_ids = pd.DataFrame({ 'PassengerId': test_ids, 'Survived': preds})
# y_pred_test_ids.Survived = y_pred_test_ids.Survived.round().astype("int")
# y_pred_test_ids.to_csv('submissiontree.csv', sep=',', encoding='utf-8', index=False)


