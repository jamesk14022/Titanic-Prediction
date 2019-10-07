from __future__ import print_function

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from data import EngineeredData
import matplotlib.pyplot as plt

x_train, x_test_kaggle, test_ids = EngineeredData.get_data()

x_train = x_train[['FamMem', 'isAlone', 'Pclass_1', 'Pclass_2', 'Pclass_3',
       'Sex_female', 'Sex_male', 'Title_Master', 'Title_Misc', 'Title_Mr',
       'Title_Mrs', 'Title_Ms', 'AgeBin_0', 'AgeBin_1', 'AgeBin_2', 'AgeBin_3',
       'AgeBin_4', 'FareBin_0', 'FareBin_1', 'FareBin_2', 'FareBin_3',
       'FareBin_4', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Survived']]

training_data = x_train.to_numpy()
testing_data_kaggle = x_test_kaggle.to_numpy()

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
print('Custom tree \n', '-'*10)
print_tree(my_tree)

gini_tree = build_tree(training_data)
print('Gini tree \n', '-'*10)
print_tree(gini_tree)

custom_preds_train = []
for row in training_data:
    custom_preds_train.append(print_leaf(classify(row, my_tree)))

gini_preds_train = []
for row in training_data:
    gini_preds_train.append(print_leaf(classify(row, gini_tree)))

actual_surv = pd.DataFrame(data = training_data, columns = header).pop('Survived')

acc = accuracy_score(actual_surv.to_numpy(), custom_preds_train)
print("Custom tree accuracy on training data: %.2f%%" % (acc * 100.0))

acc = accuracy_score(actual_surv.to_numpy(), gini_preds_train)
print("Gini tree accuracy on training data: %.2f%%" % (acc * 100.0))

custom_preds_test = []
for row in testing_data_kaggle:
    custom_preds_test.append(print_leaf(classify(row, my_tree)))

gini_preds_test = []
for row in testing_data_kaggle:
    gini_preds_test.append(print_leaf(classify(row, gini_tree)))

y_pred_test_ids = pd.DataFrame({ 'PassengerId': test_ids, 'Survived': custom_preds_test})
y_pred_test_ids.Survived = y_pred_test_ids.Survived.round().astype("int")
y_pred_test_ids.to_csv('submissioncustomtree.csv', sep=',', encoding='utf-8', index=False)

y_pred_test_ids = pd.DataFrame({ 'PassengerId': test_ids, 'Survived': gini_preds_test})
y_pred_test_ids.Survived = y_pred_test_ids.Survived.round().astype("int")
y_pred_test_ids.to_csv('submissionginitree.csv', sep=',', encoding='utf-8', index=False)

