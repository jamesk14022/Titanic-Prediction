# Titanic Prediction

## Clean Up & Feature Engineering

- Used the name field to extract titles and impute relevant age
- Filled missing values in the Embarked column with the mode value
- Replaced missing fare values with the mean column value

- COmbined the siblings and parents column to create a family memebers column
- Used the family member column to create a binary isalone? column
- Created a categorical feature containing passengers title
- Seperated the fare column values into categorical bin columns
- Seperated the age column values into categorical bin columns
- Dropped the cabin feature

## Gini Trees and Hand-coded Trees
- Hard coded my own decision tree with only 5 decision nodes, still achieved 82% accuracy on the training set. Decision tree adapted from [this repo](https://github.com/random-forests/tutorials/blob/master/decision_tree.py)
- Implemented a decision tree recursively built and optimised with gini impurity, achieving 89% accuracy on the training set.
- The gini tree was likely overfit as it wasn't pruned and had no maximum depth. Prediction not generalisable to the testing data. 

## SKLearn Tree
- Recursive feature elimination used to optimise features which maximised the negative mean squared error. The feature optimisation process is also visualised using a pyplot graph. 
- 4 fold cross validation used to tune hyperparameter to maximise the model accuracy

## XGBoost Logistic Regression
- Again used CV for hyperparameter tuning and RFE to refine the features used. 
- IMplemented XGBoost using the Scikit-Learn Wrapper. Appropriate docs can be found [here](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn)
