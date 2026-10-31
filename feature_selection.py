import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")



# YOUR NAME(s): Sean Jose, Katrina Bourgeois


# In this exercise we will write a hill-climbing search to find the best subset of features to use for learning


# load the "wine" dataset into a dataframe
# It contains 178 instances of wines, with various features. Each is classified into 1 of 3 categories
df = pd.read_csv('wine.csv', index_col=None)

# randomly assign instances to training and test sets (50% each for illustration purposes - 66.6% training maybe more typical)
train_idx = np.random.random(size=df.shape[0]) < 0.5

# separate into features (X) and classes (Y), for training and test sets
x_train = df.drop('class', axis=1).loc[train_idx]
y_train = df['class'].loc[train_idx]
x_test = df.drop('class', axis=1).loc[~train_idx]
y_test = df['class'].loc[~train_idx]

# instantiate the LogisticRegression object to use in the search
full_model = LogisticRegression()
full_model.fit(X=x_train, y=y_train)
print(f'Accuracy with all features: {full_model.score(x_test, y_test)}')

features_list = x_train.columns # start with all features.
best_score = 0

"""
Takes the wine dataset and recursively discards features one at a time to improve accuracy of classification for LogisticRegression()

param: features_list - list of features from wine dataset. Since it's recursive, the parameter may also be the list of features that are redudant for classification
param: best_score - accuracy score that needs to be surpassed to determine if another feature should be recursively removed. 0 is passed in initially
"""
def feature_selection(features_list, best_score):
    best_model = None
    accuracy_improved = False
    # try every combo of every feature except one
    for i in range(len(features_list)): 
        # create a model with one less feature
        x_minus = x_train.drop(features_list[i], axis=1)
        x_test_minus = x_test.drop(features_list[i], axis=1)
        new_model = LogisticRegression().fit(X=x_minus, y=y_train)
        # see how well the new model does
        score = new_model.score(x_test_minus, y_test)

        if score > best_score:
            best_score = score
            best_model = new_model
            feature_dropped = features_list[i]
            print(f'Feature dropped: {feature_dropped}')
            accuracy_improved = True

    if accuracy_improved:
        # drop insignificant feature.
        features_list = features_list.drop(feature_dropped)
        # Recursively remove another feature (if improvement is found).
        return feature_selection(features_list, best_score)
    else:
        # Return model & score if no improvement is found from removing a feature.
        print("No more improvement can be found!")
        return best_model, best_score

better_model, best_score = feature_selection(features_list, best_score)

print(f'Accuracy after feature selection: {best_score}')


# The dataset includes 13 features. Some may be redundant, or irrelevant to the prediction of wine category.
# Let's find the best subset of features to use for supervised learning
# write a search that incrementally builds the optimal feature subset (the subset that gets best accuracy on test set)
# The search can go forward (adding features one by one) or backward (removing features one by one)
# stop the search when another addition/removal would not improve accuracy.
# Note that you can get the complete list of feature names like this: x_train.columns, or as a set: set(x_train.columns)
# and you can reduce the dataframe to a subset of features like this: x_train_subset = x_train[desired_features_list]

