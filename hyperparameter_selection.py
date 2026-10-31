import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier


# YOUR NAME(s): Sean Jose, Katrina Bourgeois


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


# perform a grid search of criterion, max_depth, and ccp_alpha parameters to see what gives the best decision tree.
# (by "best" we mean best classification accuracy on the test dataset)
# this nested loop may help get you started:
best_accuracy = 0
for criterion in ['gini', 'entropy', 'log_loss']:
    for max_depth in range(1, 6):
        for ccp_alpha in np.arange(0, 0.25, 0.01):
            
            new_clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, ccp_alpha = ccp_alpha)
            best_clf = new_clf
            # Train with data
            new_clf.fit(x_train,y_train)
            # Get predictions from trained model
            y_prediction = new_clf.predict(x_test)
            # Compare predicted y to actual y values.
            accuracy = (y_prediction == y_test).mean()
            #print(f'Accuracy: {accuracy} ')

            # Save the hyperparameters that give the most acurracy
            if (accuracy > best_accuracy):
                best_accuracy = accuracy
                current_best_settings = np.array([criterion,max_depth,ccp_alpha,best_accuracy])
                print(current_best_settings)

        


