#1 You probably noticed you get slightly different results each time you run these scripts. Why is that?

The dataset is randomly split in half between training and test every time the program runs. So models are trained and tested on different data every time.

#2 Try the feature-selection search with KNearestNeighbors instead of LogisticRegression. Are the results different? Why or why not?

We noticed that regardless of the amount of neighbors looked at, the initial accuracy from all 13 features was substantially lower than it was with logistic regression.
While logistic regression would initially produce an accuracy around 90%, with k nearest neighbors the initial accuracy is generally around 60%-70%.
However, the final accuracy for k nearest neighbors after selection is essentially the same as logistic regression, with both having final accuracies above 90%.
This could be because k nearest neighbors is more easily impacted by redundant features than logistic regression is, 
which also causes k nearest neighbors to have a greater overall improvement in accuracy once those features are removed. 

