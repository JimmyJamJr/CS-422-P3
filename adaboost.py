# Jimson Huang
# CS 422 - Project 3

import numpy as np
from sklearn import tree


def adaboost_train(X, Y, max_iter):
    f = []
    alpha = []

    for k in range(max_iter):
        # Initialize even sample weights
        d = [1 / len(Y)] * len(Y)

        # Create and train the decision stump
        stump = tree.DecisionTreeClassifier(max_depth=1, random_state=0)
        stump.fit(X, Y)

        # Add decision stump to the list
        f.append(stump)
        # Get the stump's predictions on the data set
        predictions = stump.predict(X)
        # Calculate the stump's error and alpha, add to list
        err = sum([predictions[i] != Y[i] for i in range(len(Y))]) / len(Y)
        a = 0.5 * np.log((1 - err) / err)
        alpha.append(a)

        # Calculate new sample weights
        new_d = []
        for i in range(len(Y)):
            new_d.append(d[i] * np.exp(-alpha[k] * Y[i] * predictions[i]))

        # Normalize the new sample weights by dividing by Z
        Z = sum(new_d)
        new_d = [j / Z for j in new_d]

        # Add copies of sample to data based on its weight
        new_X = []
        new_y = []
        for i in range(len(Y)):
            copies = round(new_d[i] / min(new_d))
            new_y += copies * [Y[i]]
            new_X += copies * [X[i]]
        X = new_X
        Y = new_y

    return f, alpha


def adaboost_test(X, Y, f, alpha):
    correct = 0
    for i in range(len(Y)):
        if np.sign(np.sum([alpha[j] * f[j].predict(np.array(X[i]).reshape(1, -1)) for j in range(len(f))])) * Y[i] > 0:
            correct += 1
    return correct / len(Y)