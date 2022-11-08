import numpy as np
from sklearn import tree


def adaboost_train(X, Y, max_iter):
    f = []
    alpha = []
    for k in range(max_iter):
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

        new_X = []
        new_y = []
        for i in range(len(Y)):
            if np.exp(-a) * Y[i] * predictions[i] <= 0:
                new_y += 2 * [Y[i]]
                new_X += 2 * [X[i]]
            else:
                new_y += [Y[i]]
                new_X += [X[i]]
        X = new_X
        Y = new_y

    return f, alpha


def adaboost_test(X, Y, f, alpha):
    correct = 0
    for i in range(len(Y)):
        if np.sign(np.sum([alpha[j] * f[j].predict(np.array(X[i]).reshape(1, -1)) for j in range(len(f))])) * Y[i] > 0:
            correct += 1
    return correct / len(Y)


# X = [[2,3], [2,2], [4,6], [4,3], [4,1], [5,7], [5,3], [6,5], [8,6], [8,2]]
# Y=[1,1,1,-1,-1,1,-1,1,-1,-1]
# # X = [[-2,-2],[-3,-2],[-2,-3],[-1,-1],[-1,0],[0,-1],[1,1],[1,0],[0,1],[2,2],[3,2],[2,3]]
# # Y=[-1,-1,-1,1,1,1,-1,-1,-1,1,1,1]
# f, alpha = adaboost_train(X,Y,5)
# acc = adaboost_test(X,Y,f,alpha)
# print("Accuracy:", acc)
#
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.datasets import make_classification
# clf = AdaBoostClassifier(n_estimators=5, random_state=0)
# clf.fit(X, Y)
# print(clf.score(X, Y))