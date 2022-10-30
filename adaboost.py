import numpy as np
from sklearn import tree


def adaboost_train(X, Y, max_iter):
    f = []
    alpha = []
    for k in range(max_iter):
        stomp = tree.DecisionTreeClassifier(max_depth=1)
        stomp.fit(X, Y)
        f.append(stomp)
        predictions = stomp.predict(X)
        err = sum([predictions[i] != Y[i] for i in range(len(Y))]) / len(Y)
        a = 0.5 * np.log((1 - err) / err)
        alpha.append(a)

        # print(Y)
        # print([np.exp(-a) * Y[i] * predictions[i] for i in range(len(Y))])

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

        # print(Y)

    return f, alpha


def adaboost_test(X, Y, f, alpha):
    correct = 0
    for i in range(len(Y)):
        if np.sign(np.sum([alpha[j] * f[j].predict(np.array(X[i]).reshape(1, -1)) for j in range(len(f))])) * Y[i] > 0:
            correct += 1
    return correct / len(Y)


X = [[-2,-2],[-3,-2],[-2,-3],[-1,-1],[-1,0],[0,-1],[1,1],[1,0],[0,1],[2,2],[3,2],[2,3]]
Y=[-1,-1,-1,1,1,1,-1,-1,-1,1,1,1]
f, alpha = adaboost_train(X,Y,9)
acc = adaboost_test(X,Y,f,alpha)
print("Accuracy:", acc)