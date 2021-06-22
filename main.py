import numpy as np
#from pylab import *
from numpy.matlib import repmat
import sys
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
from helper import *

print('You\'re running python %s' % sys.version.split(' ')[0])

xTrSpiral,yTrSpiral,xTeSpiral,yTeSpiral= spiraldata(150)
xTrIon,yTrIon,xTeIon,yTeIon= iondata()

# Create a regression tree with no restriction on its depth
# and weights for each training example to be 1
# if you want to create a tree of max_depth k
# then call RegressionTree(depth=k)
tree = RegressionTree(depth=np.inf)

# To fit/train the regression tree
tree.fit(xTrSpiral, yTrSpiral)

# To use the trained regression tree to predict a score for the example
score = tree.predict(xTrSpiral)

# To use the trained regression tree to make a +1/-1 prediction
pred = np.sign(tree.predict(xTrSpiral))

tr_err = np.mean((np.sign(tree.predict(xTrSpiral)) - yTrSpiral) ** 2)
te_err = np.mean((np.sign(tree.predict(xTeSpiral)) - yTeSpiral) ** 2)

print("Training error: %.4f" % np.mean(np.sign(tree.predict(xTrSpiral)) != yTrSpiral))
print("Testing error:  %.4f" % np.mean(np.sign(tree.predict(xTeSpiral)) != yTeSpiral))


def visclassifier(fun, xTr, yTr):
    """
    visualize decision boundary
    Define the symbols and colors we'll use in the plots later
    """

    yTr = np.array(yTr).flatten()

    symbols = ["ko", "kx"]
    marker_symbols = ['o', 'x']
    mycolors = [[0.5, 0.5, 1], [1, 0.5, 0.5]]
    # get the unique values from labels array
    classvals = np.unique(yTr)

    plt.figure()

    # return 300 evenly spaced numbers over this interval
    res = 300
    xrange = np.linspace(min(xTr[:, 0]), max(xTr[:, 0]), res)
    yrange = np.linspace(min(xTr[:, 1]), max(xTr[:, 1]), res)

    # repeat this matrix 300 times for both axes
    pixelX = repmat(xrange, res, 1)
    pixelY = repmat(yrange, res, 1).T

    xTe = np.array([pixelX.flatten(), pixelY.flatten()]).T

    # test all of these points on the grid
    testpreds = fun(xTe)

    # reshape it back together to make our grid
    Z = testpreds.reshape(res, res)
    # Z[0,0] = 1 # optional: scale the colors correctly

    # fill in the contours for these predictions
    plt.contourf(pixelX, pixelY, np.sign(Z), colors=mycolors)

    # creates x's and o's for training set
    for idx, c in enumerate(classvals):
        plt.scatter(xTr[yTr == c, 0],
                    xTr[yTr == c, 1],
                    marker=marker_symbols[idx],
                    color='k'
                    )

    plt.axis('tight')
    # shows figure and blocks
    plt.show()


tree = RegressionTree(depth=np.inf)
tree.fit(xTrSpiral, yTrSpiral)  # compute tree on training data
visclassifier(lambda X: tree.predict(X), xTrSpiral, yTrSpiral)
print("Training error: %.4f" % np.mean(np.sign(tree.predict(xTrSpiral)) != yTrSpiral))
print("Testing error:  %.4f" % np.mean(np.sign(tree.predict(xTeSpiral)) != yTeSpiral))


def forest(xTr, yTr, m, maxdepth=np.inf):
    """Creates a random forest.

    Input:
        xTr:      n x d matrix of data points
        yTr:      n-dimensional vector of labels
        m:        number of trees in the forest
        maxdepth: maximum depth of tree

    Output:
        trees: list of decision trees of length m
    """

    n, d = xTr.shape
    trees = []

    for i in range(m):
        d = np.random.choice(n, n)
        onetree = RegressionTree(maxdepth)
        onetree.fit(xTr[d, :], yTr[d])
        trees.append(onetree)

    return trees


def evalforest(trees, X):
    """Evaluates X using trees.

    Input:
        trees:  list of TreeNode decision trees of length m
        X:      n x d matrix of data points
        alphas: m-dimensional weight vector

    Output:
        pred: n-dimensional vector of predictions
    """
    m = len(trees)
    n, d = X.shape

    pred = np.zeros(n)

    # sum prediction for each tree
    for t in range(m):
        pred += trees[t].predict(X)

    # get average
    pred = pred / m

    return pred

trees=forest(xTrSpiral,yTrSpiral, 50) # compute tree on training data
visclassifier(lambda X:evalforest(trees,X),xTrSpiral,yTrSpiral)
print("Training error: %.4f" % np.mean(np.sign(evalforest(trees,xTrSpiral)) != yTrSpiral))
print("Testing error:  %.4f" % np.mean(np.sign(evalforest(trees,xTeSpiral)) != yTeSpiral))


M=20 # max number of trees
err_trB=[]
err_teB=[]
alltrees=forest(xTrIon,yTrIon,M)
for i in range(M):
    trees=alltrees[:i+1]
    trErr = np.mean(np.sign(evalforest(trees,xTrIon)) != yTrIon)
    teErr = np.mean(np.sign(evalforest(trees,xTeIon)) != yTeIon)
    err_trB.append(trErr)
    err_teB.append(teErr)
    print("[%d]training err = %.4f\ttesting err = %.4f" % (i,trErr, teErr))

plt.figure()
line_tr, = plt.plot(range(M), err_trB, '-*', label="Training Error")
line_te, = plt.plot(range(M), err_teB, '-*', label="Testing error")
plt.title("Random Forest")
plt.legend(handles=[line_tr, line_te])
plt.xlabel("# of trees")
plt.ylabel("error")
plt.show()


def onclick_forest(event):
    """
    Visualize forest, including new point
    """
    global xTrain, yTrain, w, b, M
    # create position vector for new point
    pos = np.array([[event.xdata, event.ydata]])
    if event.key == 'shift':  # add positive point
        color = 'or'
        label = 1
    else:  # add negative point
        color = 'ob'
        label = -1
    xTrain = np.concatenate((xTrain, pos), axis=0)
    yTrain = np.append(yTrain, label)
    marker_symbols = ['o', 'x']
    classvals = np.unique(yTrain)

    w = np.array(w).flatten()

    mycolors = [[0.5, 0.5, 1], [1, 0.5, 0.5]]

    # return 300 evenly spaced numbers over this interval
    res = 300
    xrange = np.linspace(0, 1, res)
    yrange = np.linspace(0, 1, res)

    # repeat this matrix 300 times for both axes
    pixelX = repmat(xrange, res, 1)
    pixelY = repmat(yrange, res, 1).T

    xTe = np.array([pixelX.flatten(), pixelY.flatten()]).T

    # get forest
    trees = forest(xTrain, yTrain, M)
    fun = lambda X: evalforest(trees, X)
    # test all of these points on the grid
    testpreds = fun(xTe)

    # reshape it back together to make our grid
    Z = testpreds.reshape(res, res)
    # Z[0,0] = 1 # optional: scale the colors correctly

    plt.cla()
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    # fill in the contours for these predictions
    plt.contourf(pixelX, pixelY, np.sign(Z), colors=mycolors)

    for idx, c in enumerate(classvals):
        plt.scatter(xTrain[yTrain == c, 0],
                    xTrain[yTrain == c, 1],
                    marker=marker_symbols[idx],
                    color='k'
                    )
    plt.show()


xTrain = np.array([[5, 6]])
b = yTrIon
yTrain = np.array([1])
w = xTrIon
M = 20


fig = plt.figure()
plt.xlim(0, 1)
plt.ylim(0, 1)
cid = fig.canvas.mpl_connect('button_press_event', onclick_forest)
print('Note: You may notice a delay when adding points to the visualization.')
plt.title('Use shift-click to add negative points.')