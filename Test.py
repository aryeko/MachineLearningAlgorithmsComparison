import numpy as np
import pandas as pd
import time
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from sklearn.decomposition import PCA


if __name__ == "__main__":

    trainDataFilePath = "ExcerciseData\\real_project_data.xls";

    # Read the data
    print("Reading the data from the xls file")

    xl = pd.ExcelFile(trainDataFilePath)

'''
    data = np.genfromtxt('creditcard.csv', delimiter=',')
    rows, cols = data.shape

    # Classification is in the last column
    Y = data[:, cols - 1]
    X = data[:, :cols - 1]
    print("Selecting random indexes for train and test")
    nonzero_y_indexes = np.where(Y == 1)[0]
    zero_y_indexes = np.where(Y == -1)[0]

    # get random zero samples indexes in the length on the non zero indexes
    random_x_train_indexes = np.random.choice(zero_y_indexes, len(nonzero_y_indexes))

    train_indexes = np.append(random_x_train_indexes, nonzero_y_indexes)
    x_train = X[train_indexes, :]
    y_train = Y[train_indexes]

    number_of_test_indexes = 5000
    test_indexes = np.random.choice(range(rows), number_of_test_indexes)
    x_test, y_test = X[test_indexes, :], Y[test_indexes]
    print("Training on %d samples, testing on %d new samples" % (len(train_indexes), number_of_test_indexes))

'''