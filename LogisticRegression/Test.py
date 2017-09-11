import random
import numpy
from LogisticRegression import LogisticRegression
import matplotlib.pyplot as plt
import time


def run_k_fold(m, X, Y):
    # k-fold cross validation
    # shuffle data
    index = [i for i in range(0, m)]
    random.shuffle(index)
    X = X[index, :]
    Y = Y[index]

    # k-fold
    kfold = 10
    foldSize = int(m / kfold)

    # arrage to store training and testing error
    trainErr = [0.0] * kfold
    testErr = [0.0] * kfold
    allIndex = range(0, m)
    for k in range(0, kfold):

        test_indexes = range((foldSize * k), foldSize * (k + 1))
        train_indexes = list(set(allIndex) - set(test_indexes))

        train_x, train_y = X[train_indexes, :], Y[train_indexes]
        test_x, test_y = X[test_indexes, :], Y[test_indexes]

        # set parameter
        alpha = 0.05
        model = LogisticRegression(train_x, train_y, alpha)
        cost_array = model.gradient_decent(500, 0.01)

        #draw(cost_array)

        trainPred = model.predict(train_x)
        trainErr[k] = float(sum(trainPred != train_y)) / len(train_y)

        testPred = model.predict(test_x)
        testErr[k] = float(sum(testPred != test_y)) / len(test_y)

        print("train Err=", trainErr[k], "test Err=", testErr[k])
        print(" ")

    print("summary:")
    print("average train err =", numpy.mean(trainErr) * 100, "%")
    print("average test err =", numpy.mean(testErr) * 100, "%")


def run_ex2(m, X, Y):
    all_indexes = range(0, m)
    train_indexes = range(0, 300)
    test_indexes = list(set(all_indexes) - set(train_indexes))

    train_x, train_y = X[train_indexes, :], Y[train_indexes]
    test_x, test_y = X[test_indexes, :], Y[test_indexes]

    # set parameter
    alpha = 0.01
    start = time.time()
    model = LogisticRegression(train_x, train_y, alpha)
    iterations_cost_array = model.gradient_decent(500, 0.01)
    print("time for BGD train:", time.time() - start)

    print("last cost:", iterations_cost_array[len(iterations_cost_array) - 1])

    s_model = LogisticRegression(train_x, train_y, alpha)
    s_iterations_cost_array = s_model.stochastic_gradient_decent(500, 0.01)
    print("time for SGD train:", time.time() - start)

    draw(iterations_cost_array, s_iterations_cost_array)

    train_errors, test_errors = summarize_model(model, train_x, train_y, test_x, test_y)
    print("Using BGD: train Err=", train_errors, "test Err=", test_errors, '\n')

    train_errors, test_errors = summarize_model(s_model, train_x, train_y, test_x, test_y)
    print("Using SGD: train Err=", train_errors, "test Err=", test_errors, '\n')

def run_ex3(m, X, Y, num_of_features):
    print("Running EX3 using ", num_of_features, "features", '\n')

    all_indexes = range(0, m)
    train_indexes = range(0, 500)
    test_indexes = list(set(all_indexes) - set(train_indexes))

    train_x, train_y = X[train_indexes, :], Y[train_indexes]
    test_x, test_y = X[test_indexes, :], Y[test_indexes]

    # set parameter
    alpha = 0.01
    start = time.time()
    model = LogisticRegression(train_x, train_y, alpha, num_of_features)
    iterations_cost_array = model.gradient_decent(500, 0.01)
    print("time for BGD train:", time.time() - start)

    print("last cost:", iterations_cost_array[len(iterations_cost_array) - 1])

    s_model = LogisticRegression(train_x, train_y, alpha, num_of_features)
    s_iterations_cost_array = s_model.stochastic_gradient_decent(500, 0.01)
    print("time for SGD train:", time.time() - start)

    #draw(iterations_cost_array, s_iterations_cost_array)

    train_errors, test_errors = summarize_model(model, train_x, train_y, test_x, test_y)
    print("Using BGD: train Err=", train_errors, "test Err=", test_errors, '\n')

    train_errors, test_errors = summarize_model(s_model, train_x, train_y, test_x, test_y)
    print("Using SGD: train Err=", train_errors, "test Err=", test_errors, '\n')

    return iterations_cost_array, s_iterations_cost_array

def summarize_model(model, train_x, train_y, test_x, test_y):
    train_prediction = model.predict(train_x)
    train_errors = float(sum(train_prediction != train_y)) / len(train_y)

    test_prediction = model.predict(test_x)
    test_errors = float(sum(test_prediction != test_y)) / len(test_y)

    return train_errors, test_errors

def draw(points1, points2=[]):
    plt.plot(points1, 'r')
    plt.plot(points2, 'b')
    plt.ylabel("Cost")
    plt.ylim([0,1])
    plt.xlabel("Iteration")
    plt.show()


# Here the program starts
if __name__ == '__main__':
    # Read the data
    data = numpy.genfromtxt('input.csv', delimiter=',')
    # Classification is in the first column
    Y = data[:, 0]
    X = data[:, 1:]

    m = len(Y)

    # in order to run k-fold just un comment the row below and comment run_ex2(m, X, Y)
    #run_k_fold(m, X, Y)
    #run_ex2(m, X, Y)
    c1, cs1 = run_ex3(m, X, Y, 5)
    c2, cs2 = run_ex3(m, X, Y, 15)
    c3, cs3 = run_ex3(m, X, Y, 30)

    plt.plot(c1, 'r-')
    plt.plot(c2, 'r--')
    plt.plot(c3, 'r-.')
    plt.plot(cs1, 'b-')
    plt.plot(cs2, 'b--')
    plt.plot(cs3, 'b-.')
    plt.ylabel("Cost")
    plt.ylim([0, 1])
    plt.xlabel("Iteration")
    plt.show()




