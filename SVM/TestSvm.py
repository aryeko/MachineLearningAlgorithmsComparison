import numpy as np
import MyKernels
from MySVM import SVM
import time
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from sklearn.decomposition import PCA

def run_private_svm(x_train, y_train, x_test, y_test, ker=MyKernels.gaussian_kernel, c=1.0):
    print("STARTING PRIVATE SVM")
    total_start = time.time()
    start = time.time()
    print("Starting train for SVM with %s kernel and %.3f as the penalty parameter" % (ker.__name__, c))
    clf = SVM(C=c, kernel=ker)
    clf.fit(x_train, y_train)
    print("Time for SVM train: %.3f seconds" % (time.time() - start))

    start = time.time()
    print("Starting test")
    y_predict = clf.predict(x_test)
    correct = np.sum(y_predict == y_test)
    print("%d out of %d predictions correct - sucess rate: %.2f %%" % (
        correct, len(y_predict), (correct / len(y_predict)) * 100))
    print("Time for SVM test: %.3f seconds" % (time.time() - start))
    print("Total time is: %.3f seconds" % (time.time() - total_start))

def plot_3d(X1_train, X2_train):
    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')

    n = 100

    ax.scatter(X2_train[:, 0], X2_train[:, 1], X2_train[:, 2], c="r", marker="o")
    ax.scatter(X1_train[:, 0], X1_train[:, 1], X1_train[:, 2], c="b", marker="o")

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    pl.axis("tight")
    pl.show()


def draw_data_3d(X_set, y):
    print("Getting the 3 most influencing features using PCA")
    Xt = np.transpose(X_set)
    print("Shape before PCA is ", X_set.shape)
    print("Shape after transpose is ", Xt.shape)
    pca = PCA(n_components=3)
    fit = pca.fit(Xt)
    # summarize components
    print("Explained Variance: %s" % fit.explained_variance_ratio_)
    print(fit.components_)
    X_decomposed = np.transpose(fit.components_)
    print("New shape is ", X_decomposed.shape)

    plot_3d(X_decomposed[y == 1], X_decomposed[y == -1])


def get_trained_svm(ker, c, x_train, y_train):
    start = time.time()
    print("Starting train for SVM with %s kernel and %.3f as the penalty parameter" % (ker, c))
    clf = svm.SVC(kernel=ker, C=c, degree=2)
    clf.fit(x_train, y_train)
    print("Time for SVM train: %.3f seconds" % (time.time() - start))
    return clf


def test_svm(clf, x_test, y_test):
    start = time.time()
    print("Starting test")
    y_predict = clf.predict(x_test)
    correct = np.sum(y_predict == y_test)
    print("%d out of %d predictions correct - sucess rate: %.2f %%" % (
    correct, len(y_predict), (correct / len(y_predict)) * 100))
    print("Time for SVM test: %.3f seconds" % (time.time() - start))


def run_svm_e2e(x_train, y_train, x_test, y_test, kernel="rbf", C=1.0):
    start = time.time()
    clf = get_trained_svm(kernel, C, x_train, y_train)
    test_svm(clf, x_test, y_test)
    print("Total time is: %.3f seconds" % (time.time() - start))

if __name__ == "__main__":
    # Read the data
    print("Reading the data from csv")
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

    # Plot the data on 3d in order to visualize how it's distributes
    draw_data_3d(X, Y)

    ###############################################################################
    ########                          OUR SVM                              ########
    ###############################################################################


    """
    Test our's SVM with 'linear' kernel
    """
    run_private_svm(x_train, y_train, x_test, y_test, MyKernels.linear_kernel, 0.1)
    print("\n****************************\n")
    run_private_svm(x_train, y_train, x_test, y_test, MyKernels.linear_kernel, 1.0)
    print("\n****************************\n")
    run_private_svm(x_train, y_train, x_test, y_test, MyKernels.linear_kernel, 5)

    """
    Test our's SVM with 'gaussian' kernel
    """
    run_private_svm(x_train, y_train, x_test, y_test, MyKernels.gaussian_kernel, 0.1)
    print("\n****************************\n")
    run_private_svm(x_train, y_train, x_test, y_test, MyKernels.gaussian_kernel, 1.0)
    print("\n****************************\n")
    run_private_svm(x_train, y_train, x_test, y_test, MyKernels.gaussian_kernel, 5)

    """
    Test our's SVM with 'sigmoid' kernel
    """
    run_private_svm(x_train, y_train, x_test, y_test, MyKernels.sigmoid_kernel, 0.1)
    print("\n****************************\n")
    run_private_svm(x_train, y_train, x_test, y_test, MyKernels.sigmoid_kernel, 1.0)
    print("\n****************************\n")
    run_private_svm(x_train, y_train, x_test, y_test, MyKernels.sigmoid_kernel, 5)


    ###############################################################################
    ########                        SKLEARN SVM                            ########
    ###############################################################################

    """
    Test sklearn's SVM with 'linear' kernel
    """
    run_svm_e2e(x_train, y_train, x_test, y_test, "linear", 0.1)
    print("\n****************************\n")
    run_svm_e2e(x_train, y_train, x_test, y_test, "linear", 1.0)
    print("\n****************************\n")
    run_svm_e2e(x_train, y_train, x_test, y_test, "linear", 5)

    """
    Test sklearn's SVM with 'gaussian' kernel
    """
    run_svm_e2e(x_train, y_train, x_test, y_test, "rbf", 0.1)
    print("\n****************************\n")
    run_svm_e2e(x_train, y_train, x_test, y_test, "rbf", 1.0)
    print("\n****************************\n")
    run_svm_e2e(x_train, y_train, x_test, y_test, "rbf", 5)

    """
    Test sklearn's SVM with 'sigmoid' kernel
    """
    run_svm_e2e(x_train, y_train, x_test, y_test, "sigmoid", 0.1)
    print("\n****************************\n")
    run_svm_e2e(x_train, y_train, x_test, y_test, "sigmoid", 1.0)
    print("\n****************************\n")
    run_svm_e2e(x_train, y_train, x_test, y_test, "sigmoid", 5)