import numpy

'''
general functions for probability calculations and MI 
'''
def calc_prob(vec):
    """
    compute P(V=v) for each value in the given vector
    :param vec: the vector to compute 
    :return:P(vec)
    """
    unique_values, counts = numpy.unique(vec, return_counts=True)

    return numpy.divide(counts, len(vec)), unique_values, counts

def calc_mutual_prob(X, Y, x_values, y_values):
    """
    compute P(X,Y) for each value in the given vectors
    :param X: the X set
    :param Y: the Y set
    :param x_values: distinct x values  
    :param y_values: distinct y values
    :return: P(X,Y)
    """

    probs = []
    for x_val in x_values:
        such_y = Y[X == x_val]

        x_and_y = [len(such_y[such_y == yval]) for yval in y_values]

        probs.append(numpy.divide(x_and_y, len(Y)))  # p(xi,y)

    return numpy.array(probs)

def computeMI(x, y):
    """
    compute the mutual information rank for the given vectors
    :param x: X
    :param y: Y
    :return: mutual information rank
    """
    Px, x_values, x_values_count = calc_prob(x)  # P(x)
    Py, y_values, y_values_count = calc_prob(y)  # P(y)

    mutual_probs = calc_mutual_prob(x, y, x_values, y_values) # P(x,y)

    sum_mi = 0.0
    for i in range(len(x_values)):
        t = numpy.divide(mutual_probs[i], numpy.multiply(Py, Px[i]))  # P(x,y)/(P(x)*P(y))
        sum_mi += numpy.dot(mutual_probs[i][t > 0], numpy.log2(t[t > 0]))  # sum ( P(x,y)* log(P(x,y)/( P(x)*P(y)) )
    return sum_mi


class LogisticRegression(object):
    """
    This class provides an implementation of logistic regression using gradient decent algorithm
    By: Arye Kogan and Tomer Achdut
    """
    def __init__(self, X, Y, alpha=0.0005, num_of_features = None):
        """
        Initialize the model
        :param X: The data features
        :param Y: The corresponding data classification
        :param alpha: The step being taken for each gradient decent iteration or "learning rate"
        """
        x = numpy.array(X)
        m, n = x.shape

        # normalize the data using standard score
        self.xMean = numpy.mean(x, axis=0)
        self.xStd = numpy.std(x, axis=0)
        x = (x - self.xMean) / self.xStd

        # add x0 constant values as 1s at the first column of the feature
        const = numpy.array([1] * m).reshape(m, 1)
        self.X = numpy.append(const, x, axis=1)
        self.Y = numpy.array(Y)
        self.n_most_mis = None
        if num_of_features is not None:
            self.set_n_most_influence_features(num_of_features)
            n = self.X.shape[1]-1

        self.alpha = alpha
        self.theta = numpy.array([0.0] * (n + 1))

    @staticmethod
    def sigmoid(z):
        """
        The logistic function which transforms (normalize) values from R to [0, 1]:
        g(z) = 1/(1+exp(-z)
        """
        return 1.0 / (1.0 + numpy.exp(-z))

    def hypothesis(self, x):
        """
        calculate current model's hypothesis based on the current theta coefficients
        :param x: features input vector
        :return: current model's hypothesis output vector
        """
        return self.sigmoid(numpy.dot(x, self.theta))

    def cost(self):
        """
        cost function for single iteration of gradient decent algorithm as mentioned by Andrew Ng
        source: https://stats.stackexchange.com/questions/251982/stochastic-gradient-descent-for-regularized-logistic-regression
        :return: the cost in the current iteration
        """
        m, n = self.X.shape
        h_theta = self.hypothesis(self.X)
        epsilon = numpy.finfo(float).eps
        prob_y_1 = -1 * self.Y * numpy.log(h_theta)
        prob_y_0 = (1.0 - self.Y + epsilon) * numpy.log(1.0 - h_theta + epsilon)

        return sum(prob_y_1 - prob_y_0) / m

    def gradient_decent_step(self, j, h_theta):
        """
        performs a single step and updating model's thetas
        :param j: the current theta index being calculated
        :param h_theta: the hypothesis 
        """
        m, n = self.X.shape
        self.theta[j] = self.theta[j] - self.alpha * sum((h_theta[1:] - self.Y[1:]) * self.X[1:, j]) / m

    def gradient_decent(self, max_iterations, threshold=0.01):
        """
        Running the gradient decent algorithm
        in order to find theta values in which the error is minimal 
        :param max_iterations: The maximum number of iterations to run
        :param threshold: The threshold for the cost which will satisfy the coefficients   
        :return: the cost in each iteration of the gradient decent
        """
        m, n = self.X.shape
        cost_array = numpy.array([0.0] * max_iterations)

        for i in range(0, max_iterations):
            h_theta = self.hypothesis(self.X)

            for j in range(0, n):
                self.gradient_decent_step(j, h_theta)

            # calculate cost and exit if threshold is reached
            cost_array[i] = self.cost()
            if cost_array[i] <= threshold:
                return cost_array[0:i]

        return cost_array

    def stochastic_gradient_decent_step(self, i, h_theta):
        """
        performs a single step and updating model's thetas
        :param i: 
        :param h_theta: the hypothesis 
        """
        m, n = self.X.shape
        self.theta = self.theta - self.alpha * (h_theta[i] - self.Y[i]) * self.X[i, :]

    def stochastic_gradient_decent(self, max_iterations, threshold=0.01):
        """
        Running the stochastic gradient decent algorithm
        in order to find theta values in which the error is minimal 
        :param max_iterations: The maximum number of iterations to run
        :param threshold: The threshold for the cost which will satisfy the coefficients   
        :return: the cost in each iteration of the gradient decent
        """
        # random the data set
        m, n = self.X.shape
        cost_array = numpy.array([0.0] * max_iterations)

        for iteration in range(0, max_iterations):
            h_theta = self.hypothesis(self.X)

            for i in range(1, m):
                self.stochastic_gradient_decent_step(i, h_theta)

            # calculate cost and exit if threshold is reached
            cost_array[iteration] = self.cost()

            #if cost_array[iteration] <= threshold:
                #return cost_array[0:iteration]

        return cost_array

    def predict(self, data_set):
        """
        Predicts the classification of the given data set based on the current model
        :param data_set: the features
        :return: predicted classification vector
        """
        m, n = data_set.shape
        # normalize the data using model's standard score
        x = numpy.array(data_set)
        x = (x - self.xMean) / self.xStd
        # add x0 constant values as 1s at the first column of the feature
        const = numpy.array([1] * m).reshape(m, 1)
        data_set = numpy.append(const, x, axis=1)
        if self.n_most_mis is not None:
            data_set = numpy.take(data_set, self.n_most_mis, axis=1)

        prediction = self.sigmoid(numpy.dot(data_set, self.theta))
        numpy.putmask(prediction, prediction >= 0.5, 1.0)
        numpy.putmask(prediction, prediction < 0.5, 0.0)

        return prediction

    def set_n_most_influence_features(self, n):
        """
        Updating the trining set to use the n most influence features by MI
        :param n: the desired number of features
        """
        mi = numpy.array([computeMI(xi, self.Y) for xi in numpy.array(self.X).transpose()])
        self.n_most_mis = (-mi).argsort()[:n]
        self.X = numpy.take(self.X, self.n_most_mis, axis=1)

