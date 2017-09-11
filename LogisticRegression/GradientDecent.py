from numpy import *

def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]







'''
    # gradient descend
    def _gradientDescend(self, iters):
        """
        gradient descend:
        X: feature matrix
        Y: response
        theta: predict parameter
        alpha: learning rate
        lam: lambda, penality on theta
       """

        m, n = self.X.shape
        cost_array = numpy.array([0.0] * iters)

        for i in range(0, iters):
            theta_temp = self.theta
            print(self.X[:, 0])
            # update theta[0]
            h_theta = self.sigmoid(numpy.dot(self.X, self.theta))
            diff = h_theta - self.Y
            self.theta[0] = theta_temp[0] - self.alpha * (1.0 / m) * sum(diff * self.X[:, 0])

            for j in range(1, n):
                val = theta_temp[j] - self.alpha * (1.0 / m) * (sum(diff * self.X[:, j]) + self.lam * m * theta_temp[j])
                # print val
                self.theta[j] = val
            # calculate cost and print
            cost = self.calculate_cost()
            cost_array[i] = cost
            if self.printIter:
                print ("Iteration", i, "\tcost=", cost)
                print ("theta", self.theta)

        return cost_array
'''