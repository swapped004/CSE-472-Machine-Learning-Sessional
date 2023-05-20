import numpy as np
class LogisticRegression:
    def __init__(self, params):
        """
        figure out necessary params to take as input
        :param params:
        """
        # todo: implement
        #initialize parameters

        #initialize learning rate
        self.learning_rate = params['learning_rate']

        #initialize number of iterations
        self.num_iterations = params['num_iterations']


    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def gradient(self, X, y, theta):
        m = y.size
        h = self.sigmoid(X.dot(theta))
        grad = (X.T.dot(h - y)) / m
        return grad

    def cost_function(self, X, y, theta):
        m = y.size
        h = self.sigmoid(X.dot(theta))
        J = -1 * (1 / m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y))
        return J

    def gradient_descent(self, X, y, theta, alpha, num_iters):
        # J_history = np.zeros(num_iters)
        for i in range(num_iters):
            theta -= alpha * self.gradient(X, y, theta)
            # J_history[i] = self.cost_function(X, y, theta)
        return theta


    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement

        #initialize theta and intercept
        intercept = np.ones((X.shape[0], 1)) 
        X = np.concatenate((intercept, X), axis=1)
        self.theta = np.zeros(X.shape[1])
        self.theta = self.theta.reshape(X.shape[1], 1)


        #gradient descent
        self.theta = self.gradient_descent(X, y, self.theta, self.learning_rate, self.num_iterations)
            


    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        :param X:
        :return:
        """
        # todo: implement
        #get predictions
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        predictions = self.sigmoid(X.dot(self.theta))

        #convert predictions to 0 and 1
        predictions = [1 if i >= 0.5 else 0 for i in predictions]

        return predictions
