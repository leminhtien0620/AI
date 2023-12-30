
import numpy as np


class LogisticRegression:

    def __init__(self, alpha=0.01, regLambda=0.01, epsilon=0.0001, maxNumIters=10000, threshold=0.6):
        """
        Constructor
        """
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
        self.theta = None
        self.threshhold = threshold

    def computeCost(self, theta, X, y, regLambda):
        """
        Computes the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
        """
        n, d = X.shape
        cost = (-y.T * np.log(self.sigmoid(X * theta)) - (1.0 - y).T * np.log(
            1.0 - self.sigmoid(X * theta))) / n + self.regLambda / (2.0 * n) * (theta.T * theta)
        return cost.item((0, 0))

    def computeGradient(self, theta, X, y, regLambda):
        """
        Computes the gradient of the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        """
        n, d = X.shape
        # Gradient of the cost function
        gradient = (X.T * (self.sigmoid(X * theta) - y) + regLambda * theta) / n
        # Not to regularize the theta[0]
        gradient[0] = sum(self.sigmoid(X * theta) - y) / n
        return gradient

    def fit(self, X, y):
        """
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        """
        a, b = X.shape
        # add the 1's feature
        # Translates slice objects to concatenation along the second axis.
        X = np.c_[np.ones((a, 1)), X]

        self.theta = np.mat(np.random.rand(b + 1, 1))
        # print('init theta', self.theta)
        old_theta = self.theta
        new_theta = self.theta

        i = 0
        while i < self.maxNumIters:
            # print('Iter:', i + 1)
            new_theta = old_theta - self.alpha * self.computeGradient(new_theta, X, y, self.regLambda)
            print('New Theta in ', i + 1, ': ', new_theta)
            if self.hasConverged(new_theta, old_theta):
                self.theta = new_theta
                return
            else:
                old_theta = np.copy(new_theta)
                i = i + 1
                cost = self.computeCost(new_theta, X, y, self.regLambda)
                print('Cost: ', cost)
        self.theta = new_theta
        print (self.theta)

    def hasConverged(self, new_theta, old_theta):
        if np.linalg.norm(new_theta - old_theta) <= self.epsilon:
            return True
        else:
            return False

    def predict(self, X):
        """
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        """
        a, b = X.shape
        X_slice = np.c_[np.ones((a, 1)), X]
        p = np.array(self.sigmoid(np.dot(X_slice, self.theta)))
        predict = p >= self.threshhold
        # print(X, ' ', self.theta)
        return predict

    def sigmoid(self, Z):
        """
        Computes the sigmoid function 1/(1+exp(-z))
        """
        return 1.0 / (1.0 + np.exp(-Z))