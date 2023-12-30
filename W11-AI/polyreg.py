import numpy as np

#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(self, degree=1, regLambda=1E-8):
        '''
        Constructor
        '''
        self.degree = degree
        self.regLambda = regLambda
        self.theta = None  # This will store the model parameters after training

    def polyfeatures(self, X, degree):
        '''
        Expands the given X into an n * d array of polynomial features of
            degree d.

        Returns:
            A n-by-d numpy array, with each row comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not include the zero-th power.

        Arguments:
            X is an n-by-1 column numpy array
            degree is a positive integer
        '''
        n = len(X)
        poly_X = np.zeros((n, degree))

        for i in range(1, degree + 1):
            poly_X[:, i - 1] = np.power(X, i).flatten()

        return poly_X

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-1 array
            y is an n-by-1 array
        Returns:
            No return value
        Note:
            You need to apply polynomial expansion and scaling
            at first
        '''
        # Apply polynomial expansion
        X_poly = self.polyfeatures(X, self.degree)

        # Add a small perturbation to avoid singularity
        X_poly += 1e-10 * np.random.randn(*X_poly.shape)

        # Add a bias term to X_poly
        X_poly = np.column_stack((np.ones(len(X_poly)), X_poly))

        # Train the model using linear regression
        # Calculate parameters using the normal equation
        # (X^T*X + regLambda*I)^(-1)*X^T*y
        A = np.dot(X_poly.T, X_poly) + self.regLambda * np.identity(X_poly.shape[1])
        b = np.dot(X_poly.T, y)
        self.theta = np.linalg.solve(A, b)

    def predict(self, X):
        '''
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        '''
        # Apply polynomial expansion
        X_poly = self.polyfeatures(X, self.degree)

        # Add a bias term to X_poly
        X_poly = np.column_stack((np.ones(len(X_poly)), X_poly))

        # Use the trained model parameters to make predictions
        return np.dot(X_poly, self.theta)

#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------

def learningCurve(Xtrain, Ytrain, Xtest, Ytest, regLambda, degree):
    '''
    Compute learning curve
        
    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        regLambda -- regularization factor
        degree -- polynomial degree
        
    Returns:
        errorTrains -- errorTrains[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTests -- errorTrains[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]
        
    Note:
        errorTrains[0:1] and errorTests[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    '''
    
    n = len(Xtrain)
    
    errorTrain = np.zeros((n))
    errorTest = np.zeros((n))
    for i in range(2, n):
        Xtrain_subset = Xtrain[:(i+1)]
        Ytrain_subset = Ytrain[:(i+1)]
        model = PolynomialRegression(degree, regLambda)
        model.fit(Xtrain_subset, Ytrain_subset)
        
        predictTrain = model.predict(Xtrain_subset)
        err = predictTrain - Ytrain_subset
        errorTrain[i] = np.multiply(err, err).mean()
        
        predictTest = model.predict(Xtest)
        err = predictTest - Ytest
        errorTest[i] = np.multiply(err, err).mean()
    
    return (errorTrain, errorTest)
