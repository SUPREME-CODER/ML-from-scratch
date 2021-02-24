import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:

    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
 

    def predict(self, X):
        y_approximated = np.dot(X, self.weights) + self.bias
        return y_approximated

    def plot(self, X_train, y_train, X_test, y_test, X, y_pred):
        cmap = plt.get_cmap('viridis')
        fig = plt.figure(figsize=(8,6))
        m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
        m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
        plt.plot(X, y_pred, color='black', linewidth=2, label="Prediction")
        plt.show()

