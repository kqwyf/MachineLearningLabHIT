import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class logistic_model:
    def __init__(self, size, l=0):
        self.size = size + 1
        self.l = l
        self.w = None
    def loss(self, test_set):
        """
        Calculate the loss of this model on a given dataset.

        Parameters
        ----------
        test_set : array_like
            The dataset to be calculated.

        Returns
        -------
        result : float
            The loss on this dataset.
        """
        x, y = self._convert(test_set)
        N = np.shape(x)[0] if len(np.shape(x))>1 else 1
        X = sigmoid(self.w @ x.T)
        return -np.sum(y * np.log(X) + (1-y) * np.log(1-X)) / N + (self.l/(2*self.size)) * self.w @ self.w
    def dloss(self, test_set):
        """
        Calculate the first derivative(gradient) of the loss of this model on a given dataset.

        Parameters
        ----------
        test_set : array_like
            The dataset to be calculated.

        Returns
        -------
        result : array_like
            The first derivative(gradient) of the loss on this dataset as a float vector.
        """
        x, y = self._convert(test_set);
        N = np.shape(x)[0] if len(np.shape(x))>1 else 1
        return x.T @ (sigmoid(self.w @ x.T) - y) / N + (self.l/self.size) * self.w
    def ddloss(self, test_set):
        """
        Calculate the second derivative(gradient) of the loss of this model on a given dataset.

        In other word, calculate the Hessian matrix of the loss.

        Parameters
        ----------
        test_set : array_like
            The dataset to be calculated.

        Returns
        -------
        result : array_like
            The second derivative(gradient) of the loss on this dataset as a float vector.
        """
        x, y = self._convert(test_set);
        N = np.shape(x)[0] if len(np.shape(x))>1 else 1
        X = sigmoid(self.w @ x.T)
        R = np.diag(X * (1-X))
        return x.T @ (R/N + (self.l/self.size) * np.eye(N)) @ x
    def train_by_newton(self, train_set, num_step, w=None):
        """
        Train the model by the Newton method.

        Parameters
        ----------
        train_set : array_like
            The dataset for training as a tuple. The first element of the tuple is an N*M
            matrix containing N samples, while the second element is the labels of the
            samples.
        num_step : int
            The number of training steps.
        w : array_like
            The initial value for w.
        """
        if w is None:
            w = np.zeros(self.size)
        self.w = w
        print("Training...")
        for step in range(1, num_step+1):
            print("\rtraining process: %.0f%%    loss: %f"%(step/num_step*100, self.loss(train_set)), end='')
            self.w = self.w - np.linalg.inv(self.ddloss(train_set)) @ self.dloss(train_set)
        print("\nTraining finished.    loss: %f"%self.loss(train_set))
    def train_by_gradient_descent(self, train_set, num_step, learning_rate, w=None):
        """
        Train the model by the gradient descent method.

        Parameters
        ----------
        train_set : array_like
            The dataset for training as a tuple. The first element of the tuple is an N*M
            matrix containing N samples, while the second element is the labels of the
            samples.
        num_step : int
            The number of training steps.
        learning_rate : float
            The learning rate of the training process.
        w : array_like
            The initial value for w.
        """
        if w is None:
            w = np.zeros(self.size)
        self.w = w
        print("Training...")
        for step in range(1, num_step+1):
            print("\rtraining process: %.0f%%    loss: %f"%(step/num_step*100, self.loss(train_set)), end='')
            d = self.dloss(train_set)
            d2 = np.linalg.norm(d, 2)
            self.w -= learning_rate * (d/d2)
        print("\nTraining finished.    loss: %f"%self.loss(train_set))
    def predict(self, x):
        """
        Classify one or more observed datapoint(s).

        Parameters
        ----------
        x : array_like
            The observed datapoint(s) as a 1-d or 2-d array.
        """
        x = np.asarray(x)
        tmp = np.ones((np.shape(x)[0],1)) if len(np.shape(x))>1 else np.ones(1)
        x = np.hstack((x, tmp))
        return x @ self.w > 0
    def _convert(self, dataset):
        x, y = np.asarray(dataset[0]), np.asarray(dataset[1])
        tmp = np.ones((np.shape(x)[0],1)) if len(np.shape(x))>1 else np.ones(1)
        x = np.hstack((x, tmp))
        return x, y

