import numpy as np

class LinearClassifier(object):

    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y = c
          means that X has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        loss_history = []
        n_samples, n_features = X.shape
        n_classes = y.max() + 1
        self.W = 0.001 * np.random.normal(loc=0.0, scale=1.0, size=(n_features, n_classes))

        steps = n_samples // batch_size
        for iter in range(num_iters):
            if iter % steps == 0:
                mask = np.shuffle(np.arange(n_samples))
                X = X[mask]
                y = y[mask]

            index = iter % steps
            #X_batch = X[index*batch_size:(index+1)*batch_size]
            #y_batch = y[index*batch_size:(index+1)*batch_size]
            indices = np.random.choice(n_samples, size=(batch_size,), replace=True)
            X_batch = X[indices]
            y_batch = y[indices]
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            self.W -= learning_rate * grad

            if verbose and iter % 100 == 0:
                print("iteration %d / %d: loss %f" % (iter, num_iters, loss))
        return loss_history


    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """

        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        y_pred = np.zeros(X.shape[0])
        y_pred = np.argmax(X.dot(self.W), axis=1)
        return y_pred
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        batch_size = X_batch.shape[0]
        scores = X_batch.dot(self.W)
        scores -= scores.max(axis=1, keepdims=True)
        z = np.exp(scores)
        probs = z / z.sum(axis=1, keepdims=True)
        # cross entropy loss
        cross_entropy_loss = -np.log(scores[:, y_batch]).sum() / batch_size
        reg_loss = 0.5 * reg * (self.X * self.X).sum() / batch_size
        loss = cross_entropy_loss + reg_loss

        # gradient
        derivative = probs
        derivative[:, y_batch] -= 1.0
        grad = (np.dot(X_batch.T, derivative) + reg * self.W) / batch_size

        return (loss, grad)