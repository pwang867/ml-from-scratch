import numpy as np
import pandas as pd
import collections


class KNearestNeighbors(object):
    """
    KNearestNeighbor Classifier
    """
    def __init__(self, K: int = 3):
        self.K = K

    def fit(self, X, y):
        """ knn is lazy learning, it merely stores the training data. """
        self.X = X
        self.y = y
        return self

    def _dist(self, X, Y):
        P = np.add.outer(np.sum(X ** 2, axis=1), np.sum(Y ** 2, axis=1))
        N = np.dot(X, Y.T)
        return P - 2 * N

    def predict(self, X_test):
        D = self._dist(X_test, self.X)
        indices = np.argsort(D, axis=1)[:, :self.K]
        y_pred = np.zeros((X_test.shape[0],))
        for row in range(y_pred.shape[0]):
            counts = collections.Counter(self.y[indices[row]])
            y_pred[row] = sorted([_ for _ in counts], key=lambda c: counts[c])[-1]
        return y_pred


def main():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    iris = load_iris()
    X, X_test, y, y_test = train_test_split(iris["data"], iris["target"], stratify=iris["target"],
        test_size=0.2, random_state=42, shuffle=True)
    knn = KNearestNeighbors(3)
    knn = knn.fit(X, y)
    preds = knn.predict(X_test)
    print("Accuray: {:0.1f}%".format((preds == y_test).mean() * 100))
    print("Confusion Matrix: \n{}".format(confusion_matrix(y_test, preds)))

if __name__ == "__main__":
    main()