import torch
from torch import nn
import collections
import numpy as np

class MLP(object):
    def __init__(self, hiddens):
        self.hiddens = hiddens
        self.model = None

    def create_model(self, input_dim, output_dim, reg=0.01, lr=0.1):
        steps = collections.OrderedDict()
        dims = [input_dim] + self.hiddens
        opt_params = []
        for i in range(1, len(dims)):
            steps["linear%d" % i] = nn.Linear(dims[i-1], dims[i])
            steps["relu%d" % i] = nn.ReLU()
            steps["dropout%d" % i] = nn.Dropout(0.2)
            opt_params.append({"params": steps["linear%d"%i].weights, "weight_decay": reg})
            opt_params.append({"params": steps["linear%d"%i].bias})

        steps["output"] = nn.Linear(self.hiddens[-1], output_dim)
        self.model = nn.Sequential(steps)
        self.optimizer = torch.optim.SGD(opt_params, lr=lr)
        return self

    def data_iter(self, X, y, batch_size):
        n_samples = X.shape[0]
        batches = n_samples // batch_size
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        for i in range(batches):
            mask = indices[i*batch_size:(i+1)*batch_size]
            yield X[mask], y[mask]

    def train(self, X, y, num_epochs=100, lr=0.1, batch_size=256):
        n_samples, n_features = X.shape
        n_classes = y.max() + 1
        self.create_model(n_features, n_classes)
        #optimizer = torch.optim.SGD(self.model.parameter(), lr=lr)
        loss_func = nn.CrossEntropyLoss()

        loss_history = []
        for epoch in range(num_epochs):
            for xb, yb in self.data_iter(X, y, batch_size):
                loss = loss_func(self.model(xb), yb)
                loss_history(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print("epoch {} loss: {}".format(epoch, loss_history[-1]))

