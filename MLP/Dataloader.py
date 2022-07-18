import numpy as np


class Dataloader:
    def __init__(self, train_input_path, train_label_path, test_input_path,
                 test_label_path, classes, batch_size=None, norm=True):
        self.bs = batch_size
        self.need_norm = norm
        self.classes = classes

        with open(train_input_path) as f:
            x = f.read().splitlines()
            self.x = np.array(list(map(lambda xi: xi.split(' '), x))).astype(np.float64)
        with open(train_label_path) as f:
            y = f.read().splitlines()
            self.y = np.array(list(map(self.onehot, y)))
        assert len(x) == len(y), "请确认样本和label数量相同"

        with open(test_input_path) as f:
            x = f.read().splitlines()
            self.test_x = np.array(list(map(lambda xi: xi.split(' '), x))).astype(np.float64)
        with open(test_label_path) as f:
            y = f.read().splitlines()
            self.test_y = np.array(list(map(self.onehot, y)))
        assert len(x) == len(y), "请确认样本和label数量相同"

    def onehot(self, label):
        onehot_label = np.zeros(self.classes)
        onehot_label[int(label)] = 1
        return onehot_label

    def norm(self):
        self.mean = self.x.mean(axis=0)
        self.std = self.x.std(axis=0)
        self.x_norm = (self.x - self.mean) / (self.std + 1e-16)
        self.test_x_norm = (self.test_x - self.mean) / (self.std + 1e-16)

    def load_train_data(self):
        sample_index = np.random.choice(np.arange(self.x.shape[0]), size=self.bs)
        if self.need_norm:
            return self.x_norm[sample_index, :].T, self.y[sample_index, :].T
        else:
            return self.x[sample_index, :].T, self.y[sample_index, :].T

    def load_test_data(self):
        if self.need_norm:
            self.norm()
            return self.test_x_norm.T, self.test_y.T
        else:
            return self.test_x.T, self.test_y.T
