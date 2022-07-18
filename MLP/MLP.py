import numpy as np
import logging
import  matplotlib.pylab as plt
from tqdm import tqdm
from Dataloader import Dataloader
from utils import visualiztion

# logging.basicConfig(level=logging.INFO)


class MLP:
    def __init__(self, input_dim, output_dim, hidden_units, hidden_layers, lr,
                 actv_func='sigmoid', loss_func='mse', regular='none', alpha=0., seed=0):
        np.random.seed(seed=seed)
        self.lr = lr
        self.actv_func = actv_func
        self.loss_func = loss_func
        self.regular = regular
        self.alpha =alpha
        logging.info('---------Initializing Model---------')
        logging.info('lr:{}, input dim:{}, output dim:{}, hidden units:{}, hidden layers:{}, '
                     'activation function:{}, loss function:{}, regular:{}, alpha:{}'
                     .format(lr, input_dim, output_dim, hidden_units, hidden_layers, actv_func, loss_func, regular, alpha))

        self.softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.tanh = lambda x: (np.exp(x)-np.exp(-x)) / (np.exp(x)+np.exp(-x))

        # 随机初始化权重。只有第一层和最后一层参数维度不同，中间层都是hidden_units * hidden_units的维度
        self.weights = [np.random.randn(input_dim, hidden_units)]
        self.biases = [np.random.randn(hidden_units, 1)]
        for i in range(hidden_layers-1):
            self.weights.append(np.random.randn(hidden_units, hidden_units))
            self.biases.append(np.random.randn(hidden_units, 1))
        self.weights.append(np.random.randn(hidden_units, output_dim))
        self.biases.append(np.random.randn(output_dim, 1))
        logging.info('---------Start Training---------')

    def activation(self, input, activation: str):
        if activation == 'sigmoid':
            return self.sigmoid(input)
        if activation == 'softmax':
            return self.softmax(input)
        if activation == 'relu':
            output = input.copy()
            output[output<0] = 0
            return output
        if activation == 'tanh':
            return self.tanh(input)

    # 从激活层的输出反推激活层导数
    def activation_grad(self, output, activation:str):
        if activation == 'sigmoid' or activation == 'softmax':
            return output * (1 - output)
        if activation == 'relu':
            return (output > 0).astype(np.float64)
        if activation == 'tanh':
            return 1 - output**2

    def loss(self, out, label, loss_type:str):
        if loss_type == 'mse':
            return 0.5*np.sum((out - label)**2, axis=0)
        if loss_type == 'cross_entropy':
            return np.sum(-label * np.log(out), axis=0)

    def loss_grad(self, out, label, loss_type:str):
        if loss_type == 'mse':
            return out - label
        if loss_type == 'cross_entropy':
            return -label / (out)

    def regular_grad(self, weight, regular_type:str):
        if regular_type == 'none':
            return 0.
        if regular_type == 'l2':
            return weight
        if regular_type == 'l1':
            return np.sign(weight)

    def forward(self, input, onehot_label):
        # input维度是M*1
        self.outputs = [input]  # 为便于计算，output包含输入在内的所有层
        self.errors = []
        self.gradient = []

        # 前向传播，保存每一层输出，最后一层用softmax
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            if i == len(self.weights) - 1:
                output = self.activation(np.dot(self.outputs[-1].T, weight).T + bias, activation='softmax')
            else:
                output = self.activation(np.dot(self.outputs[-1].T, weight).T + bias, activation=self.actv_func)
            self.outputs.append(output)

        N = onehot_label.shape[-1]
        max_index = self.outputs[-1].argmax(axis=0)
        preds = np.zeros_like(self.outputs[-1])  # C*N
        preds[max_index, range(N)] = 1  # 将每一列对应最大输出的位置置1

        acc = 0
        for i in range(N):
            pred = preds[:,i]
            label = onehot_label[:,i]
            acc += float((pred==label).all())
        acc /= N

        # 计算输出层error，反向计算errors, self.errors存的顺序是由输出到输入
        if self.loss_func == 'mse':
            self.errors.append(self.loss_grad(self.outputs[-1], onehot_label, loss_type=self.loss_func)
                               * self.activation_grad(self.outputs[-1], activation='softmax'))
        # 交叉熵的loss_grad和activation_grad可以化简，防止溢出
        if self.loss_func == 'cross_entropy':
            self.errors.append(self.outputs[-1] - onehot_label)

        # 计算隐藏层error，output只要隐藏层的，weight不要第一层的
        for (output, weight) in zip(self.outputs[-2:0:-1], self.weights[:0:-1]):
            error = np.dot(weight, self.errors[-1]) * self.activation_grad(output, activation=self.actv_func)
            self.errors.append(error)

        # 计算反向传播梯度
        self.errors = self.errors[::-1]  # 反转后self.errors存的顺序是由输入到输出
        for (error, weight, output) in zip(self.errors, self.weights, self.outputs[:-1]):
            gradient = np.dot(output, error.T) / N + self.regular_grad(weight, self.regular) * self.alpha
            self.gradient.append(gradient)

        return preds, acc

    def backward(self):
        tmp_weights = []
        tmp_biases = []
        for (weight, grad, bias, error) in zip(self.weights, self.gradient, self.biases, self.errors):
            tmp_weights.append(weight - self.lr * grad)
            tmp_biases.append(bias - self.lr * np.sum(error, axis=1, keepdims=True) / error.shape[1])
        self.weights = tmp_weights
        self.biases = tmp_biases

    def eval(self, inputs, onehot_labels):
        # inputs:M*N, onehot_labels:C*N
        self.outputs = [inputs]

        # 前向传播，保存每一层输出，最后一层用softmax
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            if i == len(self.weights) - 1:
                output = self.activation(np.dot(self.outputs[-1].T, weight).T + bias, activation='softmax')
            else:
                output = self.activation(np.dot(self.outputs[-1].T, weight).T + bias, activation=self.actv_func)
            self.outputs.append(output)

        N = onehot_labels.shape[-1]
        loss = self.loss(self.outputs[-1], onehot_labels, loss_type=self.loss_func).sum() / N
        max_index = self.outputs[-1].argmax(axis=0)
        preds = np.zeros_like(self.outputs[-1], dtype=int)  # C*N
        preds[max_index, range(N)] = 1  # 将每一列对应最大输出的位置置1
        acc = 0
        for i in range(N):
            pred = preds[:,i]
            label = onehot_labels[:,i]
            acc += float((pred==label).all())
        acc /= N
        return loss, acc


# Model instantiation
mlp = MLP(input_dim=2, output_dim=2, hidden_units=5, hidden_layers=5, lr=1e-1,
          actv_func='relu', loss_func='mse', regular='none', alpha=1e-3, seed=1)

# Load Data
data_loader = Dataloader('./Exam/train/x.txt', './Exam/train/y.txt', './Exam/test/x.txt', './Exam/test/y.txt',
                         classes=2, norm=True, batch_size=8)
# data_loader = Dataloader('./Iris/train/x.txt', './Iris/train/y.txt', './Iris/test/x.txt', './Iris/test/y.txt',
#                          classes=3, norm=True, batch_size=8)
test_inputs, test_labels = data_loader.load_test_data()

# Train
step_ls, loss_ls, train_acc_ls, test_acc_ls = [], [], [], []
plt.ion()
plt.figure(figsize=(12, 4))
interval = 100
iteration = tqdm(range(10000))
for i in iteration:
    inputs, labels = data_loader.load_train_data()
    mlp.forward(inputs, labels)
    mlp.backward()
    if i % interval == interval-1:
        _, train_acc = mlp.forward(inputs, labels)
        loss, acc = mlp.eval(test_inputs, test_labels)
        step_ls.append(i)
        loss_ls.append(loss)
        train_acc_ls.append(train_acc)
        test_acc_ls.append(acc)
        visualiztion(step_ls, loss_ls, train_acc_ls, test_acc_ls, test_inputs, test_labels, mlp)
        iteration.set_description(f'Batch {i}')
        iteration.set_postfix(loss=loss, acc=acc)
