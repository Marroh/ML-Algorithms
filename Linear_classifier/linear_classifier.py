import os
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


def load_data(dir):
    x_file = os.path.join(dir, 'x.txt')
    y_file = os.path.join(dir, 'y.txt')

    with open(x_file) as f:
        x = f.read().splitlines()
        x = np.array(list(map(lambda xi: xi.split(' '), x))).astype(np.float64)
    with open(y_file) as f:
        y = f.read().splitlines()
        y = np.array(list(map(float, y)))
    assert len(x) == len(y), "请确认样本和label数量相同"

    # Convert X to an augmented matrix, convert both X and y to Mat for convenient computation
    X = np.mat(np.append(np.ones((x.shape[0], 1)), x, axis=1))
    y = np.mat(y.reshape(y.shape[0], 1))
    return X, y


class LogisticRegression():
    def __init__(self, features, labels, test_features, test_labels, lr=1e-3, bs=10, seed=0, thresh=0.5, show_period=100, show=True):
        '''

        :param features: 线性模型中的X，形式上为增广矩阵
        :param labels: 每个样本的标签
        :param test_features: 训练样本的特征，形式上为增广矩阵
        :param test_labels: 测试样本的标签
        :param lr: 学习率
        :param bs: Batch size，仅在使用随机梯度下降时起作用
        :param seed: 随机种子
        :param thresh: 样本分类的阈值
        :param show_period: 每show_period步更新一次可视化
        :param show: 是否进行可视化
        '''
        self.features = features
        self.labels = labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.lr = lr
        self.thresh = thresh
        self.show = show
        self.show_period = show_period
        self.batch_size = bs
        self.sigmoid = lambda z: 1 / (1 + np.exp(-z))
        np.random.seed(seed)

    def normalization(self):
        self.mean = self.features[:,1:].mean(axis=0)
        self.std = self.features[:,1:].std(axis=0)
        self.features[:,1:] = (self.features[:,1:]-self.mean) / (self.std+1e-16)
        self.test_features[:,1:] = (self.test_features[:,1:]-self.mean) / (self.std+1e-16)

    def minmax_norm(self):
        max = self.features[:,1:].max(axis=0)
        min = self.features[:,1:].min(axis=0)
        self.features[:,1:] = (self.features[:,1:]-min) / (max-min+1e-8)
        self.test_features[:,1:] = (self.test_features[:,1:]-min) / (max-min+1e-8)

    def train(self, optimizer='GD'):
        N, feature_dim = self.features.shape
        self.theta = np.mat(np.random.rand(feature_dim, 1))
        if self.show:
            step_ls, loss_ls, train_acc_ls, test_acc_ls = [], [], [], []
            plt.ion()
            plt.figure(figsize=(12,4))
        model = lambda input: self.sigmoid(input * self.theta)
        if optimizer == 'GD':
            feature_dim = self.features.shape[1]

            step = 0
            while 1:
                gradient = (np.diag(np.array(self.labels - model(self.features)).squeeze()) * self.features).sum(axis=0).reshape(feature_dim, 1)
                self.theta += self.lr * gradient
                step += 1

                loss = abs(np.array(self.labels - model(self.features)).sum())
                pred_t = model(self.features) > self.thresh
                acc_t = (pred_t==self.labels).sum() / self.labels.shape[0]
                acc_e = self.eval(self.test_features, self.test_labels, thresh=self.thresh)
                if step % 500 == 0:
                    print("step:{}, loss:{}, train acc:{}, test acc:{}".format(step, loss, acc_t, acc_e))
                if abs(acc_e) > 0.95:
                    plt.close()
                    break

                # Visualization
                if self.show:
                    if step % self.show_period == 0:
                        step_ls.append(step)
                        loss_ls.append(loss)
                        train_acc_ls.append(acc_t)
                        test_acc_ls.append(acc_e)
                        plt.clf()
                        plt.suptitle("Logistic", fontsize=10)

                        loss_curve = plt.subplot(131)
                        loss_curve.set_title("Loss-Step")
                        loss_curve.set_xlabel("step")
                        loss_curve.set_ylabel("loss")
                        loss_curve.set_xlim((0,60000))
                        loss_curve.set_ylim((0, max(loss_ls)+1))
                        loss_curve.plot(step_ls, loss_ls, 'r', lw=1)

                        acc_curve = plt.subplot(132)
                        acc_curve.set_title("Acc-Step")
                        acc_curve.set_xlabel("step")
                        acc_curve.set_ylabel("accuracy")
                        acc_curve.set_xlim((0,60000))
                        acc_curve.set_ylim((0, 1))
                        acc_curve.plot(step_ls, train_acc_ls, 'b', lw=1)
                        acc_curve.plot(step_ls, test_acc_ls, 'g', lw=1)
                        acc_curve.legend(["train", "test"])

                        sample_plot = plt.subplot(133)
                        sample_plot.set_xlabel("x")
                        sample_plot.set_ylabel("y")
                        sample_plot.set_xlim((-5, 5))
                        sample_plot.set_ylim((-5, 5))
                        cls_index = np.array(self.labels).squeeze()
                        sample_plot.plot(self.features[cls_index==0,1], self.features[cls_index==0,2], 'o', color='b', markersize=2)
                        sample_plot.plot(self.features[cls_index==1, 1], self.features[cls_index==1, 2], 'o',color='r', markersize=2)
                        sample_plot.legend(["class: 0", "class: 1"])
                        # revert_theta = self.theta.copy()
                        # revert_theta[1:,:] = revert_theta[1:,:]/self.std.T
                        # revert_theta[0,:] = revert_theta[0,:] + (self.mean/self.std)*self.theta[1:,:]
                        x_on_line = range(-5,6)
                        y_on_line = []
                        for x in x_on_line:
                            y = -self.theta[1,0]/(self.theta[2,0]+1e-16)*x - self.theta[0,0]/(self.theta[2,0]+1e-16)
                            y_on_line.append(y)
                        sample_plot.plot(x_on_line, y_on_line, lw=1, color='g')

                        plt.pause(0.1)



        elif optimizer == 'Newton':
            self.theta = np.mat(np.random.rand(feature_dim, 1))
            step = 0
            while 1:
                # 将逐样本加和转化成包含所有样本的矩阵运算
                gradient = (np.diag(np.array(model(self.features)-self.labels).squeeze()) * self.features).sum(axis=0).reshape(feature_dim, 1) / N
                H = self.features.T * (np.diag(np.array(np.multiply(model(self.features), 1-model(self.features))).squeeze())) * self.features / N
                H = H + np.diag(np.ones(feature_dim)*1e-10)
                self.theta -= H.I * gradient

                step += 1
                loss = abs(np.array(self.labels - model(self.features)).sum())
                pred_t = model(self.features) > self.thresh
                acc_t = (pred_t==self.labels).sum() / self.labels.shape[0]
                acc_e = self.eval(self.test_features, self.test_labels, thresh=self.thresh)
                print("step:{}, loss:{}, train acc:{}, test acc:{}".format(step, loss, acc_t, acc_e))
                if abs(acc_e) > 0.95:
                    break

                    # Visualization
                if self.show:
                    step_ls.append(step)
                    loss_ls.append(loss)
                    train_acc_ls.append(acc_t)
                    test_acc_ls.append(acc_e)
                    plt.clf()
                    plt.suptitle("Logistic", fontsize=10)

                    loss_curve = plt.subplot(131)
                    loss_curve.set_title("Loss-Step")
                    loss_curve.set_xlabel("step")
                    loss_curve.set_ylabel("loss")
                    loss_curve.set_xlim((0, 10))
                    loss_curve.set_ylim((0, max(loss_ls)+1))
                    loss_curve.plot(step_ls, loss_ls, 'r', lw=1)

                    acc_curve = plt.subplot(132)
                    acc_curve.set_title("Acc-Step")
                    acc_curve.set_xlabel("step")
                    acc_curve.set_ylabel("accuracy")
                    acc_curve.set_xlim((0, 10))
                    acc_curve.set_ylim((0, 1))
                    acc_curve.plot(step_ls, train_acc_ls, 'b', lw=1)
                    acc_curve.plot(step_ls, test_acc_ls, 'g', lw=1)
                    acc_curve.legend(["train", "test"])

                    sample_plot = plt.subplot(133)
                    sample_plot.set_xlabel("x")
                    sample_plot.set_ylabel("y")
                    sample_plot.set_xlim((-5, 5))
                    sample_plot.set_ylim((-5, 5))
                    cls_index = np.array(self.labels).squeeze()
                    sample_plot.plot(self.features[cls_index == 0, 1], self.features[cls_index == 0, 2], 'o',
                                     color='b', markersize=2)
                    sample_plot.plot(self.features[cls_index == 1, 1], self.features[cls_index == 1, 2], 'o',
                                     color='r', markersize=2)
                    sample_plot.legend(["class: 0", "class: 1"])
                    # revert_theta = self.theta.copy()
                    # revert_theta[1:,:] = revert_theta[1:,:]/self.std.T
                    # revert_theta[0,:] = revert_theta[0,:] + (self.mean/self.std)*self.theta[1:,:]
                    x_on_line = range(-5, 6)
                    y_on_line = []
                    for x in x_on_line:
                        y = -self.theta[1, 0] / (self.theta[2, 0] + 1e-16) * x - self.theta[0, 0] / (
                                    self.theta[2, 0] + 1e-16)
                        y_on_line.append(y)
                    sample_plot.plot(x_on_line, y_on_line, lw=1, color='g')

                    plt.pause(0.1)

    def eval(self, features_eval, labels_eval, thresh=0.5):
        pred = self.sigmoid(features_eval * self.theta) > thresh
        acc = (pred==labels_eval).sum() / self.test_labels.shape[0]
        return acc


class SoftmaxRegression():
    def __init__(self, features, labels, test_features, test_labels, lr=1e-3, batch_size=10, c=2, seed=0, thresh=0.5, show_period=500, show=True):
        '''

        :param features: 线性模型中的X，形式上为增广矩阵
        :param labels: 每个样本的标签
        :param test_features: 训练样本的特征，形式上为增广矩阵
        :param test_labels: 测试样本的标签
        :param lr: 学习率
        :param bs: Batch size，仅在使用随机梯度下降时起作用
        :param c: 样本有c个类别
        :param seed: 随机种子
        :param thresh: 样本分类的阈值
        :param show_period: 每show_period步更新一次可视化
        :param show: 是否进行可视化
        '''
        self.features = features
        self.labels = labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.lr = lr
        self.thresh = thresh
        self.show = show
        self.show_period = show_period
        self.bs = batch_size
        self.c = c
        self.softmax = lambda z: np.exp(z) / (np.exp(z).sum())
        np.random.seed(seed)

    def normalization(self):
        mean = self.features[:,1:].mean(axis=0)
        std = self.features[:,1:].std(axis=0)
        self.features[:,1:] = (self.features[:,1:]-mean) / (std+1e-10)
        self.test_features[:,1:] = (self.test_features[:,1:]-mean) / (std+1e-10)

    def minmax_norm(self):
        max = self.features[:,1:].max(axis=0)
        min = self.features[:,1:].min(axis=0)
        self.features[:,1:] = (self.features[:,1:]-min) / (max-min+1e-8)
        self.test_features[:,1:] = (self.test_features[:,1:]-min) / (max-min+1e-8)

    def train(self, optimizer='SGD'):
        if optimizer=='SGD':
            N, feature_dim = self.features.shape
            self.theta = np.mat(np.random.rand(feature_dim, self.c))
            step = 0
            if self.show:
                step_ls, loss_ls, train_acc_ls, test_acc_ls = [], [], [], []
                plt.ion()
                plt.figure(figsize=(12, 4))

            while 1:
                if self.lr % 500 == 499:
                    self.lr = self.lr * 0.1
                sample_index = np.random.choice(np.arange(N), size=self.bs)
                sample_onehot_index = np.zeros(N)
                for i in sample_index:
                    sample_onehot_index[i] = 1
                mini_batch = {"features": np.compress(sample_onehot_index, self.features, axis=0),
                              "labels": np.compress(sample_onehot_index, self.labels, axis=0)}
                batch_gradient = np.zeros((feature_dim, self.c))
                loss = np.zeros((feature_dim, self.c))
                output = mini_batch["features"] * self.theta
                for i, label in enumerate(mini_batch["labels"]):
                    cls = int(label[0,0])
                    for c in range(self.c):
                        batch_gradient[:, c] += ((c==cls)-self.softmax(output[i])[0,c]) * np.array(mini_batch["features"][i]).squeeze()
                        loss[:, c] += abs((c == cls) - self.softmax(output[i])[0, c]) / self.bs
                self.theta += self.lr * batch_gradient
                step += 1

                pred_t = (self.features * self.theta).argmax(axis=1)
                acc_t = (pred_t==self.labels).sum() / self.labels.shape[0]
                acc_e = self.eval(self.test_features, self.test_labels, thresh=self.thresh)
                # if step%100 == 0:
                #     print("step:{}, loss:{}, train acc:{}, test acc:{}".format(step, loss.sum(), acc_t, acc_e))

                    # Visualization
                if self.show:
                    if step % self.show_period == 0:
                        step_ls.append(step)
                        loss_ls.append(loss.sum())
                        train_acc_ls.append(acc_t)
                        test_acc_ls.append(acc_e)
                        plt.clf()
                        plt.suptitle("Softmax", fontsize=10)

                        loss_curve = plt.subplot(131)
                        loss_curve.set_title("Loss-Step")
                        loss_curve.set_xlabel("step")
                        loss_curve.set_ylabel("loss")
                        loss_curve.set_xlim((0, max(600, step)))
                        loss_curve.set_ylim((0, max(loss_ls)+1))
                        loss_curve.plot(step_ls, loss_ls, 'r', lw=1)

                        acc_curve = plt.subplot(132)
                        acc_curve.set_title("Acc-Step")
                        acc_curve.set_xlabel("step")
                        acc_curve.set_ylabel("accuracy")
                        acc_curve.set_xlim((0, max(600, step)))
                        acc_curve.set_ylim((0, 1))
                        acc_curve.plot(step_ls, train_acc_ls, 'b', lw=1)
                        acc_curve.plot(step_ls, test_acc_ls, 'g', lw=1)
                        acc_curve.legend(["train", "test"])

                        sample_plot = plt.subplot(133)
                        sample_plot.set_xlabel("x")
                        sample_plot.set_ylabel("y")
                        sample_plot.set_xlim((-6, 6))
                        sample_plot.set_ylim((-6, 6))
                        cls_index = np.array(self.labels).squeeze()
                        sample_plot.plot(self.features[cls_index == 0, 1], self.features[cls_index == 0, 2], 'o',
                                         color='b', markersize=2)
                        sample_plot.plot(self.features[cls_index == 1, 1], self.features[cls_index == 1, 2], 'o',
                                         color='r', markersize=2)
                        sample_plot.plot(self.features[cls_index == 2, 1], self.features[cls_index == 2, 2], 'o',
                                         color='g', markersize=2)
                        sample_plot.legend(["class: 0", "class: 1"])
                        # revert_theta = self.theta.copy()
                        # revert_theta[1:,:] = revert_theta[1:,:]/self.std.T
                        # revert_theta[0,:] = revert_theta[0,:] + (self.mean/self.std)*self.theta[1:,:]
                        l2_norm = lambda x: x/sqrt(float(x[0,1:]*x[0,1:].T))
                        thetas = self.theta.copy().T
                        theta_ls = [l2_norm(theta) for theta in thetas]  # theta是一个行向量
                        if self.c == 2:
                            edge = np.zeros(3)
                            tmp = theta_ls[0] + theta_ls[1]
                            edge[1] = -tmp[0,2]
                            edge[2] = tmp[0,1]
                            inter_point = np.array(-self.theta[0,:]*self.theta[1:,:].I).squeeze()
                            edge[0] = -inter_point[0]*edge[1] -inter_point[1]*edge[2]
                            x_on_line = range(-5, 6)
                            y_on_line = []
                            for x in x_on_line:
                                y = -edge[1] / (edge[2] + 1e-16) * x - edge[0] / (
                                            edge[2] + 1e-16)
                                y_on_line.append(y)
                            sample_plot.plot(x_on_line, y_on_line, lw=1, color='black')

                            y_on_line = []
                            for x in x_on_line:
                                y = -theta_ls[0][0, 1] / (theta_ls[0][0, 2] + 1e-16) * x - theta_ls[0][0, 0] / (
                                        theta_ls[0][0, 2] + 1e-16)
                                y_on_line.append(y)
                            sample_plot.plot(x_on_line, y_on_line, lw=1, color='b')

                            y_on_line = []
                            for x in x_on_line:
                                y = -theta_ls[1][0, 1] / (theta_ls[1][0, 2] + 1e-16) * x - theta_ls[1][0, 0] / (
                                        theta_ls[1][0, 2] + 1e-16)
                                y_on_line.append(y)
                            sample_plot.plot(x_on_line, y_on_line, lw=1, color='r')

                        elif self.c == 3:
                            for line1, line2 in [(0,1),(0,2),(1,2)]:
                                edge = np.zeros(3)
                                tmp = theta_ls[line1] + theta_ls[line2]
                                edge[1] = -tmp[0, 2]
                                edge[2] = tmp[0, 1]
                                inter_point = np.array(-self.theta[0,[line1,line2]] * (self.theta[1:,[line1,line2]].I)).squeeze()
                                edge[0] = -inter_point[0] * edge[1] - inter_point[1] * edge[2]
                                x_on_line = range(-5, 6)
                                y_on_line = []
                                for x in x_on_line:
                                    y = -edge[1] / (edge[2] + 1e-16) * x - edge[0] / (
                                            edge[2] + 1e-16)
                                    y_on_line.append(y)
                                sample_plot.plot(x_on_line, y_on_line, lw=1, color='black')

                            y_on_line = []
                            for x in x_on_line:
                                y = -thetas[0][0, 1] / (thetas[0][0, 2] + 1e-16) * x - thetas[0][0, 0] / (
                                        thetas[0][0, 2] + 1e-16)
                                y_on_line.append(y)
                            sample_plot.plot(x_on_line, y_on_line, lw=1, color='yellow')

                            y_on_line = []
                            for x in x_on_line:
                                y = -thetas[0][0, 1] / (thetas[0][0, 2] + 1e-16) * x - thetas[0][0, 0] / (
                                        thetas[0][0, 2] + 1e-16)
                                y_on_line.append(y)
                            sample_plot.plot(x_on_line, y_on_line, lw=1, color='b')

                            y_on_line = []
                            for x in x_on_line:
                                y = -thetas[1][0, 1] / (thetas[1][0, 2] + 1e-16) * x - thetas[1][0, 0] / (
                                        thetas[1][0, 2] + 1e-16)
                                y_on_line.append(y)
                            sample_plot.plot(x_on_line, y_on_line, lw=1, color='r')

                            y_on_line = []
                            for x in x_on_line:
                                y = -thetas[2][0, 1] / (thetas[2][0, 2] + 1e-16) * x - thetas[2][0, 0] / (
                                        thetas[2][0, 2] + 1e-16)
                                y_on_line.append(y)
                            sample_plot.plot(x_on_line, y_on_line, lw=1, color='g')

                        plt.pause(0.1)

    def eval(self, features_eval, labels_eval, thresh=0.5):
        pred = (features_eval * self.theta).argmax(axis=1)
        acc = (pred == labels_eval).sum() / self.test_labels.shape[0]
        return acc


class MultiClsPerceptron():
    def __init__(self, features, labels, test_features, test_labels, lr=1e-3, batch_size=10, c=2, seed=0, thresh=0.5, show_period=500, show=True):
        '''

        :param features: 线性模型中的X，形式上为增广矩阵
        :param labels: 每个样本的标签
        :param test_features: 训练样本的特征，形式上为增广矩阵
        :param test_labels: 测试样本的标签
        :param lr: 学习率
        :param bs: Batch size，仅在使用随机梯度下降时起作用
        :param c: 样本有c个类别
        :param seed: 随机种子
        :param thresh: 样本分类的阈值
        :param show_period: 每show_period步更新一次可视化
        :param show: 是否进行可视化
        '''
        self.features = features
        self.labels = labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.lr = lr
        self.thresh = thresh
        self.show = show
        self.show_period = show_period
        self.bs = batch_size
        self.c = c
        self.sgn = lambda z: z>=0
        np.random.seed(seed)

    def normalization(self):
        mean = self.features[:,1:].mean(axis=0)
        std = self.features[:,1:].std(axis=0)
        self.features[:,1:] = (self.features[:,1:]-mean) / (std+1e-10)
        self.test_features[:,1:] = (self.test_features[:,1:]-mean) / (std+1e-10)

    def minmax_norm(self):
        max = self.features[:,1:].max(axis=0)
        min = self.features[:,1:].min(axis=0)
        self.features[:,1:] = (self.features[:,1:]-min) / (max-min+1e-8)
        self.test_features[:,1:] = (self.test_features[:,1:]-min) / (max-min+1e-8)

    def train(self, optimizer='SGD'):
        if optimizer=='SGD':
            N, feature_dim = self.features.shape
            if self.c == 2:
                self.theta = np.mat(np.random.rand(feature_dim, 1))
            elif self.c >= 2:
                self.theta = np.mat(np.random.rand(feature_dim, self.c))
            step = 0
            if self.show:
                step_ls, loss_ls, train_acc_ls, test_acc_ls = [], [], [], []
                plt.ion()
                plt.figure(figsize=(12, 4))
            while 1:
                sample_index = np.random.choice(np.arange(N), size=self.bs)
                sample_onehot_index = np.zeros(N)
                for i in sample_index:
                    sample_onehot_index[i] = 1
                mini_batch = {"features": np.compress(sample_onehot_index, self.features, axis=0),
                              "labels": np.compress(sample_onehot_index, self.labels, axis=0)}

                # 2分类和多分类分开讨论
                # 扩展到多分类时，采用与Kesler‘s construction相同的判别条件(模式识别第四版 p.70)。这种判别条件相比one vs one和one vs rest，优点在于不存在模糊区域。
                if self.c > 2:
                    if step%1000 == 999:
                        self.lr = self.lr * 0.01
                        print(f" lr decay {self.lr} "*10 )
                    batch_gradient = np.zeros((feature_dim, self.c))
                    loss = np.zeros((feature_dim, self.c))
                    output = mini_batch["features"] * self.theta
                    for i, label in enumerate(mini_batch["labels"]):
                        cls = int(label[0,0])
                        for c in range(self.c):
                            batch_gradient[:, c] += (int(self.sgn(output[i,:])[0,c])-int((cls==c))) * np.array(mini_batch["features"][i]).squeeze().T
                            loss[:, c] += abs(int(self.sgn(output[i, :])[0, c]) - int((cls == c))) / self.bs
                    self.theta -= self.lr * batch_gradient / self.bs
                    step += 1

                    output = (self.features * self.theta).argmax(axis=1)
                    acc_t = (output == self.labels).sum() / self.labels.shape[0]
                    acc_e = self.eval(self.test_features, self.test_labels, thresh=self.thresh)
                    if step%500 == 0:
                        print("step:{}, loss:{}, train acc:{}, test acc:{}".format(step, loss.sum(), acc_t, acc_e))

                elif self.c == 2:
                    if step % 500 == 499:
                        self.lr = self.lr * 0.01
                        print(f" lr decay {self.lr} " * 10)
                    batch_gradient = np.zeros((feature_dim, 1))
                    loss = np.zeros((feature_dim, 1))
                    output = mini_batch["features"] * self.theta
                    for i, label in enumerate(mini_batch["labels"]):
                        cls = label[0,0]
                        batch_gradient += (int(self.sgn(output[i])[0,0])-cls) * np.array(mini_batch["features"][i]).T
                        loss += abs(int(self.sgn(output[i])[0,0])-cls) / self.bs
                    self.theta -= (self.lr * batch_gradient / self.bs)
                    step += 1

                    pred_t = self.sgn(self.features * self.theta)
                    acc_t = (pred_t==self.labels).sum() / self.labels.shape[0]
                    acc_e = self.eval(self.test_features, self.test_labels, thresh=self.thresh)
                    if step%500 == 0:
                        print("step:{}, loss:{}, train acc:{}, test acc:{}".format(step, loss.sum(), acc_t, acc_e))

                if self.show:
                    if step % self.show_period == 0:
                        step_ls.append(step)
                        loss_ls.append(loss.sum())
                        train_acc_ls.append(acc_t)
                        test_acc_ls.append(acc_e)
                        plt.clf()
                        plt.suptitle("Perceptron", fontsize=10)

                        loss_curve = plt.subplot(131)
                        loss_curve.set_title("Loss-Step")
                        loss_curve.set_xlabel("step")
                        loss_curve.set_ylabel("loss")
                        loss_curve.set_xlim((0, max(600,step)))
                        loss_curve.set_ylim((0, max(loss_ls)+1))
                        loss_curve.plot(step_ls, loss_ls, 'r', lw=1)

                        acc_curve = plt.subplot(132)
                        acc_curve.set_title("Acc-Step")
                        acc_curve.set_xlabel("step")
                        acc_curve.set_ylabel("accuracy")
                        acc_curve.set_xlim((0, max(600,step)))
                        acc_curve.set_ylim((0, 1))
                        acc_curve.plot(step_ls, train_acc_ls, 'b', lw=1)
                        acc_curve.plot(step_ls, test_acc_ls, 'g', lw=1)
                        acc_curve.legend(["train", "test"])

                        sample_plot = plt.subplot(133)
                        sample_plot.set_xlabel("x")
                        sample_plot.set_ylabel("y")
                        sample_plot.set_xlim((-5, 5))
                        sample_plot.set_ylim((-5, 5))
                        cls_index = np.array(self.labels).squeeze()
                        sample_plot.plot(self.features[cls_index == 0, 1], self.features[cls_index == 0, 2], 'o',
                                         color='b', markersize=2)
                        sample_plot.plot(self.features[cls_index == 1, 1], self.features[cls_index == 1, 2], 'o',
                                         color='r', markersize=2)
                        sample_plot.plot(self.features[cls_index == 2, 1], self.features[cls_index == 2, 2], 'o',
                                         color='g', markersize=2)
                        sample_plot.legend(["class: 0", "class: 1"])
                        # revert_theta = self.theta.copy()
                        # revert_theta[1:,:] = revert_theta[1:,:]/self.std.T
                        # revert_theta[0,:] = revert_theta[0,:] + (self.mean/self.std)*self.theta[1:,:]
                        l2_norm = lambda x: x/sqrt(float(x[0,1:]*x[0,1:].T))
                        thetas = self.theta.copy().T
                        theta_ls = [l2_norm(theta) for theta in thetas]  # theta是一个行向量

                        if self.c == 2:
                            x_on_line = range(-5, 6)
                            y_on_line = []
                            for x in x_on_line:
                                y = -self.theta[1, 0] / (self.theta[2, 0] + 1e-16) * x - self.theta[0, 0] / (
                                            self.theta[2, 0] + 1e-16)
                                y_on_line.append(y)
                            sample_plot.plot(x_on_line, y_on_line, lw=1, color='g')

                        elif self.c == 3:
                            for line1, line2 in [(0,1),(0,2),(1,2)]:
                                edge = np.zeros(3)
                                tmp = theta_ls[line1] + theta_ls[line2]
                                edge[1] = -tmp[0, 2]
                                edge[2] = tmp[0, 1]
                                inter_point = np.array(-self.theta[0,[line1,line2]] * self.theta[1:,[line1,line2]].I).squeeze()
                                edge[0] = -inter_point[0] * edge[1] - inter_point[1] * edge[2]
                                x_on_line = range(-5, 6)
                                y_on_line = []
                                for x in x_on_line:
                                    y = -edge[1] / (edge[2] + 1e-16) * x - edge[0] / (
                                            edge[2] + 1e-16)
                                    y_on_line.append(y)
                                sample_plot.plot(x_on_line, y_on_line, lw=1, color='black')

                            y_on_line = []
                            for x in x_on_line:
                                y = -theta_ls[0][0, 1] / (theta_ls[0][0, 2] + 1e-16) * x - theta_ls[0][0, 0] / (
                                        theta_ls[0][0, 2] + 1e-16)
                                y_on_line.append(y)
                            sample_plot.plot(x_on_line, y_on_line, lw=1, color='b')

                            y_on_line = []
                            for x in x_on_line:
                                y = -theta_ls[1][0, 1] / (theta_ls[1][0, 2] + 1e-16) * x - theta_ls[1][0, 0] / (
                                        theta_ls[1][0, 2] + 1e-16)
                                y_on_line.append(y)
                            sample_plot.plot(x_on_line, y_on_line, lw=1, color='r')

                            y_on_line = []
                            for x in x_on_line:
                                y = -theta_ls[2][0, 1] / (theta_ls[2][0, 2] + 1e-16) * x - theta_ls[2][0, 0] / (
                                        theta_ls[2][0, 2] + 1e-16)
                                y_on_line.append(y)
                            sample_plot.plot(x_on_line, y_on_line, lw=1, color='g')

                        plt.pause(0.1)

    def eval(self, features_eval, labels_eval, thresh=0.5):
        if self.c > 2:
            output = (features_eval * self.theta).argmax(axis=1)
            acc = (output==labels_eval).sum() / self.test_labels.shape[0]
        elif self.c == 2:
            pred = self.sgn(features_eval * self.theta)
            acc = (pred == labels_eval).sum() / self.test_labels.shape[0]
        return acc

