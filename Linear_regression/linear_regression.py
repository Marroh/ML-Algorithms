import os
import numpy as np
import matplotlib.pyplot as plt


def close_form(x, y):
    X = np.mat(np.append(np.ones(x.shape).reshape(x.shape[0], 1), x.reshape(x.shape[0], 1), axis=1))
    y = np.mat(y.reshape(y.shape[0], 1))
    theta = (X.T * X).I * X.T * y
    return theta


def gradient_descent(x, y, lr):
    theta = np.mat([[-1600.0],[2.0]])
    X = np.mat(np.append(np.ones(x.shape).reshape(x.shape[0], 1), x.reshape(x.shape[0], 1), axis=1))
    y = np.mat(y.reshape(y.shape[0], 1))
    step = 0
    last_loss = 0

    while 1:
        gradient = (np.diag(np.array(X*theta - y).squeeze()) * X).sum(axis=0).reshape(X.shape[1], 1)
        theta -= lr * gradient
        loss = np.multiply(X*theta - y, X*theta - y).sum() / X.shape[0]
        step += 1
        if step%10 == 0:
            print("step:{}, loss:{}".format(step,loss))
        if abs(last_loss-loss) < 1e-6:
            break
        last_loss = loss
    return theta


# Load data
data_dir = './Price'
x_file = os.path.join(data_dir, 'x.txt')
y_file = os.path.join(data_dir, 'y.txt')

with open(x_file) as f:
    x = f.read().splitlines()
    x = np.array(list(map(float, x)))

with open(y_file) as f:
    y = f.read().splitlines()
    y = np.array(list(map(float, y)))

assert len(x) == len(y), "请确认样本和label数量相同"

theta1 = close_form(x, y)
theta2 = gradient_descent(x, y, 1e-9)
X = np.mat(np.append(np.ones(x.shape).reshape(x.shape[0], 1), x.reshape(x.shape[0], 1), axis=1))
y1 = np.array(X*theta1)
y2 = np.array(X*theta2)

plt.figure(figsize=(8,4))
plt.subplot(121)
plt.title("Close form solution")
plt.plot(x, y1)
plt.plot(x, y, 'o')
plt.subplot(122)
plt.title("Gradient descent solution")
plt.plot(x, y2)
plt.plot(x, y, 'o')
plt.show()

print("Close form solution: y={:.3f}+{:.3f}x, 预测2014年南京房价为{:.3f}k元".format(theta1[0,0], theta1[1,0], theta1[0,0]+theta1[1,0]*2014))
print("Gradient descent solution: y={:.3f}+{:.3f}x, 预测2014年南京房价为{:.3f}k元".format(theta2[0,0], theta2[1,0], theta2[0,0]+theta2[1,0]*2014))