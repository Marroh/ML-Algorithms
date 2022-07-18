import matplotlib.pylab as plt
import numpy as np
import torch


def visualiztion(step_ls, loss_ls, train_acc_ls,test_acc_ls, test_inputs, test_labels, model):
    plt.clf()
    plt.suptitle("Hand coding MLP", fontsize=10)
    loss_curve = plt.subplot(131)
    loss_curve.set_title("Loss-Step")
    loss_curve.set_xlabel("step")
    loss_curve.set_ylabel("loss")
    loss_curve.set_xlim((0, max(600, max(step_ls))))
    loss_curve.set_ylim((0, max(loss_ls) + 1))
    loss_curve.plot(step_ls, loss_ls, 'r', lw=1)

    acc_curve = plt.subplot(132)
    acc_curve.set_title("Acc-Step")
    acc_curve.set_xlabel("step")
    acc_curve.set_ylabel("accuracy")
    acc_curve.set_xlim((0, max(600, max(step_ls))))
    acc_curve.set_ylim((0, 1.2))
    acc_curve.plot(step_ls, train_acc_ls, 'b', lw=1)
    acc_curve.plot(step_ls, test_acc_ls, 'g', lw=1)
    acc_curve.legend(["train", "test"])

    sample_plot = plt.subplot(133)
    sample_plot.set_xlabel("x")
    sample_plot.set_ylabel("y")
    sample_plot.set_xlim((-6, 6))
    sample_plot.set_ylim((-6, 6))
    cls_index = np.argmax(test_labels, axis=0)
    xx, yy = np.meshgrid(np.arange(-6, 6, 0.05), np.arange(-6, 6, 0.05))
    # predict the meshgrid
    sample_data = np.c_[xx.ravel(), yy.ravel()].T
    pseudo_label = np.zeros((test_labels.shape[0], sample_data.shape[1]))
    preds, _ = model.forward(sample_data, pseudo_label)
    preds = preds.argmax(axis=0)
    Z = preds.reshape(len(xx), len(yy))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
    sample_plot.scatter(test_inputs[0, :], test_inputs[1, :], c=cls_index, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.pause(0.1)


def visualization_torch(step_ls, loss_ls, train_acc_ls,test_acc_ls, test_inputs, test_labels, model):
    plt.clf()
    plt.suptitle("Pytroch MLP", fontsize=10)
    loss_curve = plt.subplot(131)
    loss_curve.set_title("Loss-Step")
    loss_curve.set_xlabel("step")
    loss_curve.set_ylabel("loss")
    loss_curve.set_xlim((0, max(600, max(step_ls))))
    loss_curve.set_ylim((0, max(loss_ls) + 1))
    loss_curve.plot(step_ls, loss_ls, 'r', lw=1)

    acc_curve = plt.subplot(132)
    acc_curve.set_title("Acc-Step")
    acc_curve.set_xlabel("step")
    acc_curve.set_ylabel("accuracy")
    acc_curve.set_xlim((0, max(600, max(step_ls))))
    acc_curve.set_ylim((0, 1.2))
    acc_curve.plot(step_ls, train_acc_ls, 'b', lw=1)
    acc_curve.plot(step_ls, test_acc_ls, 'g', lw=1)
    acc_curve.legend(["train", "test"])

    sample_plot = plt.subplot(133)
    sample_plot.set_xlabel("x")
    sample_plot.set_ylabel("y")
    sample_plot.set_xlim((-6, 6))
    sample_plot.set_ylim((-6, 6))
    cls_index = torch.argmax(test_labels, dim=1)
    xx, yy = np.meshgrid(np.arange(-6, 6, 0.05), np.arange(-6, 6, 0.05))
    # predict the meshgrid
    sample_data = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    preds = model(sample_data)
    preds = torch.argmax(preds, dim=1)
    Z = preds.reshape(len(xx), len(yy))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
    sample_plot.scatter(test_inputs[:, 0], test_inputs[:, 1], c=cls_index, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.pause(0.1)
