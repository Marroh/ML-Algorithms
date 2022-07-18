import torch
import torch.nn as nn
import torch.nn.functional as f
import matplotlib.pylab as plt
from tqdm import tqdm
from Dataloader import Dataloader
from utils import visualization_torch


class MLP(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, output_dim)

    def forward(self, inputs):
        x = f.relu(self.fc1(inputs))
        x = f.relu(self.fc2(x))
        y = f.softmax(self.fc3(x), dim=1)
        return y


def calc_acc(preds, onehot_labels):
    preds = torch.argmax(preds, dim=1)
    onehot_labels = torch.argmax(onehot_labels, dim=1)
    acc = torch.nonzero(preds == onehot_labels).shape[0] / preds.shape[0]
    # print('\n', preds == onehot_labels)
    # print(preds)
    # print(onehot_labels)
    # print(torch.nonzero(preds & onehot_labels))
    return acc


# Load data
data_loader = Dataloader('./Exam/train/x.txt', './Exam/train/y.txt', './Exam/test/x.txt', './Exam/test/y.txt',
                         classes=2, norm=True, batch_size=8)
# data_loader = Dataloader('./Iris/train/x.txt', './Iris/train/y.txt', './Iris/test/x.txt', './Iris/test/y.txt',
#                          classes=3, norm=True, batch_size=16)
test_inputs, test_labels = data_loader.load_test_data()
test_inputs = torch.Tensor(test_inputs.T)
test_labels = torch.Tensor(test_labels.T)

# Initialize model and optimizer
model = MLP(input_dim=2, output_dim=2)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-2)

# Train
step_ls, loss_ls, train_acc_ls, test_acc_ls = [], [], [], []
plt.ion()
plt.figure(figsize=(12, 4))
interval = 100
iteration = tqdm(range(10000))
for i in iteration:
    inputs, labels = data_loader.load_train_data()
    inputs = torch.Tensor(inputs.T)
    labels = torch.Tensor(labels.T)
    preds = model(inputs)
    loss = f.mse_loss(preds, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % interval == interval - 1:
        test_preds = model(test_inputs)
        test_loss = f.mse_loss(test_preds, test_labels)
        optimizer.zero_grad()

        iteration.set_description(f'Batch {i}')
        iteration.set_postfix(loss=loss.item(), acc=calc_acc(test_preds, test_labels))

        step_ls.append(i)
        loss_ls.append(loss.item())
        train_acc_ls.append(calc_acc(preds, labels))
        test_acc_ls.append(calc_acc(test_preds, test_labels))

        visualization_torch(step_ls, loss_ls, train_acc_ls, test_acc_ls, test_inputs, test_labels, model)