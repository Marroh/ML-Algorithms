import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from vit_pytorch import ViT
from utils import *

f = open('./config.yaml', 'r', encoding='utf-8')
cfg = yaml.load(f, Loader=yaml.FullLoader)

train_data = VOC07TrainLoader('./VOC2007')
test_data = VOC07TestLoader('./VOC2007')
train_loader = DataLoader(dataset=train_data,
                          batch_size=cfg['data']['bs'],
                          shuffle=cfg['data']['shuffle'],
                          num_workers=cfg['data']['workers'],
                          pin_memory=cfg['data']['pin'])
test_loader = DataLoader(dataset=test_data,
                         batch_size=cfg['data']['bs'],
                         shuffle=cfg['data']['shuffle'],
                         num_workers=cfg['data']['workers'],
                         pin_memory=cfg['data']['pin'])

model = ViT(**cfg['net'])
model.mlp_head.add_module('mlp_head_sigmoid', nn.Sigmoid())
model = model.to(cfg['train']['device'])

criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=cfg['train']['lr'])
optimizer = optim.SGD(model.parameters(), lr=cfg['train']['lr'], momentum=cfg['train']['momentum'], nesterov=True)
scheduler = MultiStepLR(optimizer, milestones=[200, 1800], gamma=cfg['train']['gamma'])
# scheduler = StepLR(optimizer, step_size=20, gamma=cfg['train']['gamma'])

if __name__ == '__main__':
    for epoch in range(cfg['train']['epochs']):
        epoch_loss = 0
        epoch_accuracy = 0

        iterator = tqdm(train_loader)
        for data, label in iterator:
            data = data.to(cfg['train']['device'])
            label = label.to(cfg['train']['device'], dtype=torch.float32)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss / len(train_loader)
            epoch_accuracy += calc_recall_precision(output, label, 0.5)[1] / len(train_loader)

            iterator.set_description(f"Epoch {epoch + 1}/{cfg['train']['epochs']}")
            iterator.set_postfix(loss=loss.item())
        scheduler.step()

        print(f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc :{epoch_accuracy:.4f}")

        if epoch % 20 == 19:
            with torch.no_grad():
                epoch_test_accuracy = 0
                epoch_test_loss = 0
                output_collect = []
                label_collect = []
                for data, label in test_loader:
                    data = data.to(cfg['train']['device'])
                    label = label.to(cfg['train']['device'], dtype=torch.float32)

                    test_output = model(data)
                    test_loss = criterion(test_output, label)

                    output_collect.append(test_output)
                    label_collect.append(label)
                    epoch_test_loss += test_loss / len(test_loader)

                output_collect = torch.vstack(output_collect)
                label_collect = torch.vstack(label_collect)
                mAP, macc = calc_mAP(output_collect, label_collect)
                print(f"test_loss : {epoch_test_loss:.4f} - test_mAP: {mAP:.4f} --test_mAcc: {macc:.4f}\n")
                with open('./log.txt', 'wa') as f:
                    f.write(f"Epoch:{epoch + 1} -test_loss:{epoch_test_loss:.4f} -test_mAP:{mAP:.4f} -test_mAcc:{macc:.4f} -loss:{epoch_loss:.4f} -acc:{epoch_accuracy:.4f}\n")

        if epoch % 100 == 99:
            torch.save(model.state_dict(), f'./model_e{epoch}_map{mAP:.2f}.pth')
