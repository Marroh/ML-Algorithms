import matplotlib.pyplot as plt

with open('log_1.txt', 'r') as f:
    log = list(map(lambda x: x.strip('\n'), f.readlines()))

epoch_ls = list(map(lambda x: int(x[x.find('Epoch')+6 : x.find('-test_loss')]), log))
test_loss_ls = list(map(lambda x: float(x[x.find('test_loss')+10 : x.find('test_loss')+16]), log))
test_map_ls = list(map(lambda x: float(x[x.find('test_mAP')+9 : x.find('test_mAP')+15]), log))
train_loss_ls = list(map(lambda x: float(x[x.find('loss')+5 : x.find('loss')+11]), log))

plt.figure(figsize=(15,5))
plt.subplot(131)
plt.title('Train loss')
plt.plot(epoch_ls, train_loss_ls, '-')
plt.subplot(132)
plt.title('Test loss')
plt.plot(epoch_ls, test_loss_ls, '-')
plt.subplot(133)
plt.title('Test mAP')
plt.plot(epoch_ls, test_map_ls, '-')

plt.show()
