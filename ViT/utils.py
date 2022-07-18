import os
import yaml
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

f = open('./config.yaml', 'r', encoding='utf-8')
cfg = yaml.load(f, Loader=yaml.FullLoader)
classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


class VOC07TrainLoader(Dataset):
    def __init__(self, voc_root):
        self.root = voc_root
        # generate label for multi-label classification
        multi_label_file = os.path.join(voc_root, 'multilabel_trainval.csv')
        if not os.path.exists(multi_label_file):
            img_name = None
            labels = None
            for cls in classes:
                gt_file = open(os.path.join(voc_root, 'ImageSets', 'Main', cls + '_trainval.txt'), 'r')
                data = gt_file.read().splitlines()
                if img_name is None:
                    img_name = np.array(list(map(lambda x: x[:7], data)))
                if labels is None:
                    labels = (np.array(list(map(lambda x: x[7:], data)), dtype=int) > 0).astype(int)
                    labels = labels.astype(str)
                else:
                    tmp_label = (np.array(list(map(lambda x: x[7:], data)), dtype=int) > 0).astype(int)
                    tmp_label = tmp_label.astype(str)
                    labels = np.char.add(labels, tmp_label)
            multi_gt = np.stack([img_name, labels]).T
            np.savetxt(multi_label_file, multi_gt, fmt='%s')

        self.gt_arr = np.loadtxt(multi_label_file, dtype=str)
        self.trans = transforms.Compose([transforms.Resize((cfg['data']['resize'], cfg['data']['resize'])),
                                         transforms.RandomResizedCrop(cfg['net']['image_size']),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.4485, 0.4250, 0.3920],
                                                              std=[0.2678, 0.2644, 0.2755])
                                         ])

    def __len__(self):
        return self.gt_arr.shape[0]

    def __getitem__(self, idx):
        img_id, label = self.gt_arr[idx]
        img = Image.open(os.path.join(self.root, 'JPEGImages', img_id + '.jpg'))
        # print(os.path.join(self.root, 'JPEGImages', img_id+'.jpg'), label)
        img_trans = self.trans(img)
        label = np.array(list(label), dtype=float)
        return img_trans, label


class VOC07TestLoader(Dataset):
    def __init__(self, voc_root):
        self.root = voc_root
        # generate label for multi-label classification
        multi_label_file = os.path.join(voc_root, 'multilabel_test.csv')
        if not os.path.exists(multi_label_file):
            img_name = None
            labels = None
            for cls in classes:
                gt_file = open(os.path.join(voc_root, 'ImageSets', 'Main', cls + '_test.txt'), 'r')
                data = gt_file.read().splitlines()
                if img_name is None:
                    img_name = np.array(list(map(lambda x: x[:7], data)))
                if labels is None:
                    labels = (np.array(list(map(lambda x: x[7:], data)), dtype=int) > 0).astype(int)
                    labels = labels.astype(str)
                else:
                    tmp_label = (np.array(list(map(lambda x: x[7:], data)), dtype=int) > 0).astype(int)
                    tmp_label = tmp_label.astype(str)
                    labels = np.char.add(labels, tmp_label)
            multi_gt = np.stack([img_name, labels]).T
            np.savetxt(multi_label_file, multi_gt, fmt='%s')

        self.gt_arr = np.loadtxt(multi_label_file, dtype=str)
        self.trans = transforms.Compose([transforms.Resize((cfg['net']['image_size'], cfg['net']['image_size'])),
                                         # transforms.RandomResizedCrop(256),
                                         # transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.4485, 0.4250, 0.3920],
                                                              std=[0.2678, 0.2644, 0.2755])
                                         ])

    def __len__(self):
        return self.gt_arr.shape[0]

    def __getitem__(self, idx):
        img_id, label = self.gt_arr[idx]
        img = Image.open(os.path.join(self.root, 'JPEGImages', img_id + '.jpg'))
        # print(os.path.join(self.root, 'JPEGImages', img_id+'.jpg'), label)
        img_trans = self.trans(img)
        label = np.array(list(label), dtype=float)
        return img_trans, label


# loader = VOC07TestLoader('./VOC2007')
# all_pixels_1, all_pixels_2, all_pixels_3 = [], [], []
# for img, label in loader:
#     print(np.array(classes, dtype=str)[label==1])
#     plt.imshow(np.array(img).transpose([1,2,0]))
#     plt.show()
#     all_pixels_1.append(img[0].flatten())
#     all_pixels_2.append(img[1].flatten())
#     all_pixels_3.append(img[2].flatten())
# all_pixels_1 = torch.hstack(all_pixels_1)
# all_pixels_2 = torch.hstack(all_pixels_2)
# all_pixels_3 = torch.hstack(all_pixels_3)
# mean_1 = torch.mean(all_pixels_1)
# mean_2 = torch.mean(all_pixels_2)
# mean_3 = torch.mean(all_pixels_3)
# std_1 = torch.std(all_pixels_1)
# std_2 = torch.std(all_pixels_2)
# std_3 = torch.std(all_pixels_3)
#
# print(mean_1, mean_2, mean_3, std_1, std_2, std_3)

def calc_recall_precision(output, label, thresh):
    pred = output > thresh
    label = label.bool()
    tp = (pred & label).sum()
    fp = (pred & (~label)).sum()
    fn = ((~pred) & label).sum()
    precision = 0.0 if tp == fp == 0 else tp / (tp + fp)
    recall = tp / (tp + fn)
    return recall, precision


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or abs(value - array[idx - 1]) < abs(value - array[idx])):
        return array[idx - 1]
    else:
        return array[idx]


def calc_mAP(output: torch.Tensor, label: torch.Tensor):
    """
    Pre = TP / TP+FP
    Recall = TP / TP+FN

    :param output: N * cls_num
    :param label: N * cls_num, multi-label
    :return: mAP.item()
    """
    num, cls = output.shape
    ap_ls = []
    acc_ls = []
    for c in range(cls):
        recall_ls, precision_ls = [], []
        ap = 0.
        output_c = output[:, c]
        label_c = label[:, c]

        # sort output-label by scores
        tmp_concat = torch.vstack([output_c, label_c]).T.detach().cpu()
        tmp_concat = torch.vstack(sorted(tmp_concat, key=lambda x: x[0], reverse=True))
        output_c = tmp_concat[:, 0]
        label_c = tmp_concat[:, 1]

        acc_ls.append(calc_recall_precision(output_c, label_c, 0.5)[1])
        for thresh in output_c:
            recall, precision = calc_recall_precision(output_c, label_c, thresh)
            recall_ls.append(recall)  # recall from 0 to 1
            precision_ls.append(precision)
        for thresh in np.arange(0, 1.1, 0.1):
            idx = np.searchsorted(recall_ls, thresh, 'left')
            ap += precision_ls[idx] / 11 if idx < num else precision_ls[idx - 1] / 11
        ap_ls.append(ap)
    mAP = sum(ap_ls) / len(ap_ls)
    macc = sum(acc_ls) / len(acc_ls)

    return mAP.item(), macc.item()
