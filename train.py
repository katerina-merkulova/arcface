import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils import data

from data import Dataset
from models import FocalLoss, ArcMarginProduct, resnet_face18

from test import lfw_test
from utils import Visualizer


def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name


if __name__ == '__main__':

    visualizer = Visualizer()
    device = torch.device('cuda:0')

    train_root = Path.cwd() / 'Market-1501-v15.09.15' / 'bounding_box_train'

    train_dataset = Dataset(train_root, phase='train', input_shape=(3, 64, 128))
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=64,
                                  shuffle=False,
                                  num_workers=4)

    print(f'{len(trainloader)} train iters per epoch:')

    criterion = FocalLoss(gamma=2)
    # criterion = torch.nn.CrossEntropyLoss()

    model = resnet_face18(use_se=False)

    # metric_fc = AddMarginProduct(512, opt.num_classes, s=30, m=0.35)
    metric_fc = ArcMarginProduct(512, 1501, s=30, m=0.5, easy_margin=False)
    # metric_fc = SphereProduct(512, opt.num_classes, m=4)
    # metric_fc = nn.Linear(512, opt.num_classes)

    print(model)
    model.to(device)
    # model = DataParallel(model)
    metric_fc.to(device)
    # metric_fc = DataParallel(metric_fc)

    # optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
    #                             lr=1e-1, weight_decay=5e-4)
    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                 lr=1e-1, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    start = time.time()
    max_epoch = 50
    display = True
    for i in range(max_epoch):
        scheduler.step()

        model.train()
        for ii, data in enumerate(trainloader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = i * len(trainloader) + ii

            if iters % 10 == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                # print(output)
                # print(label)
                acc = np.mean((output == label).astype(int))
                speed = 100 / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print(f'{time_str} train epoch {i} iter {ii} {speed} iters/s loss {loss.item()} acc {acc}')
                if display:
                    visualizer.display_current_results(iters, loss.item(), name='train_loss')
                    visualizer.display_current_results(iters, acc, name='train_acc')

                start = time.time()

        if i % 10 == 0 or i == 50:
            save_model(model, 'checkpoints', 'resnet18', i)

        model.eval()
        acc = lfw_test(model, batch_size=64)
        if display:
            visualizer.display_current_results(iters, acc, name='test_acc')
