"""
    @Name: find_model.py
    @Author: yicheng Yang
    @Time: 2022/3/29
"""


import torch
from torchvision import datasets, models, transforms
import os
from dataLoader import dataLoader
from torch.utils.data import DataLoader
from train_test_func import *

import sys
import time

BATCH_SIZE = 8
NUM_WORKERS = 0
train_data = dataLoader(txtFileName='testImages_artphoto_trainset.txt')
train_dataset = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_data = dataLoader(txtFileName='testImages_artphoto_testset.txt')
test_dataset = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=NUM_WORKERS)

netList = ['vgg19', 'vgg16_bn','vgg13','squeezenet1_1', 'convnext_tiny','convnext_base', 'convnext_large', 'wide_resnet101_2', 'resnet18', 'inception_v3_google',
           'densenet201', 'mnasnet1_0', 'efficientnet_b5', 'vit_b_16', 'shufflenetv2_x1.0']

for model_name in netList:
    # 保存过程
    sys_stdout = open("%s.log" % model_name, "w+")
    sys.stdout = sys_stdout

    try:
        model = eval('models.%s(pretrained=True)' % model_name)
    except:
        print('%s model 不可用！单独调试！' % model_name)
        continue

    print('\n--------------model %s load!-------------------------\n\n\n' % model_name)
    model.classifier = torch.nn.Sequential(torch.nn.Linear(7, 4096),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(p=0.5),
                                           torch.nn.Linear(4096, 4096),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(p=0.5),
                                           torch.nn.Linear(4096, 8))

    model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters())

    DEVICE = 'cuda'
    EPOCHS = 30
    model_can_be_use = False
    for epoch in range(1, EPOCHS + 1):
        if model_can_be_use:
            try:
                train_model(model, DEVICE, train_dataset, optimizer, criterion, epoch)
                test_model(model, DEVICE, criterion, test_dataset)
                continue
            except:
                print('%s model 不可用！单独调试！' % model_name)
                break
        else:
            train_model(model, DEVICE, train_dataset, optimizer, criterion, epoch)
            test_model(model, DEVICE, criterion, test_dataset)
