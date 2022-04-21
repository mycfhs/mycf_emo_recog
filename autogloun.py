# !pip install autogluon
# !pip install mxnet-cu110
# 在colab上跑的，win失败了报错
import autogluon.core as ag
from autogluon.vision import ImagePredictor
import os
import pandas as pd


def labelName2number(label):
    labelNameList = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sad']
    return labelNameList.index(label)


def get_dataset(mypath='data/train/'):
    train = []
    baseRoot = os.getcwd().replace('\\','/') + '/'
    for _, _, files in os.walk(mypath, topdown=False):
        for file in files:
            label = labelName2number(file.split('_')[0])
            train.append([baseRoot+mypath+file, label])
    data = pd.DataFrame(columns=['image', 'label'], data=train)#数据有三列，列名分别为one,two,three
    return data

trainData = get_dataset('/content/drive/MyDrive/autogluon/data/train/')
testData = get_dataset('/content/drive/MyDrive/autogluon/data/test/')

model = ag.Categorical('vit_small_patch16_224', 'vit_tiny_r_s16_p8_224_in21k','vit_large_r50_s32_384','vit_giant_patch14_224','vit_base_patch16_224')
# model = ag.Categorical('resnet18_v1', 'resnet152_v2','se_resnet101_v1','se_resnet50_v2','alexnet')
# you may choose more than 70+ available model in the model zoo provided by GluonCV:
model_list = ImagePredictor.list_models()
batch_size = 32
lr = ag.Real(1e-4, 1e-1, log=True)
# hyperparameters={'model': model, 'batch_size': batch_size, 'lr': lr, 'epochs': 15}
hyperparameters={'model': model, 'lr': lr}
predictor = ImagePredictor()
predictor.fit(trainData, testData, time_limit=None, hyperparameters=hyperparameters,presets='best_quality')
predictor.leaderboard(testData, silent=True)