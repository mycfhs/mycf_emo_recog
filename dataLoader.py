"""
    @Name: dataLoader.py
    @Author: yicheng Yang
    @Time: 2022/3/28
"""
import cv2
import os
import torch
import numpy as np
from torchvision import transforms


def labelName2number(label):
    labelNameList = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sad']
    return labelNameList.index(label)


def dataLoader(txtFileName='testImages_artphoto_trainset.txt', imgPath='testImages_artphoto', imgResize=(224,224)):
    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    # transforms.CenterCrop(imgResize[0]),
                                    transforms.Resize(imgResize),
                                    transforms.ColorJitter(brightness=0.5),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataList, labelList = [], []

    with open(txtFileName, "r", encoding='utf-8') as txtData:
        data = txtData.readlines()

    for imgName in data:
        # imgName有个换行，不清掉无法读取
        if imgName[-1] == '\n':
            imgName = imgName[:-1]
        # 读取图片
        img = cv2.imread(os.path.join(imgPath, imgName))
        # 图片保存进列表
        dataList.append(transform(img))
        # 标签同样转成列表，最后统一转tensor
        labelList.append(labelName2number(imgName.split('_')[0]))

    train_data = torch.stack(dataList, 0)
    # print(train_data.shape)
    # 返回值用torch.utils.data.DataLoader打开。 先转numpy再tensor是为了提速（大概）

    # return torch.utils.data.TensorDataset(train_data.float(), torch.from_numpy(np.array(labelList)).to(torch.int64))
    return train_data.float(), torch.from_numpy(np.array(labelList)).to(torch.int64)

def dataLoader_v2(txtFileName='testImages_artphoto_trainset.txt', imgPath='testImages_artphoto', imgResize=(224, 224)):
    with open(txtFileName, "r", encoding='utf-8') as txtData:
        data = txtData.readlines()
    dataList, labelList = [], []
    trans = ['CenterCrop(imgResize[0])',
             'RandomCrop(imgResize[0])',
             'RandomResizedCrop(imgResize[0])',
             'RandomHorizontalFlip(p=1.0)',
             "RandomVerticalFlip(p=1.0)",
             'transforms.ColorJitter(brightness=0.5)'
             ]
    for tran in trans:
        transform = transforms.Compose([transforms.ToTensor(),
                                        eval('transforms.%s' % (tran)),
                                        transforms.Resize(imgResize),
                                        transforms.Normalize([0.3, 0.5, 0.8], [0.3, 0.5, 0.8])])

        for imgName in data:
            # imgName有个换行，不清掉无法读取
            if imgName[-1] == '\n':
                imgName = imgName[:-1]
            # 读取图片
            img = cv2.imread(os.path.join(imgPath, imgName))
            # 图片保存进列表
            dataList.append(transform(img))
            # 标签同样转成列表，最后统一转tensor
            labelList.append(labelName2number(imgName.split('_')[0]))


def dataLoader_v3(txtFileName='testImages_artphoto_trainset.txt', imgPath='testImages_artphoto', imgResize=(224,224)):

    import cv2

    transform = transforms.Compose([transforms.ToTensor(),
                                    # transforms.CenterCrop(imgResize[0]),
                                    transforms.Resize(imgResize),
                                    transforms.ColorJitter(brightness=0.5),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
    data1List, label1List = [], []
    data2List, label2List = [], []

    jud1 = cv2.CascadeClassifier('haarshare/haarcascade_frontalface_default.xml')
    jud2 = cv2.CascadeClassifier('haarshare/haarcascade_fullbody.xml')
    jud3 = cv2.CascadeClassifier('haarshare/haarcascade_upperbody.xml')
    juds = ['jud1','jud2','jud3']

    with open(txtFileName, "r", encoding='utf-8') as txtData:
        data = txtData.readlines()

    for imgName in data:
        catelog = False
        # imgName有个换行，不清掉无法读取
        if imgName[-1] == '\n':
            imgName = imgName[:-1]
        # 读取图片
        img = cv2.imread(os.path.join(imgPath, imgName))

        for jud in juds:
            if len(eval('%s.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.1, 3)' % jud)):
                catelog = True

        if catelog:
            # 图片保存进列表
            data1List.append(transform(img))
            # 标签同样转成列表，最后统一转tensor
            label1List.append(labelName2number(imgName.split('_')[0]))
        else:
            data2List.append(transform(img))
            label2List.append(labelName2number(imgName.split('_')[0]))

    train_data1 = torch.stack(data1List, 0)
    train_data2 = torch.stack(data2List, 0)
    # print(train_data.shape)
    # 返回值用torch.utils.data.DataLoader打开。 先转numpy再tensor是为了提速（大概）

    return torch.utils.data.TensorDataset(train_data1.float(), torch.from_numpy(np.array(label1List)).to(torch.int64)),\
           torch.utils.data.TensorDataset(train_data2.float(), torch.from_numpy(np.array(label2List)).to(torch.int64))


if __name__ == '__main__':
    dataLoader()

