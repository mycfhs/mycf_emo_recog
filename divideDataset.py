import cv2
import os
from tqdm import tqdm


def divideDataset():
    MODEL = 'test'
    txtFileName = 'testImages_artphoto_%sset.txt'%MODEL
    imgPath = 'testImages_artphoto'
    outDir = 'data/%s/'%MODEL

    with open(txtFileName, "r", encoding='utf-8') as txtData:
        data = txtData.readlines()

    for imgName in tqdm(data):
        # imgName有个换行，不清掉无法读取
        if imgName[-1] == '\n':
            imgName = imgName[:-1]
        # 读取图片
        img = cv2.imread(os.path.join(imgPath, imgName))
        cv2.imwrite(os.path.join(outDir, imgName), img)


if __name__ == '__main__':
    divideDataset()
