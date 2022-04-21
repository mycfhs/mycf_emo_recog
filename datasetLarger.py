import cv2
import os
import torch
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
pict_files = 'data/train1/'
pict_save_files = 'data/train5/'
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.CenterCrop((224, 224)),
    # transforms.Resize(imgResize),
    transforms.ColorJitter(brightness=0.9),
    transforms .LinearTransformation(),
    transforms.ToPILImage('RGB')

])
if not os.path.exists(pict_save_files):
    os.makedirs(pict_save_files)
for _, _, files in os.walk(pict_files, topdown=False):
    for file in tqdm(files):
        img = cv2.imread(os.path.join(pict_files, file))
        img = transform(img)

        img = np.array(img)
        cv2.imwrite(pict_save_files + '%s_v4.jpg' % file[:-4],img)
        # break