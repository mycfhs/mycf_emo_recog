from torchvision import models
from dataLoader import *
from torch.utils.data import DataLoader
from train_test_func import *


BATCH_SIZE = 8
NUM_WORKERS = 0
LEARNING_RATE = 0.5
train_data,_ = dataLoader_v3(txtFileName='testImages_artphoto_trainset.txt')
train_dataset = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_data,_ = dataLoader_v3(txtFileName='testImages_artphoto_testset.txt')
test_dataset = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=NUM_WORKERS)
print('data loaded!')
# 数据加载，模型训练等代码需要自己补全

# model = models.vgg19_bn(pretrained=True)
model = models.vit_b_32(pretrained=True)
# model.heads=torch.nn.Sequential(torch.nn.Linear(1000, 8))
# model.classifier[6] = torch.nn.Sequential(torch.nn.Linear(4096, 8))
model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 8),
                                       # torch.nn.ReLU(),
                                       # torch.nn.Dropout(p=0.5),
                                       # torch.nn.Linear(4096, 4096),
                                       # torch.nn.ReLU(),
                                       # torch.nn.Dropout(p=0.5),
                                       # torch.nn.Linear(4096, 8)
                                       )

model = model.cuda()

criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
# optimizer = torch.optim.Adagrad(model.classifier.parameters(), lr=LEARNING_RATE, lr_decay=0.003)
optimizer = torch.optim.SGD(model.classifier.parameters(), lr=LEARNING_RATE,momentum=0.9)

StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

DEVICE = 'cuda'
EPOCHS = 150

total_step = len(train_data)


print('start training!')
for epoch in range(1, EPOCHS + 1):
    train_model(model, DEVICE, train_dataset, optimizer, criterion, epoch)
    test_model(model, DEVICE, criterion, test_dataset)
    # StepLR.step()
# torch.save(model.state_dict(), 'model_lenet.pth')  # 模型的状态字典

