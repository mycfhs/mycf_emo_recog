import torch


def train_model(model, device, train_iter, optimizer,criterion, epoch):
    # 模型训练
    model.train()
    for batch_index, (data, label) in enumerate(train_iter):
        data, label = data.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(data)

        # 计算当前损失
        loss = criterion(output, label)

        # 反向传播计算梯度
        loss.backward()

        # 更新所有参数
        optimizer.step()

        if batch_index % 200 == 0:
            print('Epoch [{}], Step [{}], Loss: {:.4f}'.format(epoch, batch_index, loss))



def test_model(model, device,criterion, test_loader):
    # 模型验证
    model.eval()
    # 正确率
    correct = 0.0
    # 测试损失
    test_loss = 0.0
    with torch.no_grad(): # 在测试阶段，不用计算梯度，也不用进行反向传播
        for data, label in test_loader:
            # 部署到DEVICE上
            data, label = data.to(device), label.to(device)
            # 测试数据
            output = model(data)
            # 计算测试损失
            test_loss += criterion(output, label).item()
            # 找到概率最大值的下标
            _, predict = torch.max(output, dim=1) # 这里返回最大值和最大值索引
            # 累计正确的值
            correct += predict.eq(label.view_as(predict)).sum().item()
        test_loss /= len(test_loader.dataset)
        print("Test -- Average loss : {:.4f}, Accuracy : {:.3f}\n".format(test_loss, 100.0 * correct/len(test_loader.dataset)))