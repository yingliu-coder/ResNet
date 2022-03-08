import torch
from torchvision import datasets, transforms
from torch.utils.data import dataloader
from resnet import ResNet18
import torch.nn as nn


batch_sz = 128
learn_rate = 1e-3
train_num = 1000

# 导入训练集数据
train_data = dataloader.DataLoader(
    datasets.CIFAR10(root='data/', train=True, transform=transforms.Compose([
        transforms.Resize(32, 32),      # 重新设置图片大小
        transforms.ToTensor(),      # 将图片转化为tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])         # 进行归一化
    ]), download=True), shuffle=True, batch_size=batch_sz
)

# 导入测试集数据
train_test = dataloader.DataLoader(
    datasets.CIFAR10(root='data/', train=False, transform=transforms.Compose([
        transforms.Resize(32, 32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True), shuffle=True, batch_size=batch_sz
)

device = torch.device('cuda')           # 设置在GPU上进行训练

# 定义模型
model = ResNet18().to(device)

# 定义损失函数和优化方式
criteon = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

# 训练模型
for epoch in range(1):
    model.train()
    for batch_idx, (x, label) in enumerate(train_data):
        x = x.to(device)
        label = label.to(device)

        logits = model(x)       # 经过模型得到的数据

        loss = criteon(logits, label)
        print('logits:', logits[0])
        print('label:', label[0])
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx == len(train_data) - 1:
            print(epoch, 'loss:', loss.item())

    # 进行测试
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_num = 0
        for x, label in train_test:
            x = x.to(device)
            label = label.to(device)

            logits = model(x)

            pred = logits.argmax(dim=1)

            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)

        acc = total_correct / total_num
        print(epoch, 'test acc:', acc)
