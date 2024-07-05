import os
import sys
import time
import json
import matplotlib

from caltech256 import Caltech256

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from tqdm import tqdm
from model import resnet101
from torchprofile import profile_macs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    tb_writer = SummaryWriter()
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(size=(224, 224), padding=4),
            transforms.RandomGrayscale(p=0.1),
            transforms.ColorJitter(brightness=0.2, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.507, 0.487, 0.441], [0.267, 0.256, 0.276])
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.507, 0.487, 0.441], [0.267, 0.256, 0.276])
        ])
    }

    data_root = 'D:/Code/resnet/data/archive/256_ObjectCategories/'  # 存储数据的根目录
    train_dataset = Caltech256(root=data_root, transform=data_transform['train'], splits=[1, 2, 3, 4])
    validate_dataset = Caltech256(root=data_root, transform=data_transform['val'], splits=[5])

    train_num = len(train_dataset)
    val_num = len(validate_dataset)

    batch_size = 100
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    print(f"Is CUDA available: {torch.cuda.is_available()}")
    print(f"Current device: {device}")

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)  # 替换 num_workers=0 为 num_workers=nw

    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = resnet101(num_classes=257, include_top=True)
    new_weights_dict = net.state_dict()
    resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
    weights_dict = resnet.state_dict()
    # net.load_state_dict(weights_dict)
    for k in weights_dict.keys():
        if k in new_weights_dict.keys() and not k.startswith('fc'):
            new_weights_dict[k] = weights_dict[k]
    net.load_state_dict(new_weights_dict, strict=False)
    train_params = []
    train_layer = ['part']
    for name, params in net.named_parameters():
        if any(name.startswith(prefix) for prefix in train_layer):
            print(name)
            train_params.append(params)
        else:
            params.requires_grad = False

    ## net.fc = nn.Linear(net.fc.in_features, 100)params.requires_grad 设置为 True
    net.to(device)
    for param in net.parameters():
        # 设置参数的 requires_grad 属性为 False，即冻结参数
        param.requires_grad = False
    for param in net.fc.parameters():
        param.requires_grad = True
    print(f'Total parameters: {count_parameters(net)}')

    # define loss function
    loss_function = nn.CrossEntropyLoss()
    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001, weight_decay=1e-5)
    epochs = 100
    best_acc = 0.0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    save_path = './resNet101.pth'
    train_steps = len(train_loader)
    val_steps = len(validate_loader)
    for epoch in range(epochs):
        # train
        net.train()
        train_correct = 0  # 初始化训练集准确匹配数量
        train_total = 0  # 初始化训练集样本总数
        running_loss = 0.0
        validate_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            # 计算训练集准确匹配数量
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == labels.to(device)).sum().item()
            train_total += labels.size(0)
            train_bar.desc = "train epoch[{}/{}] loss:{:.4f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
            # 计算训练集准确率
            train_accuracy = train_correct / train_total
            train_loss = running_loss / train_steps

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        val_loss = 0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                loss1 = loss_function(outputs, val_labels.to(device))
                validate_loss += loss1.item()
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_bar.desc = "valid epoch[{}/{}] loss:{:.4f}".format(epoch + 1,
                                                           epochs, loss1)
        val_accurate = acc / val_num
        val_loss = validate_loss /val_steps
        print('[epoch %d]train_accuracy: %.4f train_loss: %.4f  val_loss: %.4f val_accuracy: %.4f' %
              (epoch + 1, train_accuracy, running_loss / train_steps, val_loss, val_accurate))
        # 打印训练集准确率
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
            with open('training_results_base.txt', 'a') as file:
                file.write(
                    f'Epoch {epoch + 1}: train_loss: {running_loss / train_steps}, val_accuracy: {val_accurate}\n')
                file.write(f'Best validation accuracy: {val_accurate}\n')
        tags = ["train_loss", "train_acc", "val_acc", "val_loss"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_accuracy, epoch)
        tb_writer.add_scalar(tags[2], val_accurate, epoch)
        tb_writer.add_scalar(tags[3], val_loss, epoch)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accurate)
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('(cifar-10-fc)Loss_Plot.png')  # 保存图像到文件
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(train_accuracies, label='Training Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('(cifar-10-fc)Accuracy_Plot.png')  # 保存图像到文件
        plt.close()  # 释放资源

    print(f'Best validation accuracy: {best_acc}')
    print('Training results saved to training_results.txt')
    print('Finished Training')


if __name__ == '__main__':
    main()

