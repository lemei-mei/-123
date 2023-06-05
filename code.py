# 导入必要的库
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage.filters import threshold_otsu
from skimage.transform import resize

# 定义Unet网络
class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2+2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = nn.functional.relu(self.conv1(x))
        x2 = nn.functional.relu(self.conv2(nn.functional.max_pool2d(x1, 2)))
        x3 = nn.functional.relu(self.conv3(nn.functional.max_pool2d(x2, 2)))
        x4 = nn.functional.relu(self.conv4(nn.functional.max_pool2d(x3, 2)))
        x5 = nn.functional.relu(self.conv5(nn.functional.max_pool2d(x4, 2)))
        x = nn.functional.relu(self.upconv1(x5))
        x = torch.cat([x, x4], dim=1)
        x = nn.functional.relu(self.upconv2(x))
        x = torch.cat([x, x3], dim=1)
        x = nn.functional.relu(self.upconv3(x))
        x = torch.cat([x, x2], dim=1)
        x = nn.functional.relu(self.upconv4(x))
        x = torch.cat([x, x1], dim=1)
        x = nn.functional.relu(self.conv6(x))
        return x

# 定义数据预处理函数
def preprocess(img):
    # 线性变换
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = img * 255
    img = np.uint8(img)

    # 图像增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # 大津法分割
    thresh = threshold_otsu(img)
    binary = img > thresh
    binary = binary.astype(np.uint8)

    # Gabor方向滤波器
    kernel_size = 31
    sigma = 4
    theta = np.pi / 4
    lambda_ = 10
    gamma = 0.5
    psi = 0
    kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lambda_, gamma, psi, ktype=cv2.CV_32F)
    filtered = cv2.filter2D(img, cv2.CV_8UC3, kernel)

    # 调整图像大小
    resized_img = resize(filtered,(256, 256), anti_aliasing=True)

    # 将二值图像转为单通道灰度图像
    binary = np.expand_dims(binary, axis=2)
    binary = binary.astype(np.float32)
    binary = binary * 255

    # 将调整大小后的图像和二值图像合并成单个输入
    input_img = np.concatenate((resized_img, binary), axis=2)

    # 将图像转为张量并返回
    input_img = torch.from_numpy(input_img).permute(2, 0, 1)
    return input_img

# 定义函数用于读取数据集
def load_dataset(dataset_path):
    # 读取图像和标签
    images_path = os.path.join(dataset_path, 'images')
    labels_path = os.path.join(dataset_path, 'labels')

    images = []
    labels = []
    for i in range(len(os.listdir(images_path))):
        img = cv2.imread(os.path.join(images_path, f'{i}.png'), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(os.path.join(labels_path, f'{i}.png'), cv2.IMREAD_GRAYSCALE)
        images.append(img)
        labels.append(label)

    # 对图像进行预处理
    images = [preprocess(img) for img in images]

    # 将图像的张量和标签张量分别存储在X和y中，并返回
    X = torch.stack(images)
    y = torch.stack(labels)
    return X, y

# 定义函数用于训练模型
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs):
    train_losses = []
    valid_losses = []
    for epoch in range(num_epochs):
        train_loss = 0.0
        valid_loss = 0.0
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        model.eval()
        with torch.no_grad():
            for inputs, targets in valid_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                valid_loss += loss.item() * inputs.size(0)
            valid_loss /= len(valid_loader.dataset)
            valid_losses.append(valid_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}")
    return model, train_losses, valid_losses

# 定义函数用于测试模型
def test_model(model, test_loader):
    # 设置模型为评估模式
    model.eval()

    # 定义变量以计算指标
    total_correct = 0
    total_pixels = 0
    dice_coefficient = 0

    # 遍历测试集并进行预测
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            predicted = torch.round(torch.sigmoid(outputs))

            # 计算准确率
            total_correct += (predicted == targets).sum().item()
            total_pixels += targets.numel()

            # 计算Dice系数
            intersection = (predicted * targets).sum().item()
            dice_coefficient += (2.0 * intersection) / (predicted.sum().item() + targets.sum().item())

    # 计算指标
    accuracy = total_correct / total_pixels
    dice_coefficient /= len(test_loader)

    return accuracy, dice_coefficient