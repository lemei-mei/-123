import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.filters import gabor
import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.cluster import KMeans

def preprocess(image):
    # 数据增强
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    image = cv2.flip(image, 1)

    # 线性变换
    min_val = np.min(image)
    max_val = np.max(image)
    image = (image - min_val) / (max_val - min_val)

    # 直方图均衡化
    image = cv2.equalizeHist((image * 255).astype(np.uint8))

    # 大津法分割
    thresh = threshold_otsu(image)
    binary = image > thresh

    # Gabor方向滤波器
    gabor_filtered = [gabor(binary, frequency=0.6, theta=theta) for theta in np.arange(0, np.pi, np.pi / 8)]

    processed_image = np.stack(gabor_filtered, axis=2)
    return processed_image

class DeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DeformConv2d, self).__init__()

        self.deform_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.p_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size, padding=padding)

    def forward(self, x):
        p = self.p_conv(x)
        p = p.view(x.size(0), -1, 2, x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)
        return F.deform_conv2d(x, p, self.deform_conv.weight)

class CBAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CBAM, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channel, channel // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channel // reduction, channel, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.sigmoid(self.fc2(self.relu(self.fc1(self.avg_pool(x)))))
        max_out = self.sigmoid(self.fc2(self.relu(self.fc1(self.max_pool(x)))))
        out = avg_out + max_out
        return x * out

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            DeformConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DeformConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet_CBAM_DeformConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet_CBAM_DeformConv, self).__init__()

        self.encoder = nn.Sequential(
            DoubleConv(in_channels, 64),
            nn.MaxPool2d(2),
            DoubleConv(64, 128),
            nn.MaxPool2d(2),
            DoubleConv(128, 256),
            nn.MaxPool2d(2),
            DoubleConv(256, 512),
            nn.MaxPool2d(2)
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(512, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(256, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(128, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.cbam = CBAM(out_channels)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.encoder[0](x)
        x2 = self.encoder[1](x1)
        x3 = self.encoder[2](x2)
        x4 = self.encoder[3](x3)
        x5 = self.encoder[4](x4)

        y = self.decoder[0](x5)
        y = torch.cat([y, x4], dim=1)
        y = self.decoder[1](y)
        y = torch.cat([y, x3], dim=1)
        y = self.decoder[2](y)
        y = torch.cat([y, x2], dim=1)
        y = self.decoder[3](y)
        y = torch.cat([y, x1], dim=1)
        y = self.decoder[4](y)

        y = self.cbam(y)
        y = self.final_conv(y)

        return y

def detect_and_segment(image):
    # 多尺度处理
    scales = [(0.5, 1), (0.75, 1.25), (1, 1), (1.25, 0.75), (1, 0.5)]
    detections = []
    for scale in scales:
        resized_image = cv2.resize(image, (int(image.shape[1] * scale[1]), int(image.shape[0] * scale[0])))
        processed_image = preprocess(resized_image)
        processed_image = torch.from_numpy(processed_image).unsqueeze(0).float()

        # 加载模型
        model = UNet_CBAM_DeformConv(in_channels=16, out_channels=2)
        model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
        model.eval()

        # 预测
        with torch.no_grad():
            output = model(processed_image)

        output = output[0].numpy()
        output = cv2.resize(output.transpose(1, 2, 0), (resized_image.shape[1], resized_image.shape[0]))
        output = cv2.resize(output, (image.shape[1], image.shape[0]))

        detections.append(output)

    # 聚类分析
    flattened_detections = np.concatenate([detection.reshape(-1, 2) for detection in detections], axis=0)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(flattened_detections)
    segmented_image = kmeans.labels_.reshape(image.shape[:2])

    return segmented_image