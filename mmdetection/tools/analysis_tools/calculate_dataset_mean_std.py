import numpy as np
import os
import cv2
import os.path as osp


def compute_mean_std(dataset_dir):
    """
    计算给定数据集中所有图像的RGB通道的平均值和标准差

    参数:
        dataset_dir (str): 数据集路径，包含所有图像文件的目录

    返回:
        mean (np.ndarray): RGB通道的均值，形状为(3,)
        std (np.ndarray): RGB通道的标准差，形状为(3,)
    """
    # 初始化变量
    pixel_num = 0
    channel_sum = np.zeros(3)
    channel_sum_squared = np.zeros(3)
    idx = 0

    # # 遍历数据集中的所有图像
    # for i in os.listdir(dataset_dir):
    #     for video in os.listdir(osp.join(dataset_dir, i, 'Images')):
    #         for filename in os.listdir(osp.join(dataset_dir, i, 'Images', video)):
    #             img = cv2.imread(osp.join(dataset_dir, i, 'Images', video, filename))
    #             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB通道顺序
    #             img = img / 255.0  # 将像素值从[0, 255]缩放到[0, 1]
    #             pixel_num += img.size / 3
    #             channel_sum += np.sum(img, axis=(0, 1))
    #             channel_sum_squared += np.sum(np.square(img), axis=(0, 1))
    #             print(idx)
    #             idx += 1

    # 遍历数据集中的所有图像
    for i in os.listdir(dataset_dir):
        img = cv2.imread(osp.join(dataset_dir, i))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB通道顺序
        img = img / 255.0  # 将像素值从[0, 255]缩放到[0, 1]
        pixel_num += img.size / 3
        channel_sum += np.sum(img, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(img), axis=(0, 1))
        print(idx)
        idx += 1


    # 计算RGB通道的均值和标准差
    rgb_mean = channel_sum / pixel_num
    rgb_std = np.sqrt(channel_sum_squared / pixel_num - np.square(rgb_mean))

    return rgb_mean, rgb_std


if __name__ == '__main__':
    rgb_mean, rgb_std = compute_mean_std("/home/wsco/SATA_DATA/gty/dataset/CVC-VideoClinicDB/image-train")
    print("数据集的mean为：", rgb_mean)
    print("数据集的std为：", rgb_std)

### LDPolypVideo
# TrainValid RGB
# 数据集的mean为： [168.81 104.04 73.95]
# 数据集的std为： [68.595 52.785 41.82]

# PreTrain RGB
# 数据集的mean为： [166.77 108.12 67.83]   [0.654 0.424 0.266]
# 数据集的std为： [65.535 46.41 37.74]     [0.257 0.182 0.148]


### CVC-VideoClinicDB
# Train RGB
# 数据集的mean为： [0.38817579 0.24357532 0.15155816] [98.94 61.965 38.505]
# 数据集的std为： [0.31526254 0.20623617 0.14466483]  [80.325 52.53 36.975]
