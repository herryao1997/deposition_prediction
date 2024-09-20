import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self, json_file, image_folder, transform=None):
        """
        初始化数据集
        Args:
            json_file (str): JSON 文件路径，包含图片与条件的对应关系
            image_folder (str): 图片文件夹路径
            transform (callable, optional): 对图片进行的变换操作
        """
        # 读取 JSON 文件，得到图片和条件的对应关系
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        """返回数据集的长度"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        根据索引返回单个数据，包括条件向量（con）和图片
        Args:
            idx (int): 数据索引
        Returns:
            con (torch.Tensor): 条件向量
            image (torch.Tensor): 图片 (50x50)
        """
        # 获取图片文件名和条件向量
        img_name = list(self.data.keys())[idx]
        con = torch.tensor(self.data[img_name], dtype=torch.float32)

        # 读取图片
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert('L')  # 读取为灰度图

        # 如果有预处理操作，应用到图片上
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)  # 默认转换为 Tensor

        return con, image


def get_dataloader(json_file, image_folder, batch_size=32, shuffle=True, transform=None):
    """
    返回 DataLoader
    Args:
        json_file (str): JSON 文件路径
        image_folder (str): 图片文件夹路径
        batch_size (int): 每个 batch 的大小
        shuffle (bool): 是否打乱数据集
        transform (callable, optional): 对图片的变换操作
    Returns:
        DataLoader: Pytorch 的 DataLoader 对象
    """
    dataset = CustomDataset(json_file, image_folder, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# 调用示例
# dataloader = get_dataloader('image_conditions.json', 'output_images', batch_size=32, shuffle=True)
