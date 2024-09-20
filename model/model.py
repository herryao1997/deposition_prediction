# -*- coding: utf-8 -*-
"""
model.py

Contains the generator and discriminator models and the metrics used
to evaluate them.

Author: Chengxi Yao
Email: stevenyao@g.skku.edu
"""


__author__ = "Chengxi Yao"
__email__ = "stevenyao@g.skku.edu"

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, condition_dim, output_dim=50*50):
        """
        初始化生成器模型
        Args:
            condition_dim (int): 条件向量的维度
            output_dim (int): 生成图片的像素数量，默认 50x50
        """
        super(Generator, self).__init__()

        self.noise = nn.Sequential(

        )

    def forward(self, con):
        """
        前向传播，生成图片
        Args:
            con (torch.Tensor): 条件向量
        Returns:
            img (torch.Tensor): 生成的图片 (50x50)
        """
        img = self.fc(con)
        img = img.view(con.size(0), 1, 50, 50)  # reshape 为 (batch_size, 1, 50, 50)
        return img

class Discriminator(nn.Module):
    def __init__(self, condition_dim, image_size=50*50):
        """
        初始化判别器模型
        Args:
            condition_dim (int): 条件向量的维度
            image_size (int): 图片的尺寸 (50*50)
        """
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(condition_dim + image_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 输出 0~1 之间的概率
        )

    def forward(self, con, img):
        """
        前向传播，判断图片是真实还是生成的
        Args:
            con (torch.Tensor): 条件向量
            img (torch.Tensor): 输入的图片
        Returns:
            validity (torch.Tensor): 判别结果
        """
        img_flat = img.view(img.size(0), -1)  # 展平图片
        combined_input = torch.cat((con, img_flat), dim=1)  # 将条件和图片连接
        validity = self.fc(combined_input)
        return validity
