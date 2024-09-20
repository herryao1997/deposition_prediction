import torch
import torch.nn as nn
import torch.optim as optim
from model.model import Generator, Discriminator  # 导入生成器和判别器
from model.dataloader import get_dataloader  # 导入自定义的数据加载器

def train(generator, discriminator, dataloader, epochs, lr=0.0002, device='cpu'):
    """
    训练生成器和判别器模型 (cGAN)
    Args:
        generator (nn.Module): 生成器模型
        discriminator (nn.Module): 判别器模型
        dataloader (DataLoader): 加载数据的DataLoader
        epochs (int): 训练的轮数
        lr (float): 学习率
        device (str): 使用的设备 'cpu' 或 'cuda'
    """
    # 定义损失函数为二元交叉熵损失
    criterion = nn.BCELoss()

    # 定义生成器和判别器的优化器
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

    # 将模型移至设备（CPU 或 GPU）
    generator.to(device)
    discriminator.to(device)

    for epoch in range(epochs):
        for i, (con, real_images) in enumerate(dataloader):
            con = con.to(device)
            real_images = real_images.to(device)

            # 真标签 (real) 和 假标签 (fake)
            valid = torch.ones(con.size(0), 1, device=device)
            fake = torch.zeros(con.size(0), 1, device=device)

            # ------------------
            # 训练判别器 (Discriminator)
            # ------------------
            optimizer_D.zero_grad()

            # 判别真实图片
            real_pred = discriminator(con, real_images)
            real_loss = criterion(real_pred, valid)

            # 使用生成器生成图片
            fake_images = generator(con)
            fake_pred = discriminator(con, fake_images.detach())  # detach不反向传播
            fake_loss = criterion(fake_pred, fake)

            # 判别器总损失
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # ------------------
            # 训练生成器 (Generator)
            # ------------------
            optimizer_G.zero_grad()

            # 生成图片，并希望判别器认为是“真实”的
            fake_pred = discriminator(con, fake_images)
            g_loss = criterion(fake_pred, valid)

            g_loss.backward()
            optimizer_G.step()

            # 打印训练进度
            if i % 100 == 99:  # 每 100 个 batch 打印一次
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{len(dataloader)}], "
                      f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    print("训练完成！")

if __name__ == "__main__":
    # 超参数设置
    epochs = 20
    batch_size = 32
    lr = 0.0002
    condition_dim = 3  # 条件向量维度

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化生成器和判别器模型
    generator = Generator(condition_dim=condition_dim)
    discriminator = Discriminator(condition_dim=condition_dim)

    # 加载数据
    dataloader = get_dataloader('image_conditions.json', 'output_images', batch_size=batch_size, shuffle=True)

    # 开始训练
    train(generator, discriminator, dataloader, epochs=epochs, lr=lr, device=device)
