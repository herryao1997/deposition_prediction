import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
import os
import json

def process_csv_and_generate_image(csv_file, img_count, output_folder):
    """处理单个CSV文件，生成热力图并返回文件名和条件向量"""
    # 读取CSV文件
    data = pd.read_csv(csv_file, sep=',')

    # 去除列名中的空格
    data.columns = data.columns.str.strip()

    # 只保留第2到第50行的数据
    data = data.iloc[1:50]

    # 清理数据中的空白和非数值内容
    data_cleaned = data[['x(mm)', 'y(mm)', 'd(nm)_L1']].apply(pd.to_numeric, errors='coerce')
    data_cleaned = data_cleaned.dropna()

    # 提取x, y和数值
    x = data_cleaned['x(mm)']
    y = data_cleaned['y(mm)']
    values = data_cleaned['d(nm)_L1']

    # 创建一个更密集的插值网格
    grid_x, grid_y = np.mgrid[min(x):max(x):50j, min(y):max(y):50j]

    # 对数据进行插值
    grid_values = griddata((x, y), values, (grid_x, grid_y), method='cubic')

    # 读取第1行第10列的条件向量
    condition_str = data.iloc[0, 9]  # 第10列（索引从0开始，第10列索引为9）
    conditions = [float(i) for i in condition_str.split('-')]

    # 生成图片名称
    img_name = f"image_{img_count:02d}.jpg"
    img_path = os.path.join(output_folder, img_name)

    # 生成热力图
    plt.figure(figsize=(0.5, 0.5))
    sns.heatmap(grid_values, cmap='viridis', cbar=False,
                vmin=550, vmax=750, xticklabels=False, yticklabels=False, square=True)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # 保存图片
    plt.savefig(img_path, dpi=100)
    plt.close()

    # 返回图片名称和对应的条件
    return img_name, conditions

def process_all_csv_files(input_folder, output_folder, json_file):
    """遍历所有CSV文件，生成图片并保存到JSON文件中"""
    # 创建输出文件夹，如果不存在就创建
    os.makedirs(output_folder, exist_ok=True)

    # 初始化用于存储图片和条件对应的JSON数据
    image_conditions = {}

    # 遍历文件夹中的所有 CSV 文件
    img_count = 1
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):
            csv_file = os.path.join(input_folder, file_name)

            # 处理单个CSV文件并生成图片
            img_name, conditions = process_csv_and_generate_image(csv_file, img_count, output_folder)

            # 将图片名和对应的条件存入字典
            image_conditions[img_name] = conditions

            # 更新图片编号
            img_count += 1

    # 保存 JSON 文件
    with open(json_file, 'w') as f:
        json.dump(image_conditions, f, indent=4)

    print(f"所有图片已生成并保存至 {output_folder}")
    print(f"JSON 文件已保存为 {json_file}")

# 运行函数
process_all_csv_files('csv_folder', 'output_images', 'image_conditions.json')
