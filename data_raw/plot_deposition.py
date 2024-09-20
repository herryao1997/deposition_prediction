import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata

# 读取CSV文件，指定分隔符为逗号
data = pd.read_csv('RESULT.CSV', sep=',')

# 去除列名中的空格
data.columns = data.columns.str.strip()

# 只保留前50行并删除非数值的列，选择第2-50行的数据
data = data.iloc[1:50]  # 跳过第1行并获取第2到第50行

# 清理数据中的空白和非数值内容
data_cleaned = data[['x(mm)', 'y(mm)', 'd(nm)_L1']].apply(pd.to_numeric, errors='coerce')

# 移除包含NaN值的行
data_cleaned = data_cleaned.dropna()

# 提取x, y和数值
x = data_cleaned['x(mm)']
y = data_cleaned['y(mm)']
values = data_cleaned['d(nm)_L1']

# 创建一个更密集的插值网格
grid_x, grid_y = np.mgrid[min(x):max(x):50j, min(y):max(y):50j]

# 对数据进行插值
# grid_values = griddata((x, y), values, (grid_x, grid_y), method='cubic')
# 对数据进行插值
grid_values = griddata((x, y), values, (grid_x, grid_y), method='cubic')

# 设置colorbar范围，假设你想要设置最小值为600，最大值为750
vmin = 550
vmax = 750

# 绘制热力图
plt.figure(figsize=(0.5, 0.5))
sns.heatmap(grid_values, cmap='viridis', cbar=False, cbar_kws={'label': 'd(nm)_L1'},
            vmin=vmin, vmax=vmax, xticklabels=False, yticklabels=False, square=True)
# # 隐藏坐标轴
# plt.gca().set_xticks([])
# plt.gca().set_yticks([])
#
# # 隐藏边框
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)
#
# # 设置坐标轴标签
# plt.xlabel('x (mm)')
# plt.ylabel('y (mm)')
# plt.title('Interpolated Heatmap of d(nm)_L1')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# 显示图形
plt.show()
