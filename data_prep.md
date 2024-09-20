## 代码解析：

### `process_csv_and_generate_image` 函数：

- 处理单个 CSV 文件，生成热力图并保存图片。
- 返回生成的图片名称和条件向量。

### `process_all_csv_files` 函数：

- 遍历指定的文件夹 `input_folder` 中所有的 CSV 文件。
- 对每个 CSV 文件调用 `process_csv_and_generate_image` 函数，生成图片并获取条件。
- 将每张图片的名称和对应的条件记录到字典 `image_conditions` 中。
- 最终保存为 JSON 文件。

## 目录结构：

- **输入**：`csv_folder` 文件夹下包含多个 CSV 文件。
- **输出**：生成的图片保存在 `output_images` 文件夹中，文件名依次为 `image_01.jpg`, `image_02.jpg` 等。所有图片和它们的条件信息保存到 `image_conditions.json` 文件中。

### 调用示例：

```python
# 假设所有 CSV 文件都保存在 'csv_folder' 文件夹中
# 生成的图片保存在 'output_images' 文件夹中
# JSON 文件保存为 'image_conditions.json'

process_all_csv_files('csv_folder', 'output_images', 'image_conditions.json')