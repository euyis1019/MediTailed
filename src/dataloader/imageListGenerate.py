import os
import glob
import random

# 类别与标签的对应关系
classes_to_labels = {
    'basophil': 0,
    'eosinophil': 1,
    'erythroblast': 2,
    'ig': 3,
    'lymphocyte': 4,
    'monocyte': 5,
    'neutrophil': 6,
    'platelet': 7
}

# 根目录，包含各个类别的子文件夹
root_dir = r'D:\project\MediTailed\data\PBC_dataset_normal_DIB'

# 分割比例
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# 遍历每个类别，生成对应的txt文件
for cls, label in classes_to_labels.items():
    pattern = os.path.join(root_dir, f"{cls}*")
    matched_dirs = glob.glob(pattern)

    if not matched_dirs:
        print(f"警告: 未找到类别 '{cls}' 的目录，匹配模式: '{pattern}'")
        continue

    cls_dir = matched_dirs[0]
    image_files = glob.glob(os.path.join(cls_dir, '*.jpg'))

    if not image_files:
        print(f"警告: 在目录 '{cls_dir}' 中未找到任何 .jpg 文件。")
        continue

    # 打乱图像列表
    random.shuffle(image_files)

    # 计算各部分数量
    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]

    # 定义输出txt文件路径
    train_txt = os.path.join(root_dir, f"{cls}_train.txt")
    val_txt = os.path.join(root_dir, f"{cls}_val.txt")
    test_txt = os.path.join(root_dir, f"{cls}_test.txt")

    # 写入训练集
    with open(train_txt, 'w') as f:
        for img_path in train_files:
            f.write(f"{img_path},{label}\n")
    print(f"创建训练文件: {train_txt}，包含 {len(train_files)} 条记录。")

    # 写入验证集
    with open(val_txt, 'w') as f:
        for img_path in val_files:
            f.write(f"{img_path},{label}\n")
    print(f"创建验证文件: {val_txt}，包含 {len(val_files)} 条记录。")

    # 写入测试集
    with open(test_txt, 'w') as f:
        for img_path in test_files:
            f.write(f"{img_path},{label}\n")
    print(f"创建测试文件: {test_txt}，包含 {len(test_files)} 条记录。")
