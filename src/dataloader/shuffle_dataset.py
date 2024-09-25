import os
import glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 定义类别与标签的对应关系
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

# 定义类别列表
classes = list(classes_to_labels.keys())


class ImageListDataset(Dataset):
    def __init__(self, txt_files, transform=None):
        """
        自定义数据集，读取多个txt文件，包含图像路径和标签。

        Args:
            txt_files (list): 包含多个txt文件路径的列表。
            transform (callable, optional): 图像变换函数。
        """
        self.image_labels = []
        for txt_file in txt_files:
            with open(txt_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            path, label = line.split(',')
                            self.image_labels.append((path, int(label)))
                        except ValueError:
                            print(f"警告: 文件 {txt_file} 中的行格式不正确: '{line}'")
        self.transform = transform

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_path, label = self.image_labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise IOError(f"无法打开图像文件 {img_path}: {e}")

        if self.transform:
            image = self.transform(image)
        return image, label


def get_txt_files(root_dir, include_splits=None):
    """
    获取根目录下的所有 .txt 文件，或者仅包含特定后缀的txt文件。

    Args:
        root_dir (str): 包含txt文件的根目录。
        include_splits (list, optional): 需要包含的txt文件后缀，如 ['train', 'val']。
                                        如果为None，则包含所有主类的txt文件（如 'basophil.txt'）。

    Returns:
        list: 符合条件的txt文件路径列表。
    """
    txt_files = []
    for cls in classes:
        if include_splits:
            for split in include_splits:
                txt_file = os.path.join(root_dir, f"{cls}_{split}.txt")
                if os.path.exists(txt_file):
                    txt_files.append(txt_file)
                else:
                    print(f"警告: 预期的文件 {txt_file} 不存在。")
        else:
            # 包含所有主类的txt文件，但排除带有特定后缀的txt文件
            pattern = os.path.join(root_dir, f"{cls}.txt")
            if os.path.exists(pattern):
                txt_files.append(pattern)
            else:
                print(f"警告: 主类文件 {pattern} 不存在。")
    return txt_files


def construct_dataloader_from_txt(root_dir, splits=None, batch_size=32, num_workers=4, shuffle=True):
    """
    从指定的txt文件构建DataLoader。

    Args:
        root_dir (str): 数据集根目录。
        splits (list, optional): 需要包含的txt文件后缀，如 ['train', 'val', 'test']。
                                 如果为None，则包含所有主类的txt文件。
        batch_size (int): 每个批次的样本数量。
        num_workers (int): DataLoader的工作线程数量。
        shuffle (bool): 是否打乱数据顺序。

    Returns:
        DataLoader: 构建好的DataLoader对象。
    """
    txt_files = get_txt_files(root_dir, include_splits=splits)
    if not txt_files:
        raise ValueError("未找到符合条件的txt文件。请检查目录和文件名。")

    # 定义图像变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize([0.874, 0.749, 0.722],  # 使用你的数据集的均值
                             [0.158, 0.185, 0.079])  # 使用你的数据集的标准差
    ])

    # 创建自定义数据集
    dataset = ImageListDataset(txt_files, transform=transform)

    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


if __name__ == "__main__":
    labels_to_description = {

    }
    root_dir = r'D:\project\MediTailed\data\PBC_dataset_normal_DIB'  # 替换为实际路径

    # 选择要包含的txt文件后缀
    # 例如，仅加载训练集和验证集的txt文件
    splits = None  # 可选: ['train', 'val', 'test'], 或者设置为 None 以加载主类的txt文件

    # 构建 DataLoader
    dataloader = construct_dataloader_from_txt(root_dir, splits=splits, batch_size=32, num_workers=4, shuffle=True)

    # 使用 DataLoader 进行迭代
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        #print(f" - Images shape: {images.shape}")  # 例如: torch.Size([32, 3, 224, 224])
        print(f" - Labels: {labels}")  # 例如: tensor([0, 1, 0, ..., 7])

        # 在这里添加你的训练代码
        if batch_idx == 2000:  # 仅演示前3个批次
            break
