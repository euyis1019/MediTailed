import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

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
classes_to_labels_WBC = {
    'basophil': 0,
    'eosinophil':1,
    'lymphocyte': 2,
    'monocyte': 3,
    'neutrophil': 4,
}
classes_to_labels = classes_to_labels_WBC
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

def construct_dataloader_from_txt(root_dir, splits=None, batch_size=32, num_workers=4, shuffle=True, transform=None):
    """
    从指定的txt文件构建DataLoader。

    Args:
        root_dir (str): 数据集根目录。
        splits (list, optional): 需要包含的txt文件后缀，如 ['train', 'val', 'test']。
                                 如果为None，则包含所有主类的txt文件。
        batch_size (int): 每个批次的样本数量。
        num_workers (int): DataLoader的工作线程数量。
        shuffle (bool): 是否打乱数据顺序。
        transform (callable, optional): 图像变换函数。

    Returns:
        DataLoader: 构建好的DataLoader对象。
    """
    txt_files = get_txt_files(root_dir, include_splits=splits)
    if not txt_files:
        raise ValueError("未找到符合条件的txt文件。请检查目录和文件名。")

    # 创建自定义数据集
    dataset = ImageListDataset(txt_files, transform=transform)

    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def compute_mean_std(dataloader):
    """
    计算数据集的RGB均值和标准差。

    Args:
        dataloader (DataLoader): 数据加载器。

    Returns:
        tuple: (均值, 标准差)，每个为3元素的列表，对应RGB通道。
    """
    n_channels = 3
    mean = torch.zeros(n_channels)
    std = torch.zeros(n_channels)
    total_pixels = 0

    for batch_idx, (images, _) in enumerate(dataloader):
        # images.shape = [batch_size, 3, H, W]
        batch_samples = images.size(0)
        pixels = images.size(2) * images.size(3)
        total_pixels += batch_samples * pixels

        # 计算每个通道的和
        mean += images.sum([0, 2, 3])

        # 计算每个通道的平方和
        std += (images ** 2).sum([0, 2, 3])

        if batch_idx % 100 == 0:
            print(f"处理批次 {batch_idx + 1}")

    mean /= total_pixels
    std = torch.sqrt(std / total_pixels - mean ** 2)

    return mean.tolist(), std.tolist()

if __name__ == "__main__":
    root_dir = '/home/yf_guo/PAPER/MediTailed/data/LDWBC'  # 替换为实际路径

    # 选择要包含的txt文件后缀
    splits = None  # 通常建议仅在训练集上计算

    # 定义图像变换（不包含归一化）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为Tensor
    ])

    # 构建 DataLoader
    dataloader = construct_dataloader_from_txt(
        root_dir,
        splits=splits,
        batch_size=32,
        num_workers=4,
        shuffle=False,  # 计算统计量时无需打乱
        transform=transform
    )

    # 计算均值和标准差
    print("开始计算数据集的均值和标准差...")
    mean, std = compute_mean_std(dataloader)
    print(f"数据集的RGB均值: {mean}")
    print(f"数据集的RGB标准差: {std}")

    # 使用计算得到的均值和标准差进行归一化
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize(mean, std)  # 使用计算得到的均值和标准差
    ])

    # 重新构建 DataLoader，使用新的归一化参数
    dataloader = construct_dataloader_from_txt(
        root_dir,
        splits=splits,
        batch_size=32,
        num_workers=4,
        shuffle=True,  # 训练时通常需要打乱
        transform=transform
    )

    # 示例迭代
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        # print(f" - Images shape: {images.shape}")  # 例如: torch.Size([32, 3, 224, 224])
        print(f" - Labels: {labels}")  # 例如: tensor([0, 1, 0, ..., 7])

        # 在这里添加你的训练代码
        if batch_idx == 2:  # 仅演示前3个批次
            break
