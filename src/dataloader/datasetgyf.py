import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler
from torchvision import transforms
from collections import defaultdict
import random
import numpy as np

# Classes and their corresponding labels
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

# Classes and the number of images for each
classes_to_num = {
    'basophil': 1218,
    'eosinophil': 3117,
    'erythroblast': 1551,
    'ig': 2895,
    'lymphocyte': 1214,
    'monocyte': 1420,
    'neutrophil': 3330,
    'platelet': 2348
}

classes = ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']

def MatchDirectory(cls, root_dir):
    import glob
    cls_dir_pattern = os.path.join(root_dir, f"{cls}*")

    # 使用 glob 查找匹配该类别的文件夹
    matched_dirs = glob.glob(cls_dir_pattern)
    if not matched_dirs:
        raise ValueError(f"No directory matched for class {cls} in {root_dir}")
    return matched_dirs[0]

class SubDataset(Dataset):
    """
    Register a dataset
    """
    def __init__(self, root_dir, cls: str, num_images: int, transform=None):
        """
        自定义图像数据集，用于加载 .jpg 格式的图像。

        Args:
            root_dir (str): 包含 .jpg 文件的最小目录，也就是图像文件所在的根目录。
            transform (callable, optional): 用于数据变换的函数，默认为 None。
        """
        self.root_dir = root_dir
        self.transform = transform or transforms.ToTensor()  # 默认情况下使用 ToTensor 转换
        self.cls = cls
        self.num_images = num_images
        self.label = classes_to_labels[cls]

        # 加载图像路径
        self.image_paths = self._load_image_paths()

    def _load_image_paths(self):
        image_paths = [os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir) if f.endswith('.jpg')]
        if not image_paths:
            raise ValueError(f"No .jpg files found in {self.root_dir}")
        # Shuffle the image paths to ensure random sampling
        random.shuffle(image_paths)
        return image_paths[:self.num_images]  # 预先选择指定数量的图像

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.label

        # 将图像转换为 PyTorch Tensor
        if self.transform:
            image = self.transform(image)

        return image, label

def ConstructSubdataset():
    '''
    Return a dictionary. Key is class name, Value is corresponding dataset.
    '''
    root_dir = 'D:\\project\\MediTailed\\data\\PBC_dataset_normal_DIB'  # 替换为你存放 .jpg 图像的实际路径
    datasets = {}
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor()  # 转换为Tensor
    ])
    for cls in classes:
        num = classes_to_num[cls]
        dir = MatchDirectory(cls, root_dir)
        dataset = SubDataset(dir, cls, num, transform=transform)
        datasets[cls] = dataset

    return datasets

class CustomBatchSampler(Sampler):
    def __init__(self, datasets, sampling_strategy, batch_size):
        """
        初始化自定义批采样器。

        Args:
            datasets (dict): 类别到数据集的映射
            sampling_strategy (dict): 类别到每批次采样数量的映射
            batch_size (int): 每个批次的总样本数
        """
        self.datasets = datasets
        self.sampling_strategy = sampling_strategy
        self.batch_size = batch_size

        # 验证 sampling_strategy 的总数是否等于 batch_size
        total_samples = sum(sampling_strategy.values())
        if total_samples != batch_size:
            raise ValueError(f"Sampling strategy total {total_samples} does not match batch size {batch_size}")

        # 为每个类别创建一个索引列表
        self.class_indices = {}
        for cls, dataset in datasets.items():
            indices = list(range(len(dataset)))
            random.shuffle(indices)  # 随机打乱索引
            self.class_indices[cls] = indices

        # 计算每个类别的批次数
        self.num_batches = min(len(indices) // sampling_strategy[cls] for cls, indices in self.class_indices.items())

    def __iter__(self):
        # 创建每个类别的指针
        pointers = {cls: 0 for cls in self.datasets.keys()}

        for _ in range(self.num_batches):
            batch = []
            for cls, num_samples in self.sampling_strategy.items():
                start = pointers[cls]
                end = start + num_samples
                cls_indices = self.class_indices[cls]
                if end > len(cls_indices):
                    # 如果某个类别的数据不足以继续采样，停止迭代
                    return
                batch.extend([self._get_global_index(cls, idx) for idx in cls_indices[start:end]])
                pointers[cls] += num_samples
            yield batch

    def _get_global_index(self, cls, local_idx):
        """
        将类别内的索引转换为全局索引。

        Args:
            cls (str): 类别名称
            local_idx (int): 类别内的本地索引

        Returns:
            int: 全局索引
        """
        # 计算全局索引，假设 ConcatDataset 按照 classes 的顺序连接
        class_order = list(self.datasets.keys())
        global_idx = 0
        for c in class_order:
            if c == cls:
                break
            global_idx += len(self.datasets[c])
        return global_idx + local_idx

    def __len__(self):
        # 计算可以生成的批次数
        return self.num_batches

def ConstructCustomDataset(sampling_strategy, batch_size):
    """
    构建自定义的大数据集和相应的 BatchSampler。

    Args:
        sampling_strategy (dict): 类别到每批次采样数量的映射
        batch_size (int): 每个批次的总样本数

    Returns:
        DataLoader: 自定义的 DataLoader
    """
    datasets = ConstructSubdataset()

    # 创建组合数据集
    combined_dataset = ConcatDataset([datasets[cls] for cls in classes])

    # 创建自定义批采样器
    batch_sampler = CustomBatchSampler(datasets, sampling_strategy, batch_size)

    # 创建 DataLoader
    dataloader = DataLoader(combined_dataset, batch_sampler=batch_sampler, num_workers=4)

    return dataloader

if __name__ == "__main__":
    # 示例采样策略: 从每个类别中采样一定数量的样本
    sampling_strategy = {
        'basophil': 2,
        'eosinophil': 3,
        'erythroblast': 2,
        'ig': 1,
        'lymphocyte': 1,
        'monocyte': 1,
        'neutrophil': 3,
        'platelet': 2
    }

    batch_size = sum(sampling_strategy.values())

    dataloader = ConstructCustomDataset(sampling_strategy, batch_size)

    # 使用 DataLoader
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Images size: {images.size()}")
        print(f"  Labels: {labels}")
        # 在这里可以进行训练或其他操作
