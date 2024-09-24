# sampling_strategy = { 'basophil': 2, 'eosinophil': 3, 'erythroblast': 2, 'ig': 2, 'lymphocyte': 1, 'monocyte': 1, 'neutrophil': 3, 'platelet': 2 }，意味着在第一个batch（假设batch=8）的选取中我先选basophil的dataloader的两张图，然后eosinophil的3张图，erythroblast的2张图， ig只能拿一张图了。然后在第二个batch（假设batch=8）中，先拿ig的第一张图，再拿lymphocyte的一张图，以此类推直到拿到了8张图。
#

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
import random
import glob

# Classes and their corresponding labels
classes_to_labels = {
    'basophil': 0, 'eosinophil': 1, 'erythroblast': 2, 'ig': 3,
    'lymphocyte': 4, 'monocyte': 5, 'neutrophil': 6, 'platelet': 7
}

# Classes and the number of images for each
classes_to_num = {
    'basophil': 1218, 'eosinophil': 3117, 'erythroblast': 1551, 'ig': 2895,
    'lymphocyte': 1214, 'monocyte': 1420, 'neutrophil': 3330, 'platelet': 2348
}

classes = ['basophil', 'eosinophil', 'erythroblast', 'ig',
           'lymphocyte', 'monocyte', 'neutrophil', 'platelet']

def MatchDirectory(cls, root_dir):
    cls_dir_pattern = os.path.join(root_dir, f"{cls}*")
    # 使用 glob 查找匹配该类别的文件夹
    return glob.glob(cls_dir_pattern)[0]

class SubDataset(Dataset):
    """ Register a dataset """
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
        return image_paths

    def __len__(self):
        return min(self.num_images, len(self.image_paths))

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.label
        # 将图像转换为 PyTorch Tensor
        if self.transform:
            image = self.transform(image)
        return image, label

def ConstructSubdataset():
    ''' Return a dictionary. Key is class name, Value is corresponding dataset. '''
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

# 自定义采样器类，用于控制每个类别的数据采样
class CustomSampler(Sampler):
    def __init__(self, datasets, sampling_strategy):
        """
        初始化采样器。

        Args:
            datasets (dict): 类别到数据集的映射
            sampling_strategy (dict): 类别到采样数量的映射
        """
        self.datasets = datasets
        self.sampling_strategy = sampling_strategy
        self.indices = self._generate_indices()

    def _generate_indices(self):
        indices = []
        for cls, num_samples in self.sampling_strategy.items():
            dataset = self.datasets[cls]
            dataset_indices = list(range(len(dataset)))
            selected_indices = random.sample(dataset_indices, num_samples)
            indices.extend(selected_indices)
        return indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def ConstructCustomDataset(sampling_strategy):
    datasets = ConstructSubdataset()
    dataloaders = {}
    for cls, dataset in datasets.items():
        dataloaders[cls] = DataLoader(dataset, batch_size=1, shuffle=False)
    return datasets, CustomSampler(datasets, sampling_strategy)

# 示例采样策略: 从每个类别中采样一定数量的样本
sampling_strategy = {
    'basophil': 200, 'eosinophil': 300, 'erythroblast': 150,
    'ig': 250, 'lymphocyte': 100, 'monocyte': 120,
    'neutrophil': 330, 'platelet': 230
}

# 创建自定义数据集和采样器
combined_dataset, custom_sampler = ConstructCustomDataset(sampling_strategy)

# 创建 DataLoader
dataloader = DataLoader(combined_dataset, batch_size=1, sampler=custom_sampler)

# 使用 DataLoader
for images, labels in dataloader:
    print(f"Batch of images: {images.size()}, Batch of labels: {labels[0]}")

