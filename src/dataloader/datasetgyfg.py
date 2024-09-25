import os
import glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, ConcatDataset
from torchvision import transforms
from collections import defaultdict
import numpy as np

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

# 类别与图像数量的对应关系
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

classes = list(classes_to_labels.keys())


def match_directory(cls, root_dir):
    """
    使用 glob 查找匹配类别的目录。

    Args:
        cls (str): 类别名称。
        root_dir (str): 数据集根目录。

    Returns:
        str: 匹配的目录路径。

    Raises:
        ValueError: 如果未找到匹配的目录。
    """
    cls_dir_pattern = os.path.join(root_dir, f"{cls}*")
    matched_dirs = glob.glob(cls_dir_pattern)
    if not matched_dirs:
        raise ValueError(f"No directory matched for class '{cls}' in '{root_dir}'")
    return matched_dirs[0]


class CustomImageDataset(Dataset):
    """
    自定义图像数据集，用于加载来自不同类别子目录的 .jpg 图像。
    """

    def __init__(self, root_dir, classes, classes_to_labels, classes_to_num, transform=None):
        """
        初始化数据集。

        Args:
            root_dir (str): 数据集根目录。
            classes (list): 类别名称列表。
            classes_to_labels (dict): 类别名称到标签的映射。
            classes_to_num (dict): 类别名称到图像数量的映射。
            transform (callable, optional): 图像变换函数。
        """
        self.root_dir = root_dir
        self.classes = classes
        self.classes_to_labels = classes_to_labels
        self.classes_to_num = classes_to_num
        self.transform = transform or transforms.ToTensor()
        self.image_labels = self._generate_image_list()

    def _generate_image_list(self):
        """
        生成图像路径和对应标签的列表。

        Returns:
            list: [(image_path, label), ...]
        """
        image_list = []
        for cls in self.classes:
            num_images = self.classes_to_num.get(cls, 0)
            cls_dir = match_directory(cls, self.root_dir)
            # 获取所有 .jpg 文件
            all_images = glob.glob(os.path.join(cls_dir, '*.jpg'))
            if not all_images:
                raise ValueError(f"No .jpg files found in directory '{cls_dir}' for class '{cls}'")
            # 随机选择指定数量的图像
            selected_images = random.sample(all_images, min(num_images, len(all_images)))
            label = self.classes_to_labels[cls]
            # 添加到图像列表
            image_list.extend([(img_path, label) for img_path in selected_images])
        return image_list

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        image_path, label = self.image_labels[idx]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise IOError(f"无法打开图像文件 {image_path}: {e}")

        if self.transform:
            image = self.transform(image)
        return image, label


class BalancedBatchSampler(Sampler):
    """
    自定义批采样器，确保每个批次中每个类别的样本数量均衡。
    当某个类别的样本耗尽时，继续从其他未耗尽的类别中采样。
    """

    def __init__(self, dataset, classes, classes_to_labels, batch_size, shuffle=True):
        """
        初始化采样器。

        Args:
            dataset (CustomImageDataset): 自定义数据集。
            classes (list): 类别名称列表。
            classes_to_labels (dict): 类别名称到标签的映射。
            batch_size (int): 每个批次的总样本数，必须是类别数的倍数。
            shuffle (bool): 是否在每个 epoch 开始时打乱每个类别的样本顺序。
        """
        self.dataset = dataset
        self.classes = classes
        self.classes_to_labels = classes_to_labels
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.num_classes = len(classes)
        assert batch_size % self.num_classes == 0, "Batch size must be divisible by the number of classes."
        self.samples_per_class = batch_size // self.num_classes

        # 创建类别到样本索引的映射
        self.class_to_indices = self._map_class_to_indices()

    def _map_class_to_indices(self):
        """
        创建类别到样本索引的映射。

        Returns:
            dict: {cls: [indices], ...}
        """
        class_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(self.dataset.image_labels):
            cls = [k for k, v in self.classes_to_labels.items() if v == label][0]
            class_to_indices[cls].append(idx)

        # 打乱每个类别的索引
        if self.shuffle:
            for cls in self.classes:
                random.shuffle(class_to_indices[cls])
        return class_to_indices

    def __iter__(self):
        """
        迭代器，生成批次索引。

        Yields:
            list: 批次样本的索引列表。
        """
        # 初始化每个类别的指针
        pointers = {cls: 0 for cls in self.classes}

        active_classes = set(self.classes)  # 当前活跃的类别

        while active_classes:
            batch = []
            classes_to_remove = set()
            for cls in list(active_classes):
                start = pointers[cls]
                end = start + self.samples_per_class
                cls_indices = self.class_to_indices[cls]
                if end > len(cls_indices):
                    # 某个类别的数据不足以继续采样，标记为需移除
                    classes_to_remove.add(cls)
                    # 取剩余的样本
                    batch.extend(cls_indices[start:])
                    pointers[cls] = len(cls_indices)  # 移动指针到末尾
                else:
                    batch.extend(cls_indices[start:end])
                    pointers[cls] += self.samples_per_class

            # 移除已耗尽的类别
            active_classes -= classes_to_remove

            # 检查当前批次是否满足 batch_size
            if len(batch) == self.batch_size:
                random.shuffle(batch)  # 打乱批次内的样本顺序
                yield batch
            else:
                # 处理不满的批次，根据需求选择如何处理
                # 这里选择跳过不满的批次
                print(f"警告: 批次大小为 {len(batch)}，小于预期的 {self.batch_size}，将跳过该批次。")
                continue

    def __len__(self):
        """
        返回可生成的批次数量。

        Returns:
            int: 批次数量。
        """
        min_class_size = min([len(indices) for indices in self.class_to_indices.values()])
        return min_class_size // self.samples_per_class


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
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise IOError(f"无法打开图像文件 {img_path}: {e}")

        if self.transform:
            image = self.transform(image)
        return image, label


def construct_dataloader_from_txt(root_dir, splits=None, batch_size=32, num_workers=4, shuffle=True):
    """
    从指定的txt文件构建DataLoader。

    Args:
        root_dir (str): 数据集根目录。
        splits (list, optional): 需要包含的txt文件后缀，如 ['train', 'val', 'test']。
                                 如果为None，则包含所有主类的txt文件。
        batch_size (int): 每个批次的样本数量，必须是类别数的倍数。
        num_workers (int): DataLoader的工作线程数量。
        shuffle (bool): 是否在每个 epoch 开始时打乱每个类别的样本顺序。

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
        transforms.Normalize([0.485, 0.456, 0.406],  # 使用ImageNet的均值
                             [0.229, 0.224, 0.225])  # 使用ImageNet的标准差
    ])

    # 创建自定义数据集
    dataset = ImageListDataset(txt_files, transform=transform)

    # 创建平衡批采样器
    sampler = BalancedBatchSampler(
        dataset=dataset,
        classes=classes,
        classes_to_labels=classes_to_labels,
        batch_size=batch_size,
        shuffle=shuffle
    )

    # 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers
    )

    return dataloader


if __name__ == "__main__":
    # 示例使用
    root_dir = r'D:\project\MediTailed\data\PBC_dataset_normal_DIB'  # 替换为实际路径
    batch_size = 32  # 确保 batch_size 是类别数的倍数
    num_workers = 4
    splits = ['train', 'val']  # 可选: ['train', 'val', 'test'], 或者设置为 None 以加载主类的txt文件

    # 构建 DataLoader
    dataloader = construct_dataloader_from_txt(root_dir, splits=splits, batch_size=batch_size, num_workers=num_workers,
                                               shuffle=True)

    # 使用 DataLoader 进行迭代
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f" - Images shape: {images.shape}")  # 例如: torch.Size([32, 3, 224, 224])
        print(f" - Labels: {labels}")  # 例如: tensor([0, 1, 0, ..., 7])

        # 在这里添加你的训练代码
        if batch_idx == 200:import os
import glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, ConcatDataset
from torchvision import transforms
from collections import defaultdict
import numpy as np

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

# 类别与图像数量的对应关系
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

classes = list(classes_to_labels.keys())


def match_directory(cls, root_dir):
    """
    使用 glob 查找匹配类别的目录。

    Args:
        cls (str): 类别名称。
        root_dir (str): 数据集根目录。

    Returns:
        str: 匹配的目录路径。

    Raises:
        ValueError: 如果未找到匹配的目录。
    """
    cls_dir_pattern = os.path.join(root_dir, f"{cls}*")
    matched_dirs = glob.glob(cls_dir_pattern)
    if not matched_dirs:
        raise ValueError(f"No directory matched for class '{cls}' in '{root_dir}'")
    return matched_dirs[0]


class CustomImageDataset(Dataset):
    """
    自定义图像数据集，用于加载来自不同类别子目录的 .jpg 图像。
    """

    def __init__(self, root_dir, classes, classes_to_labels, classes_to_num, transform=None):
        """
        初始化数据集。

        Args:
            root_dir (str): 数据集根目录。
            classes (list): 类别名称列表。
            classes_to_labels (dict): 类别名称到标签的映射。
            classes_to_num (dict): 类别名称到图像数量的映射。
            transform (callable, optional): 图像变换函数。
        """
        self.root_dir = root_dir
        self.classes = classes
        self.classes_to_labels = classes_to_labels
        self.classes_to_num = classes_to_num
        self.transform = transform or transforms.ToTensor()
        self.image_labels = self._generate_image_list()

    def _generate_image_list(self):
        """
        生成图像路径和对应标签的列表。

        Returns:
            list: [(image_path, label), ...]
        """
        image_list = []
        for cls in self.classes:
            num_images = self.classes_to_num.get(cls, 0)
            cls_dir = match_directory(cls, self.root_dir)
            # 获取所有 .jpg 文件
            all_images = glob.glob(os.path.join(cls_dir, '*.jpg'))
            if not all_images:
                raise ValueError(f"No .jpg files found in directory '{cls_dir}' for class '{cls}'")
            # 随机选择指定数量的图像
            selected_images = random.sample(all_images, min(num_images, len(all_images)))
            label = self.classes_to_labels[cls]
            # 添加到图像列表
            image_list.extend([(img_path, label) for img_path in selected_images])
        return image_list

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        image_path, label = self.image_labels[idx]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise IOError(f"无法打开图像文件 {image_path}: {e}")

        if self.transform:
            image = self.transform(image)
        return image, label


class BalancedBatchSampler(Sampler):
    """
    自定义批采样器，确保每个批次中每个类别的样本数量均衡。
    当某个类别的样本耗尽时，继续从其他未耗尽的类别中采样。
    """

    def __init__(self, dataset, classes, classes_to_labels, batch_size, shuffle=True):
        """
        初始化采样器。

        Args:
            dataset (CustomImageDataset): 自定义数据集。
            classes (list): 类别名称列表。
            classes_to_labels (dict): 类别名称到标签的映射。
            batch_size (int): 每个批次的总样本数，必须是类别数的倍数。
            shuffle (bool): 是否在每个 epoch 开始时打乱每个类别的样本顺序。
        """
        self.dataset = dataset
        self.classes = classes
        self.classes_to_labels = classes_to_labels
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.num_classes = len(classes)
        assert batch_size % self.num_classes == 0, "Batch size must be divisible by the number of classes."
        self.samples_per_class = batch_size // self.num_classes

        # 创建类别到样本索引的映射
        self.class_to_indices = self._map_class_to_indices()

    def _map_class_to_indices(self):
        """
        创建类别到样本索引的映射。

        Returns:
            dict: {cls: [indices], ...}
        """
        class_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(self.dataset.image_labels):
            cls = [k for k, v in self.classes_to_labels.items() if v == label][0]
            class_to_indices[cls].append(idx)

        # 打乱每个类别的索引
        if self.shuffle:
            for cls in self.classes:
                random.shuffle(class_to_indices[cls])
        return class_to_indices

    def __iter__(self):
        """
        迭代器，生成批次索引。

        Yields:
            list: 批次样本的索引列表。
        """
        # 初始化每个类别的指针
        pointers = {cls: 0 for cls in self.classes}

        active_classes = set(self.classes)  # 当前活跃的类别

        while active_classes:
            batch = []
            classes_to_remove = set()
            for cls in list(active_classes):
                start = pointers[cls]
                end = start + self.samples_per_class
                cls_indices = self.class_to_indices[cls]
                if end > len(cls_indices):
                    # 某个类别的数据不足以继续采样，标记为需移除
                    classes_to_remove.add(cls)
                    # 取剩余的样本
                    batch.extend(cls_indices[start:])
                    pointers[cls] = len(cls_indices)  # 移动指针到末尾
                else:
                    batch.extend(cls_indices[start:end])
                    pointers[cls] += self.samples_per_class

            # 移除已耗尽的类别
            active_classes -= classes_to_remove

            # 检查当前批次是否满足 batch_size
            if len(batch) == self.batch_size:
                random.shuffle(batch)  # 打乱批次内的样本顺序
                yield batch
            else:
                # 处理不满的批次，根据需求选择如何处理
                # 这里选择跳过不满的批次
                print(f"警告: 批次大小为 {len(batch)}，小于预期的 {self.batch_size}，将跳过该批次。")
                continue

    def __len__(self):
        """
        返回可生成的批次数量。

        Returns:
            int: 批次数量。
        """
        min_class_size = min([len(indices) for indices in self.class_to_indices.values()])
        return min_class_size // self.samples_per_class


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
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise IOError(f"无法打开图像文件 {img_path}: {e}")

        if self.transform:
            image = self.transform(image)
        return image, label


def construct_dataloader_from_txt(root_dir, splits=None, batch_size=32, num_workers=4, shuffle=True):
    """
    从指定的txt文件构建DataLoader。

    Args:
        root_dir (str): 数据集根目录。
        splits (list, optional): 需要包含的txt文件后缀，如 ['train', 'val', 'test']。
                                 如果为None，则包含所有主类的txt文件。
        batch_size (int): 每个批次的样本数量，必须是类别数的倍数。
        num_workers (int): DataLoader的工作线程数量。
        shuffle (bool): 是否在每个 epoch 开始时打乱每个类别的样本顺序。

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
        transforms.Normalize([0.485, 0.456, 0.406],  # 使用ImageNet的均值
                             [0.229, 0.224, 0.225])  # 使用ImageNet的标准差
    ])

    # 创建自定义数据集
    dataset = ImageListDataset(txt_files, transform=transform)

    # 创建平衡批采样器
    sampler = BalancedBatchSampler(
        dataset=dataset,
        classes=classes,
        classes_to_labels=classes_to_labels,
        batch_size=batch_size,
        shuffle=shuffle
    )

    # 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers
    )

    return dataloader


if __name__ == "__main__":
    # 示例使用
    root_dir = r'D:\project\MediTailed\data\PBC_dataset_normal_DIB'  # 替换为实际路径
    batch_size = 32  # 确保 batch_size 是类别数的倍数
    num_workers = 4
    splits = ['train', 'val']  # 可选: ['train', 'val', 'test'], 或者设置为 None 以加载主类的txt文件

    # 构建 DataLoader
    dataloader = construct_dataloader_from_txt(root_dir, splits=splits, batch_size=batch_size, num_workers=num_workers,
                                               shuffle=True)

    # 使用 DataLoader 进行迭代
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f" - Images shape: {images.shape}")  # 例如: torch.Size([32, 3, 224, 224])
        print(f" - Labels: {labels}")  # 例如: tensor([0, 1, 0, ..., 7])

        # 在这里添加你的训练代码
        if batch_idx == 200:  # 仅演示前3个批次
            break
  # 仅演示前3个批次
