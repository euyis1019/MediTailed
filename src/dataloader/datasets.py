import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import transforms
from collections import defaultdict
import random
from torch.utils.data import Sampler

class PBCDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): 数据集根目录路径。
            transform (callable, optional): 应用于图像的转换。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes()
        print(self.classes)
        self.image_paths, self.labels = self._make_dataset()

    def _find_classes(self):
        """
        查找所有类别，并创建类别到索引的映射。
        返回：
            classes (list): 类别名称列表。
            class_to_idx (dict): 类别名称到索引的映射。
        """
        classes = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        class_to_num = {cls_name.split()[0]: int(cls_name.split()[1]) for cls_name in classes}
        #sorted_class_to_num = dict(sorted(class_to_num.items(), key=lambda item: item[1]))
        # 提取类别名称（去掉后面的数字）
        #----
        classes = [cls_name.split()[0] for cls_name in classes]
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self):
        """
        创建图像路径和标签的列表。
        返回：
            image_paths (list): 图像文件路径列表。
            labels (list): 对应的标签索引列表。
        """
        image_paths = []
        labels = []

        for cls in self.classes:
            # 拼接类别文件夹路径，查找包含类别名称的文件夹
            # 使用 glob 模式匹配，以捕捉类似 'basophil 1218' 的文件夹
            import glob
            cls_dir_pattern = os.path.join(self.root_dir, f"{cls}*")

            # 使用 glob 查找匹配该类别的文件夹
            matching_dirs = glob.glob(cls_dir_pattern)[0]

            if not matching_dirs:
                # 如果没有匹配的目录，继续下一个类别
                print(f"未找到匹配的目录: {cls_dir_pattern}")
                continue

            # for cls_dir in matching_dirs:
            #     # 查找 cls_dir 下的所有子文件夹（例如 basophil 1218）
            #     subdirs = [d for d in os.listdir(cls_dir) if os.path.isdir(os.path.join(cls_dir, d))]
            #     for subdir in subdirs:
            #         subdir_path = os.path.join(cls_dir, subdir)
            for fname in sorted(os.listdir(matching_dirs)):
                if self._is_image_file(fname):
                    path = os.path.join(matching_dirs, fname)
                    image_paths.append(path)
                    labels.append(self.class_to_idx[cls])  # 确保使用类别的正确索引

        return image_paths, labels

    def _is_image_file(self, filename):
        """检查文件是否为图像文件。"""
        IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        return filename.lower().endswith(IMG_EXTENSIONS)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        获取指定索引的图像和标签。
        Args:
            idx (int): 索引。
        Returns:
            image (Tensor): 转换后的图像。
            label (int): 标签索引。
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')  # 确保是RGB图像

        if self.transform:
            image = self.transform(image)

        return image, label

class PBCClassDataset(Dataset):
    def __init__(self, full_dataset, class_idx, transform=None):
        """
        Args:
            full_dataset (PBCDataset): 主数据集实例。
            class_idx (int): 需要隔离的类别索引。
            transform (callable, optional): 应用于图像的转换。
        """
        self.full_dataset = full_dataset
        self.class_idx = class_idx
        self.transform = transform
        self.indices = [i for i, label in enumerate(self.full_dataset.labels) if label == self.class_idx]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        image, label = self.full_dataset[actual_idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class BalancedBatchSampler(Sampler):
    def __init__(self, class_datasets, batch_size):
        """
        Args:
            class_datasets (dict): 每个类别对应的子数据集。
            batch_size (int): 每个批次的样本数量，必须是类别数量的倍数。
        """
        self.class_datasets = class_datasets
        self.batch_size = batch_size
        self.num_classes = len(class_datasets)
        assert batch_size % self.num_classes == 0, "Batch size must be divisible by the number of classes."
        self.samples_per_class = batch_size // self.num_classes
        self.class_indices = {cls: list(range(len(ds))) for cls, ds in self.class_datasets.items()}
        # 打乱每个类别的索引
        for cls in self.class_indices:
            random.shuffle(self.class_indices[cls])

    def __iter__(self):
        # 确定每个类别可用的最小样本数
        min_class_size = min([len(indices) for indices in self.class_indices.values()])
        num_batches = min_class_size // self.samples_per_class

        for batch_num in range(num_batches):
            batch = []
            for cls in self.class_datasets.keys():
                start = batch_num * self.samples_per_class
                end = start + self.samples_per_class
                batch += self.class_indices[cls][start:end]
            #random.shuffle(batch)  # 打乱批次内的样本顺序
            yield batch

    def __len__(self):
        min_class_size = min([len(indices) for indices in self.class_indices.values()])
        return min_class_size // self.samples_per_class

def create_dataloaders(root_dir, batch_size=32, num_workers=4):
    # 创建主数据集
    main_dataset = PBCDataset(root_dir=root_dir, transform=data_transforms)

    # 创建每个类别的子数据集
    class_datasets = {}
    for cls_name, cls_idx in main_dataset.class_to_idx.items():
        class_datasets[cls_name] = PBCClassDataset(main_dataset, cls_idx, transform=data_transforms)

    # 创建平衡采样器
    sampler = BalancedBatchSampler(class_datasets, batch_size)

    # 创建训练 DataLoader
    train_loader = DataLoader(main_dataset, batch_sampler=sampler, num_workers=num_workers)

    # 创建测试 DataLoaders（每个类别一个）
    test_loaders = {}
    for cls_name, cls_dataset in class_datasets.items():
        test_loader = DataLoader(cls_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loaders[cls_name] = test_loader

    return train_loader, test_loaders, main_dataset.class_to_idx

# 定义数据转换
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),          # 调整图像大小
    transforms.RandomHorizontalFlip(),      # 随机水平翻转
    transforms.ToTensor(),                  # 转换为张量
    transforms.Normalize([0.485, 0.456, 0.406],  # 标准化（基于ImageNet）
                         [0.229, 0.224, 0.225])
])

def main():
    # 数据集根目录
    dataset_root = r'D:\project\MediTailed\data\PBC_dataset_normal_DIB'

    # 定义批次大小和工作线程数
    batch_size = 32
    num_workers = 4

    # 创建 DataLoaders
    train_loader, test_loaders, class_to_idx = create_dataloaders(
        root_dir=dataset_root,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # 显示类别映射
    print("类别到索引的映射：")
    print(class_to_idx)

    # 迭代训练 DataLoader
    print("\n迭代训练 DataLoader：")
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f'批次 {batch_idx + 1}:')
        print(f' - 图像形状: {images.size()}')
        print(f' - 标签: {labels}')
        # 在这里添加训练代码
        if batch_idx == 10:  # 仅演示前 3 个批次
            break

    # 迭代每个类别的测试 DataLoader
    print("\n迭代测试 DataLoaders：")
    for cls_name, test_loader in test_loaders.items():
        print(f'\n测试类别：{cls_name}')
        for batch_idx, (images, labels) in enumerate(test_loader):
            print(f'  批次 {batch_idx + 1}:')
            print(f'   - 图像形状: {images.size()}')
            print(f'   - 标签: {labels}')
            # 在这里添加测试代码
            if batch_idx == 0:  # 仅演示每个类别的第一个批次
                break

if __name__ == "__main__":
    main()
