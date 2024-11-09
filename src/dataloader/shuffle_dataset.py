import os
import glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import List

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
classes_to_description = {
    0:"This image is a basophil, it is characterized by its round shape and dark-staining granules. The nucleus is often obscured by these granules, which appear dense and dark purple when stained.",
    1:"This image displays an eosinophil, it is bilobed nucleus and distinctive red to pink staining granules. The granules fill most of the cell's cytoplasm, and the bilobed nucleus appears dark purple.",
    2:"This image shows an erythroblast，it is characterized by its large, dark-stained nucleus occupying much of the cell. The cytoplasm is relatively scant and takes on a bluish hue due to the presence of ribosomes. The erythroblast's round, dense nucleus and the smaller, darker cell body distinguish it from other cell types in the blood smear.",
    3:"this is a photo of ig, it is characterized by a large, round nucleus that occupies most of the cell’s volume, stained dark purple or blue.",
    4:"This is a photo of lymphocytes, it is characterized by their large, darkly stained nucleus, which occupies most of the cell’s interior, leaving only a thin rim of lighter cytoplasm. The nucleus is dense and round or slightly irregular in shape",
    5:"The nucleus is stained dark purple or blue, with the surrounding cytoplasm appearing lighter in color, it has a irregular or lobulated nucleus",
    6:"This is a neutrophil, it is horseshoe-shaped nucleus, the nucleus is stained dark purple, while the cytoplasm is lighter in color and appears to have a granular texture.",
    7:"The platelet in this image appears as a small, round, granular structure stained deep purple or blue. It is significantly smaller than other cells."        
}
# 定义类别列表
classes = list(classes_to_labels.keys())


class ImageListDataset(Dataset):
    def __init__(self, txt_files, transform=None, num_list:List[int]=None):
        """
        自定义数据集，读取多个txt文件，包含图像路径和标签。

        Args:
            txt_files (list): 包含多个txt文件路径的列表。
            transform (callable, optional): 图像变换函数。
        """
        self.image_labels = []
        for txt_file,num in zip(txt_files,num_list):
            with open(txt_file, 'r') as f:
                if num!=0:
                    for line,i in zip(f,range(num)):
                        line = line.strip()
                        if line:
                            try:
                                path, label = line.split(',')
                                self.image_labels.append((path, int(label)))
                            except ValueError:
                                print(f"警告: 文件 {txt_file} 中的行格式不正确: '{line}'")
                else:
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
                                        如果为None，则包含所有主类的txt文件（如 'basophil.txt'）
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
                    print(f"警告: 预期的文件 {root_dir}/{txt_file} 不存在。")
        else:
            # 包含所有主类的txt文件，但排除带有特定后缀的txt文件
            pattern = os.path.join(root_dir, f"{cls}.txt")
            if os.path.exists(pattern):
                txt_files.append(pattern)
            else:
                print(f"警告: 主类文件 {pattern} 不存在。")
    return txt_files


def construct_dataloader_from_txt(root_dir, splits=None, batch_size=32, num_workers=4, shuffle=True, num_list:List[int]=None, dataset_transform:list=None):
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
    if num_list is None:
        num_list = [0]*8
    txt_files = get_txt_files(root_dir, include_splits=splits)
    print(txt_files)
    if not txt_files:
        raise ValueError("未找到符合条件的txt文件。请检查目录和文件名。")

    # 定义图像变换
    if dataset_transform is None:
        transform = transforms.Compose([
            transforms.Resize((384, 384)),  # 调整图像大小
            transforms.ToTensor(),  # 转换为Tensor
            transforms.Normalize([0.874, 0.749, 0.722],  # 使用你的数据集的均值
                                [0.158, 0.185, 0.079])  # 使用你的数据集的标准差
        ])
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((384, 384)),  # 调整图像大小
                transforms.ToTensor(),  # 转换为Tensor
                transforms.Normalize(transform[0],transform[1])  # 使用你的数据集的标准差
            ]
        )

    # 创建自定义数据集
    dataset = ImageListDataset(txt_files, transform=transform, num_list=num_list)

    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


if __name__ == "__main__":
    labels_to_description = {
        0:"This image is a basophil, it is characterized by its round shape and dark-staining granules. The nucleus is often obscured by these granules, which appear dense and dark purple when stained.",
        1:"This image displays an eosinophil, it is bilobed nucleus and distinctive red to pink staining granules. The granules fill most of the cell's cytoplasm, and the bilobed nucleus appears dark purple.",
        2:"This image shows an erythroblast，it is characterized by its large, dark-stained nucleus occupying much of the cell. The cytoplasm is relatively scant and takes on a bluish hue due to the presence of ribosomes. The erythroblast's round, dense nucleus and the smaller, darker cell body distinguish it from other cell types in the blood smear.",
        3:"this is a photo of ig, it is characterized by a large, round nucleus that occupies most of the cell’s volume, stained dark purple or blue.",
        4:"This is a photo of lymphocytes, it is characterized by their large, darkly stained nucleus, which occupies most of the cell’s interior, leaving only a thin rim of lighter cytoplasm. The nucleus is dense and round or slightly irregular in shape",
        5:"The nucleus is stained dark purple or blue, with the surrounding cytoplasm appearing lighter in color, it has a irregular or lobulated nucleus",
        6:"This is a neutrophil, it is horseshoe-shaped nucleus, the nucleus is stained dark purple, while the cytoplasm is lighter in color and appears to have a granular texture.",
        7:"The platelet in this image appears as a small, round, granular structure stained deep purple or blue. It is significantly smaller than other cells."        
    }

