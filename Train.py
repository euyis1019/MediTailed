import torch
from torch import nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ram.models import ram_plus
from ram import get_transform
import numpy as np
from ram.utils import build_openset_llm_label_embedding
import json
from dataloader.shuffle_dataset import construct_dataloader_from_txt
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import datetime
# Attributes  of each dataset are different, so we need to convert the labels to onehot encoding
# Transform

def convert_to_onehot(labels, num_classes=8):
    batch_size = labels.size(0)
    onehot = torch.zeros(batch_size, num_classes, device=labels.device)
    onehot.scatter_(1, labels.unsqueeze(1), 1)
    return onehot

# 初始化设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化 TensorBoard writer
# 使用当前时间创建唯一的运行目录
current_time = datetime.datetime.now().strftime("%m-%d %H:%M")
log_dir = f'Medi/{current_time}/ram_experiment_onlyVisual'
writer = SummaryWriter(log_dir)

# 初始化模型
model = ram_plus(
    pretrained="/home/yf_guo/PAPER/model_weights.pth",
    image_size=384,
    vit='swin_l'
)
# 设定模型的标记嵌入等参数
with open("dataset_config.json", "r") as f:
    dataset_config = json.load(f)

# List available dataset names
dataset_names = list(dataset_config.keys())
print("Available datasets:")
for i, name in enumerate(dataset_names, 1):
    print(f"{i}: {name}")

# Prompt the user to choose a dataset by number
try:
    dataset_choice = int(input("Enter the number corresponding to the dataset: ")) - 1
    #dataset_choice = 
    if 0 <= dataset_choice < len(dataset_names):
        dataset_name = dataset_names[dataset_choice]
        dataset_config = dataset_config[dataset_name]
        folder_name = dataset_config["folder_name"]
        class_name = dataset_config["class"]
        num_class = len(class_name)
    else:
        print("Invalid choice. Please enter a number from the list.")
except ValueError:
    print("Please enter a valid integer.")


llm_tag_des = dataset_config["details"]

openset_label_embedding, openset_categories = build_openset_llm_label_embedding(llm_tag_des)

model.tag_list = np.array(openset_categories)
model.label_embed = nn.Parameter(openset_label_embedding.float())
model.num_class = len(openset_categories)

model = model.to(device)
model.train()

# for name, param in model.named_parameters():
#     param.requires_grad = "visual_encoder" in name
# #打印允许微调的参数
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name)
# 构建 DataLoader
root_dir = '/home/yf_guo/PAPER/MediTailed/data/' + folder_name
splits = ['train']
num_list = [0]
dataloader = construct_dataloader_from_txt(root_dir, splits=splits, batch_size=18, num_workers=4, shuffle=True, num_list=[0]*8)
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.000001)

global_step = 0
for epoch in range(10):  # 假设我们训练10个epoch
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        onehot_labels = convert_to_onehot(labels, num_class)
        loss = criterion(outputs, onehot_labels)
        loss.backward()
        optimizer.step()
        
        # 获取预测结果
        predicted = outputs.argmax(dim=1)
        accuracy = (predicted == labels).float().mean().item()
        
        # 记录到 TensorBoard
        writer.add_scalar('Loss/train', loss.item(), global_step)
        writer.add_scalar('Accuracy/train', accuracy, global_step)
        
                # 记录每个参数的变化情况到 TensorBoard
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         # 记录参数的L2范数
        #         param_norm = torch.norm(param, p=2).item()
        #         writer.add_scalar(f'Param_L2/{name}', param_norm, global_step)

        #         # 记录参数的直方图分布
        #         writer.add_histogram(f'Param_Hist/{name}', param, global_step)
                                     
        print(f"Epoch {epoch+1}, Batch {batch_idx + 1}:")
        print(f" - Labels: {labels}")
        print(f" - Predictions: {predicted}")
        print(f" - Accuracy: {accuracy:.4f}")
        print(f" - Loss: {loss.item():.4f}")
        
        global_step += 1
    save_dir = f'Medi/{dataset_name}/{global_step}_numlist_{num_list}'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), f'Medi/{dataset_name}/{global_step}_numlist_{num_list}/model_weights_ram.pth')
# 关闭 TensorBoard writer
writer.close()

