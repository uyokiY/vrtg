import os
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler

from utils import compute_features, TimeSeriesDataset, BiLSTM, train, load_whole_data

from config import checkpoint_dir, model_load, features, window_size, rolling_window_size, num_epochs, batch_size, input_size, hidden_size, output_size, num_layers


train_data, test_data = load_whole_data('data', sample_ids=[29,30,32,40,46,47,51,62,79,92])
train_data = compute_features(train_data, rolling_window_size)

X_train, y_train = train_data[features].values, (train_data['status'] == 'abnormal').astype(int).values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

flight_ids_train = train_data['flight_id'].values
indices_train = train_data['index'].values

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"训练集 异常样本比例: {sum(y_train)/len(y_train):.4f}")

train_dataset = TimeSeriesDataset(X_train, y_train, flight_ids_train, indices_train, window_size)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True) # 随机改变各个样本的顺序，但是窗口内部不会打乱

# test_dataset = TimeSeriesDataset(X_test, y_test, flight_ids_test, indices_test, window_size)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)

a = 0
for batch_x, batch_y, flight_id, index in train_loader:
    # batch_x: (batch_size, 101, feature_dim)
    # batch_y: (batch_size, 101)
    # flight_id: (batch_size,)
    # index: (batch_size,)
    print("Batch X shape:", batch_x.shape)
    print("Batch Y shape:", batch_y.shape)
    a += 1
    if a > 0:   
        break

print(f"训练集样本数: {len(train_dataset)}")
# print(f"测试集样本数: {len(test_dataset)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# 初始化模型
model = BiLSTM(input_size, hidden_size, output_size, num_layers)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

### 直接训练，不加载模型
if not model_load:
    start_epoch = 0  # 从头训练, 继续训练 num_epochs 轮
    train(model, train_loader, epochs=start_epoch + num_epochs, start_epoch=start_epoch, device=device, optimizer=optimizer, criterion=criterion, checkpoint_dir=checkpoint_dir)  # 继续训练

# else:  ### 加载模型
#     # 检查是否有已保存的 checkpoint
#     checkpoint_path = "best_checkpoint.pth"
#     if os.path.exists(checkpoint_path):
#         checkpoint = torch.load(checkpoint_path)
#         model.load_state_dict(checkpoint["model_state_dict"])  # 加载模型参数
#         optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # 加载优化器状态
#         start_epoch = checkpoint["epoch"]  # 继续从之前的 epoch 开始
#         print(f"✅ 已加载 checkpoint，继续训练从 Epoch {start_epoch} 开始！")
#     else:
#         print("❌ 未找到 checkpoint，重新开始训练！")
#         start_epoch = 0  # 从头训练

#     print(f"Model parameters are on device: {next(model.parameters()).device}")

    
    # 检查是否有已保存的 checkpoint
    # checkpoint_path = "best_checkpoint.pth"

    # if os.path.exists(checkpoint_path):
    #     checkpoint = torch.load(checkpoint_path)
    #     # 确保模型在 GPU
    #     model.load_state_dict(checkpoint["model_state_dict"])  # 加载模型参数
    #     model.to(device)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # 加载优化器状态
    #     # **手动确保优化器的状态在 GPU**
    #     for param_group in optimizer.param_groups:
    #         for param in param_group["params"]:
    #             if param is not None:
    #                 param.data = param.data.to(device)
    #                 if param.grad is not None:
    #                     param.grad.data = param.grad.data.to(device)
    #     start_epoch = checkpoint["epoch"]  # 继续从之前的 epoch 开始
    #     print(f"✅ 已加载 checkpoint，继续训练从 Epoch {start_epoch} 开始！")
        
    # else:
    #     print("❌ 未找到 checkpoint，重新开始训练！")
    #     start_epoch = 0  # 从头训练









