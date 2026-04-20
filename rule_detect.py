import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedLocator
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset
import torch
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


class ManyToManyOverlappingDataset(Dataset):
    def __init__(self, X, y, flight_ids, indices, window_size, stride=1):
        self.X = torch.tensor(X, dtype=torch.float32) if not isinstance(X, torch.Tensor) else X
        self.y = torch.tensor(y, dtype=torch.long) if not isinstance(y, torch.Tensor) else y
        self.flight_ids = flight_ids
        self.indices = indices
        self.window_size = window_size
        self.seq_len = 2 * window_size + 1
        self.stride = stride

        self.valid_indices = []
        i = 0
        while i < len(self.X):
            fid = self.flight_ids[i]
            start = i - window_size
            end = i + window_size
            if start >= 0 and end < len(self.X):
                if all(self.flight_ids[j] == fid for j in range(start, end + 1)):
                    self.valid_indices.append(i)
                    i += stride
                    continue
            i += 1

        self.num_flights = len(np.unique(self.flight_ids))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, i):
        center_idx = self.valid_indices[i]
        start = center_idx - self.window_size
        end = center_idx + self.window_size

        window_x = self.X[start:end + 1]
        window_y = self.y[start:end + 1]
        flight_id = self.flight_ids[center_idx]
        center_index = self.indices[center_idx]

        return window_x, window_y, flight_id, center_index

    def __repr__(self):
        return (f"<ManyToManyOverlappingDataset>\n"
                f"  Total samples       : {len(self)}\n"
                f"  Input shape         : ({self.seq_len}, {self.X.shape[1]})\n"
                f"  Window size         : {self.window_size} (Total len: {self.seq_len})\n"
                f"  Stride              : {self.stride}\n"
                f"  Number of flights   : {self.num_flights}")


## ====================================================================
# 预定义 flight_id 和 filename 的映射，并且前后顺序对应着 train val test 的分配
df = pd.read_excel('问题类型记录.xlsx')

# 确保 csv_name 统一以 .csv 结尾（保险处理）
df['csv_name'] = df['csv_name'].apply(lambda x: str(x) + '.csv' if not str(x).endswith('.csv') else str(x))

# 构建映射列表
flight_files = list(zip(df['flight_id'], df['csv_name']))


# 加载数据，添加 航段记号 flight_id 和 原始索引 index
def load_data(data_folder):
    all_data = []

    for flight_id, filename in flight_files:
        file_path = os.path.join(data_folder, filename)
        
        # 检查文件是否存在，避免因缺失文件导致错误
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping...")
            continue

        print(f"Loading flight_id: {flight_id}, filename: {filename}")
        df = pd.read_csv(file_path, usecols=['status', 'VRTG'])

        # 添加 flight_id 和原始索引 index
        df['flight_id'] = flight_id
        df['index'] = df.index  # 记录在该航段中的原始索引，从 0 开始

        all_data.append(df)

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()  # 避免空数据报错


# 构造特征，rolling_window_size 为滑动窗口大小
def compute_features(df, rolling_window_size): # 小窗口，记录
    # 用于存储计算完特征的所有航段数据
    processed_data = []

    # 通过航段 ID 分组，逐个计算特征
    flight_ids = df['flight_id'].unique()  # 获取所有航段 ID
    
    # 避免跨航段
    for flight_id in flight_ids:
        flight_data = df[df['flight_id'] == flight_id].copy()  # 必须用 copy() 避免修改原 DataFrame
        
        # 一阶、二阶差分
        flight_data['VRTG_diff1'] = flight_data['VRTG'].diff().fillna(0)
        flight_data['VRTG_diff2'] = flight_data['VRTG'].diff(2).fillna(0)
        
        # rolling_window_size 大小的滑动统计特征
        flight_data['VRTG_median'] = flight_data['VRTG'].rolling(window=rolling_window_size, min_periods=1).median().fillna(0)
        flight_data['VRTG_std'] = flight_data['VRTG'].rolling(window=rolling_window_size, min_periods=1).std().fillna(0)
        
        # 当前值与滑动中位数的差值
        flight_data['VRTG_diff_median'] = flight_data['VRTG'] - flight_data['VRTG_median']
        
        # 多步差分
        flight_data['VRTG_diff_10'] = flight_data['VRTG'] - flight_data['VRTG'].shift(10).fillna(0)
        
        # 将计算完的航段数据存入列表
        processed_data.append(flight_data)

    # **拼接所有航段的数据，确保原始索引结构不丢失**
    return pd.concat(processed_data, ignore_index=True)



# 分为三块：
# 1 train-val1，其中 normal 样本与2共用，abnormal 样本只用 type 为 jump 的 (跳点)
# 2 train-val2，其中 normal 样本与1共用，abnormal 样本只用 type 为 vshape 的 (v形，要模拟)
# 3 train-val3，其中 normal 样本与1共用，abnormal 样本只用 type 为 whole 的 (整段)
# 4 test 测试集（只在最后单独用于测评），包括 normal 和 abnormal 样本，
# 读取 split_seq 为 test 的数据为 4 test 数据集（包括 normal 和 abnormal）
# 读取 split_seq 为 train 的数据为 1、2、3 train 数据集（仅normal）
# 读取 split_seq 为 val 的数据为 1、2、3 val 数据集（仅normal）
# 再读取 split_seq 为 trainx 的数据为 x 的 train 数据集（仅abnormal）

## 总结：训练集x由 train、trainx 组成，验证集x由 val、valx 组成，测试集由 test 组成
## 但是 vshape 的 train 和 val 需要模拟生成

train_x = "train1"
val_x = "val1"

train_condition = (df['split_seq'] == 'train') | (df['split_seq'] == train_x)
test_condition = (df['split_seq'] == 'val') | (df['split_seq'] == val_x)

if 'split_seq' in df.columns:
    train_flight_ids = df[train_condition]['flight_id'].tolist()
    test_flight_ids = df[test_condition]['flight_id'].tolist()

    print("train_flight_ids: ", train_flight_ids)
    print("test_flight_ids:", test_flight_ids)
else:
    print("列 'split_seq' 不存在，请检查表格格式。")
# 选取特征
features = ['VRTG', 'VRTG_diff1', 'VRTG_diff2', 'VRTG_diff_median', 'VRTG_median', 'VRTG_std', 'VRTG_diff_10']

data = load_data('data')

# 根据 flight_id 过滤数据
# train_data = data[data['flight_id'].isin(train_flight_ids)]
# test_data = data[data['flight_id'].isin(test_flight_ids)]

train_data = data[data['flight_id'].isin(train_flight_ids)]
test_data = data[data['flight_id'].isin(test_flight_ids)]


print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

# 定义规则
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def is_gradual_red_event(vrtg_series, start, end, pre_window=10, normal_range=(0.8, 1.2)):
    """
    判断异常段是否为“渐变红色事件”：异常段前一段是正常的连续点
    """
    if start < pre_window:
        return False  # 起始太靠前，无法判断

    pre_vals = vrtg_series[start - pre_window:start].values
    in_normal = [(normal_range[0] <= v <= normal_range[1]) for v in pre_vals]

    # 如果前几项大部分不在正常区间内（比例 ≥ 80%），那么认为是渐变红色事件
    return sum(in_normal) / len(in_normal) <= 0.2  # 至少有 80% 的点不在正常范围内


def detect_jump_segments_filtered(vrtg, low_thresh=0.5, high_thresh=1.8, jump_thresh=0.5, min_abnormal_len=1):
    abnormal_mask = (vrtg < low_thresh) | (vrtg > high_thresh)
    segments = []
    i = 0
    while i < len(vrtg):
        if abnormal_mask[i]:
            start = i
            while i < len(vrtg) and abnormal_mask[i]:
                i += 1
            end = i - 1
            
            if end - start + 1 >= min_abnormal_len:
                pre_val = vrtg[start - 1] if start > 0 else None
                post_val = None
                for j in range(end + 1, len(vrtg)):
                    if low_thresh <= vrtg[j] <= high_thresh:
                        post_val = vrtg[j]
                        break

                jump_in = abs(vrtg[start] - pre_val) if pre_val is not None else 0
                jump_out = abs(post_val - vrtg[end]) if post_val is not None else 0
                
                if jump_in > jump_thresh and jump_out > jump_thresh:
                    if not is_gradual_red_event(vrtg, start, end):
                        segments.append((start, end))

        else:
            i += 1
    return segments

def plot_jump(segments, flight_data, min_index=0, max_index=200000):
    # 绘制时间序列数据
    flight_data = flight_data.iloc[min_index:max_index]  # 仅绘制指定范围内的数据

    vrtg_data = flight_data[['VRTG', 'status']].copy()

    # 根据 status 标记 normal 和 abnormal 点
    normal_points = vrtg_data[vrtg_data['status'] == 'normal']
    # print("normal_points: ", len(normal_points))
    abnormal_points = vrtg_data[vrtg_data['status'] != 'normal']
    # print("abnormal_points: ", abnormal_points)

    plt.figure(figsize=(5, 3))

    plt.axhline(1.8, color='gray', linestyle='-.', linewidth=1, label='y = 1.8')
    plt.axhline(1, color='gray', linestyle='-',  linewidth=1, label='y = 1')
    plt.axhline(0.3, color='gray', linestyle='-.', linewidth=1, label='y = 0.3')

    # 标记 normal 点（绿色圆点）
    plt.scatter(normal_points.index, normal_points['VRTG'], color='green', s=0.8, label='normal')
    # 标记 abnormal 点（蓝色正方形）
    plt.scatter(abnormal_points.index, abnormal_points['VRTG'], color='blue', s=2.0, label='abnormal')

    # 标记突变点（红色 'x'）
    # 确保索引在 vrtg_data 的范围内
    jump_points = []
    for start, end in segments:
        jump_points.extend(range(start, end + 1))

    jump_points = [point for point in jump_points if point >= min_index and point < max_index]
    plt.scatter(jump_points, vrtg_data.loc[jump_points, 'VRTG'], color='red', marker='x', s=0.1, label='jump')

    # 设置标题和标签
    plt.title(f"Misclassified Data")
    plt.xlabel("Time Steps")
    plt.ylabel("VRTG Values")
    # plt.legend()
    plt.show()
    plt.rcParams['figure.dpi'] = 200  # 设置图像分辨率

print("train_flight_ids: ", train_flight_ids)
print("test_flight_ids:", test_flight_ids)
train_normal_ids = [28, 29, 30, 31, 32, 33, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97]
test_normal_ids = [34, 35, 36, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]
train_jump_ids = [15, 16, 17, 18, 20, 23, 28, 29, 30, 31, 32, 33, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97]
test_jump_ids = [14, 22, 34, 35, 36, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]
file_name = flight_files_dict[flight_id]
csv = pd.read_csv(os.path.join("data", file_name))

jump_index = []
for start, end in segments:
    jump_index.extend(range(start, end + 1))

csv.loc[csv['index'].isin(jump_index), ['status', 'type']] = ['abnormal', 'jump']

# 查看是否会将 normal 样本误判为 abnormal(jump)

for flight_id in test_normal_ids: # [14, 15, 16, 17, 18, 20, 22, 23]
    print(f"========= Flight ID: {flight_id}")

    flight_files_dict = {flight_id: file_name for flight_id, file_name in flight_files}
    file_name = flight_files_dict[flight_id]
    print(file_name)

    # data = train_data_aug.copy()
    data = test_data.copy()

    flight_data = data[data['flight_id'] == flight_id].reset_index(drop=True)
    vrtg = flight_data['VRTG']

    # segments = detect_jump_segments(vrtg)
    segments = detect_jump_segments_filtered(vrtg)
    if len(segments) > 0:
        print(f"Flight ID: {flight_id}, Jump Segments: {segments}")
        plot_jump(segments, flight_data, min_index=0, max_index=5000000)

    else:
        print(f"Flight ID: {flight_id}, No Jump Segments Detected")
        plot_jump([], flight_data, min_index=0, max_index=5000000)
        # 仅绘制正常数据



data = test_data.copy()
flight_id = 35

flight_data = data[data['flight_id'] == flight_id].reset_index(drop=True)
vrtg = flight_data['VRTG']

segments = detect_jump_segments(vrtg)

plot_jump(segments, flight_data, min_index=0, max_index=5000000)

flight_id = 15



