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
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler


DEFAULT_METADATA_PATH = "问题类型记录.xlsx"
VALID_EXPERIMENT_TYPES = ["whole", "jump", "red", "jolt"]


def normalize_csv_name(value):
    value = str(value)
    return value if value.endswith(".csv") else f"{value}.csv"


def load_experiment_metadata(metadata_path=DEFAULT_METADATA_PATH):
    metadata = pd.read_excel(metadata_path).copy()
    metadata["csv_name"] = metadata["csv_name"].apply(normalize_csv_name)
    return metadata[metadata["type"].isin(VALID_EXPERIMENT_TYPES)].copy()


def read_flight_rows(data_folder, metadata, flight_ids):
    selected = metadata[metadata["flight_id"].isin(flight_ids)].copy()
    selected["flight_id"] = pd.Categorical(
        selected["flight_id"],
        categories=list(flight_ids),
        ordered=True,
    )
    selected = selected.sort_values("flight_id")

    all_data = []
    for _, row in selected.iterrows():
        flight_id = int(row["flight_id"])
        filename = row["csv_name"]
        file_path = os.path.join(data_folder, filename)
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping...")
            continue

        print(f"Loading flight_id: {flight_id}, filename: {filename}")
        flight_data = pd.read_csv(file_path, usecols=["status", "VRTG"])
        flight_data["flight_id"] = flight_id
        flight_data["index"] = flight_data.index
        flight_data["csv_name"] = filename
        flight_data["flight_type"] = row["type"]
        flight_data["split_seq"] = row["split_seq"]
        all_data.append(flight_data)

    if not all_data:
        return pd.DataFrame(
            columns=["status", "VRTG", "flight_id", "index", "csv_name", "flight_type", "split_seq"]
        )
    return pd.concat(all_data, ignore_index=True)


def load_bilstm_data(
    data_folder,
    metadata_path=DEFAULT_METADATA_PATH,
    train_sample_ids=None,
    val_sample_ids=None,
):
    """
    Load the module-level BiLSTM split:
    train/val focus on whole-vs-representative-normal learning, while test is
    the hold-out multi-type split for final BiLSTM-only evaluation.
    """
    metadata = load_experiment_metadata(metadata_path)
    train_sample_ids = [] if train_sample_ids is None else list(train_sample_ids)
    val_sample_ids = [] if val_sample_ids is None else list(val_sample_ids)

    train_whole_ids = metadata[
        (metadata["type"] == "whole") & (metadata["split_seq"] == "train3")
    ]["flight_id"].tolist()
    train_normal_ids = metadata[
        (metadata["status"] == "normal") & (metadata["flight_id"].isin(train_sample_ids))
    ]["flight_id"].tolist()
    val_whole_ids = metadata[
        (metadata["type"] == "whole") & (metadata["split_seq"] == "val3")
    ]["flight_id"].tolist()
    val_normal_ids = metadata[
        (metadata["status"] == "normal") & (metadata["flight_id"].isin(val_sample_ids))
    ]["flight_id"].tolist()
    test_ids = metadata[metadata["split_seq"] == "test"]["flight_id"].tolist()

    train_ids = list(dict.fromkeys(train_whole_ids + train_normal_ids))
    val_ids = list(dict.fromkeys(val_whole_ids + val_normal_ids))

    print("BiLSTM train_flight_ids:", train_ids, "length:", len(train_ids))
    print("BiLSTM val_flight_ids:", val_ids, "length:", len(val_ids))
    print("Hold-out test_flight_ids:", test_ids, "length:", len(test_ids))

    train_data = read_flight_rows(data_folder, metadata, train_ids)
    val_data = read_flight_rows(data_folder, metadata, val_ids)
    test_data = read_flight_rows(data_folder, metadata, test_ids)
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}, Test size: {len(test_data)}")
    return train_data, val_data, test_data


def build_split_summary(split_frames):
    rows = []
    for split_name, data in split_frames.items():
        if data.empty:
            continue
        group_cols = ["flight_id", "csv_name", "flight_type", "split_seq"]
        for keys, flight_data in data.groupby(group_cols, sort=False):
            flight_id, csv_name, flight_type, split_seq = keys
            abnormal_points = int((flight_data["status"] == "abnormal").sum())
            normal_points = int((flight_data["status"] == "normal").sum())
            total_points = int(len(flight_data))
            rows.append({
                "split": split_name,
                "flight_id": int(flight_id),
                "csv_name": csv_name,
                "flight_type": flight_type,
                "split_seq": split_seq,
                "normal_points": normal_points,
                "abnormal_points": abnormal_points,
                "total_points": total_points,
                "abnormal_ratio": abnormal_points / total_points if total_points else 0.0,
            })
    return pd.DataFrame(rows)

def load_data(data_folder, train_x, val_x):
    df = pd.read_excel('问题类型记录.xlsx')
    df['csv_name'] = df['csv_name'].apply(lambda x: str(x) + '.csv' if not str(x).endswith('.csv') else str(x))
    flight_files = list(zip(df['flight_id'], df['csv_name']))

    # optional: 采样一部分正样本
    full_train_x = df[df['split_seq'] == train_x]['flight_id'].tolist()
    sample_n = 10
    sampled_train = df[df['split_seq'] == 'train'].sample(n=sample_n, random_state=42)['flight_id'].tolist()
    train_flight_ids = full_train_x + sampled_train
    # 如果不采样：
    # train_condition = (df['split_seq'] == 'train') | (df['split_seq'] == train_x)
    # train_flight_ids = df[train_condition]['flight_id'].tolist()
    test_condition = (df['split_seq'] == 'val') | (df['split_seq'] == val_x)
    test_flight_ids = df[test_condition]['flight_id'].tolist()
    print("train_flight_ids: ", train_flight_ids)
    print("test_flight_ids:", test_flight_ids)

    all_data = []
    for flight_id, filename in flight_files:
        file_path = os.path.join(data_folder, filename)
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping...")
            continue
        print(f"Loading flight_id: {flight_id}, filename: {filename}")
        df = pd.read_csv(file_path, usecols=['status', 'VRTG'])
        df['flight_id'] = flight_id
        df['index'] = df.index  # 记录在该航段中的原始索引，从 0 开始
        all_data.append(df)
    data = pd.concat(all_data, ignore_index=True)

    train_data = data[data['flight_id'].isin(train_flight_ids)]
    test_data = data[data['flight_id'].isin(test_flight_ids)]
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

    return train_data, test_data


def load_whole_data(data_folder, train_x='train3', sample_ids=[29,30,32,40,46,47,51,62,79,92]):  # [29,30,32,40,46,47,51,62,79,92]
    df = pd.read_excel('问题类型记录.xlsx')
    df['csv_name'] = df['csv_name'].apply(lambda x: str(x) + '.csv' if not str(x).endswith('.csv') else str(x))
    flight_files = list(zip(df['flight_id'], df['csv_name']))

    # 采样训练集：
    train_condition = (df['flight_id'].isin(sample_ids)) | (df['split_seq'] == train_x)
    train_flight_ids = df[train_condition]['flight_id'].tolist()
    # test_flight_ids = [id for id in df['flight_id'].to_list() if id not in train_flight_ids]
    test_flight_ids = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
                       19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 
                       31, 33, 34, 35, 36, 37, 38, 39, 58, 59, 
                       60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 
                       71, 72, 73, 74, 75, 76, 77, 78, 80, 81, 
                       82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 
                       93, 94, 95, 96, 97, 113, 114, 115, 116, 
                       117, 118, 119, 120, 121, 122, 123, 124, 
                       125, 126, 127, 128, 129, 130, 131, 132, 133]
    print("train_flight_ids: ", train_flight_ids, "length: ", len(train_flight_ids))
    print("test_flight_ids:", test_flight_ids, "length: ", len(test_flight_ids))

    all_data = []
    for flight_id, filename in flight_files:
        file_path = os.path.join(data_folder, filename)
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping...")
            continue
        print(f"Loading flight_id: {flight_id}, filename: {filename}")
        df = pd.read_csv(file_path, usecols=['status', 'VRTG'])
        df['flight_id'] = flight_id
        df['index'] = df.index  # 记录在该航段中的原始索引，从 0 开始
        all_data.append(df)
    data = pd.concat(all_data, ignore_index=True)

    train_data = data[data['flight_id'].isin(train_flight_ids)]
    test_data = data[data['flight_id'].isin(test_flight_ids)]
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

    return train_data, test_data


def load_VAE_data(data_folder):
    df = pd.read_excel('问题类型记录.xlsx')
    df['csv_name'] = df['csv_name'].apply(lambda x: str(x) + '.csv' if not str(x).endswith('.csv') else str(x))
    flight_files = list(zip(df['flight_id'], df['csv_name']))

    train_flight_ids = [id for id in range(40, 58)]  # 保持对比，用了18个航段训练
    val_flight_ids = [id for id in range(98, 113)]   # 验证集用15个航段
    base_test_ids =  [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 
                      23, 24, 25, 26, 27, 28, 31, 33, 34, 35, 36, 37, 38, 39, 
                      41, 42, 43, 44, 45, 48, 49, 50, 52, 53, 54, 55, 56, 57, 
                      58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 
                      73, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 
                      88, 89, 90, 91, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 
                      103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 
                      115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 
                      127, 128, 129, 130, 131, 132, 133]
    # 测试集比base用的实际更小
    test_flight_ids = [id for id in df['flight_id'].tolist() if id not in train_flight_ids + val_flight_ids and id in base_test_ids]
    print("train_flight_ids: ", train_flight_ids, "length: ", len(train_flight_ids))
    print("val_flight_ids:", val_flight_ids, "length: ", len(val_flight_ids))
    print("test_flight_ids:", test_flight_ids, "length: ", len(test_flight_ids))

    all_data = []
    for flight_id, filename in flight_files:
        file_path = os.path.join(data_folder, filename)
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping...")
            continue
        df = pd.read_csv(file_path, usecols=['status', 'VRTG'])
        df['flight_id'] = flight_id
        df['index'] = df.index  # 记录在该航段中的原始索引，从 0 开始
        all_data.append(df)
    data = pd.concat(all_data, ignore_index=True)

    train_data = data[(data['flight_id'].isin(train_flight_ids)) & (data['status'] == 'normal')]
    val_data = data[(data['flight_id'].isin(val_flight_ids)) & (data['status'] == 'normal')]
    test_data = data[data['flight_id'].isin(test_flight_ids)]
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}, Test size: {len(test_data)}")

    return train_data, val_data, test_data



def compute_features(df, rolling_window_size):
    processed_data = []
    flight_ids = df['flight_id'].unique()  # 通过航段 ID 分组，逐个计算特征
    for flight_id in flight_ids:  # 避免跨航段
        flight_data = df[df['flight_id'] == flight_id].copy()
        # 一阶、二阶差分
        flight_data['VRTG_diff1'] = flight_data['VRTG'].diff().fillna(0)
        # flight_data['VRTG_diff2'] = flight_data['VRTG'].diff(2).fillna(0)
        # 多步差分
        flight_data['VRTG_diff_10'] = flight_data['VRTG'] - flight_data['VRTG'].shift(10).fillna(0)        

        # # rolling_window_size 大小的滑动统计特征
        # flight_data['VRTG_median'] = flight_data['VRTG'].rolling(window=rolling_window_size, min_periods=1).median().fillna(0)
        # flight_data['VRTG_std'] = flight_data['VRTG'].rolling(window=rolling_window_size, min_periods=1).std().fillna(0)
        # # 当前值与滑动中位数的差值
        # flight_data['VRTG_diff_median'] = flight_data['VRTG'] - flight_data['VRTG_median']

        # 计算稳健z分数
        # 1. 滚动中位数 (代替均值)
        median = flight_data['VRTG'].rolling(window=rolling_window_size, min_periods=1).median()

        # 2. 计算 MAD (中位数绝对偏差)
        # 先算绝对偏差，再滚动求中位数
        abs_dev = (flight_data['VRTG'] - median).abs()
        mad = abs_dev.rolling(window=rolling_window_size, min_periods=1).median()

        # 3. 计算修正 Z-score
        eps = 1e-6
        # 这里的 0.6745 是为了让 MAD 估计的值与标准差一致 (一致性常数)
        flight_data['VRTG_robust_z'] = 0.6745 * (flight_data['VRTG'] - median) / (mad + eps)
        processed_data.append(flight_data)
        
    return pd.concat(processed_data, ignore_index=True)


def compute_features_v2(df, rolling_window_size):
    processed_data = []
    flight_ids = df['flight_id'].unique()  # 通过航段 ID 分组，逐个计算特征

    for flight_id in flight_ids:  # 避免跨航段
        flight_data = df[df['flight_id'] == flight_id].copy()

        # 1. 一阶/n阶差分（ΔVRTG_t）
        flight_data['VRTG_diff_1'] = flight_data['VRTG'].diff().fillna(0)
        flight_data['VRTG_diff_10'] = (flight_data['VRTG'] - flight_data['VRTG'].shift(10).fillna(0))

        # 2. 滚动均值 & std，用来算 z-score
        roll = flight_data['VRTG'].rolling(
            window=rolling_window_size,
            min_periods=1
        )
        mu = roll.mean()
        sigma = roll.std().fillna(0)

        eps = 1e-6
        flight_data['VRTG_z'] = (flight_data['VRTG'] - mu) / (sigma + eps)

        ## 计算稳健z分数
        # 1. 滚动中位数 (代替均值)
        median = flight_data['VRTG'].rolling(window=rolling_window_size, min_periods=1).median()

        # 2. 计算 MAD (中位数绝对偏差)
        # 先算绝对偏差，再滚动求中位数
        abs_dev = (flight_data['VRTG'] - median).abs()
        mad = abs_dev.rolling(window=rolling_window_size, min_periods=1).median()

        # 3. 计算修正 Z-score
        # 注意：MAD 需要乘以一个常数 0.6745 才能在正态分布下等同于标准差，
        # 但如果不做概率推断仅做异常检测，直接用 MAD 也可以。
        eps = 1e-6
        # 这里的 0.6745 是为了让 MAD 估计的值与标准差一致 (一致性常数)
        flight_data['VRTG_robust_z'] = 0.6745 * (flight_data['VRTG'] - median) / (mad + eps)


        processed_data.append(flight_data)



    return pd.concat(processed_data, ignore_index=True)



class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, flight_ids, indices, window_size):
        """
        :param X: 特征数据 (numpy array 或 tensor)
        :param y: 标签数据 (numpy array 或 tensor)
        :param flight_ids: 航段 ID
        :param indices: 在航段中的索引
        :param window_size: 窗口大小 (前后各 50 条，总共 101 条)
        """
        self.X = torch.tensor(X, dtype=torch.float32) if not isinstance(X, torch.Tensor) else X
        self.y = torch.tensor(y, dtype=torch.long) if not isinstance(y, torch.Tensor) else y
        self.flight_ids = flight_ids
        self.indices = indices
        self.window_size = window_size

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        flight_id = self.flight_ids[idx]
        index = self.indices[idx]
        # 计算窗口范围，但不能跨越航段
        start = idx
        while (
            start > 0
            and self.flight_ids[start - 1] == flight_id
            and (idx - (start - 1)) <= self.window_size
        ):
            start -= 1
        end = idx
        while (
            end < len(self.X) - 1
            and self.flight_ids[end + 1] == flight_id
            and ((end + 1) - idx) <= self.window_size
        ):
            end += 1
        window_data = self.X[start:end + 1]

        # 如果数据长度不足 101，则填充 0 ???填充1更好？那其他特征怎么办？是否需要归一化？？⭕️
        pad_left = max(0, self.window_size - (idx - start))
        pad_right = max(0, self.window_size - (end - idx))
        if pad_left > 0:
            pad_tensor_left = torch.zeros((pad_left, self.X.shape[1]))
            window_data = torch.cat([pad_tensor_left, window_data], dim=0)
        if pad_right > 0:
            pad_tensor_right = torch.zeros((pad_right, self.X.shape[1]))
            window_data = torch.cat([window_data, pad_tensor_right], dim=0)
        
        return window_data, self.y[idx], flight_id, index

    def __repr__(self):
        return (f"<TimeSeriesDataset>\n"
                f"  Total samples       : {len(self)}\n"
                f"  Input shape         : ({self.window_size*2+1}, {self.X.shape[1]})\n"
                f"  Window size         : {self.window_size} (Total len: {self.window_size*2+1})\n"
                )


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc  = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):                      # x: [B, T, F]
        out, _ = self.lstm(x)                  # [B, T, 2H]
        # 方案A：取中心时刻（与“中心点打标”对齐）
        t0 = x.size(1) // 2
        feat = out[:, t0, :]                  # [B, 2H]
        # 方案B：或改为时间平均（更鲁棒，避免 padding 影响）
        # feat = out.mean(dim=1)              # [B, 2H]
        return self.fc(feat)



@torch.no_grad()
def compute_confusion_on_loader(model, loader, device):
    model.eval()  # 关键：关闭dropout，固定BN
    all_preds, all_labels = [], []
    for inputs, labels, _, _ in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return confusion_matrix(all_labels, all_preds)


@torch.no_grad()
def evaluate_loader_metrics(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    for inputs, labels, _, _ in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        predicted = outputs.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
        all_preds.append(predicted.detach().cpu())
        all_labels.append(labels.detach().cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return {
        "loss": total_loss / total_samples,
        "accuracy": float((all_preds == all_labels).mean()),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "confusion_matrix": confusion_matrix(all_labels, all_preds, labels=[0, 1]),
    }


def train(model, train_loader, epochs, start_epoch, device, optimizer, criterion, checkpoint_dir, val_loader=None):
    model.to(device)
    best_score = -1.0
    history = []

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        correct_preds = 0
        total_preds = 0

        all_preds = []
        all_labels = []

        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{epochs}]', leave=False)

        for inputs, labels, flight_ids, indices in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)

            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

            # 记录用于混淆矩阵的值
            all_preds.append(predicted.detach().cpu())
            all_labels.append(labels.detach().cpu())

            train_loader_tqdm.set_postfix(loss=total_loss/total_preds, accuracy=correct_preds/total_preds)

        train_loss = total_loss / total_preds
        train_acc = correct_preds / total_preds
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        train_precision = precision_score(all_labels, all_preds, zero_division=0)
        train_recall = recall_score(all_labels, all_preds, zero_division=0)
        train_f1 = f1_score(all_labels, all_preds, zero_division=0)
        print(
            f"Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}, "
            f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}"
        )

        cm_train = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        print(f"[Epoch {epoch+1}] Train CM (eval mode, final weights):\n{cm_train}")

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_precision": train_precision,
            "train_recall": train_recall,
            "train_f1": train_f1,
        }

        if val_loader is not None:
            val_metrics = evaluate_loader_metrics(model, val_loader, criterion, device)
            print(
                f"[Epoch {epoch+1}] Val Loss: {val_metrics['loss']:.4f}, "
                f"Accuracy: {val_metrics['accuracy']:.4f}, Precision: {val_metrics['precision']:.4f}, "
                f"Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}"
            )
            print(f"[Epoch {epoch+1}] Val CM:\n{val_metrics['confusion_matrix']}")
            row.update({
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
            })
            score = val_metrics["f1"]
        else:
            score = train_f1

        history.append(row)
        if score >= best_score:
            best_score = score
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_score": best_score,
                "best_score_name": "val_f1" if val_loader is not None else "train_f1",
            }
            torch.save(checkpoint, f"{checkpoint_dir}/best_checkpoint.pth")
            print(f"Best checkpoint saved at epoch {epoch+1} ({checkpoint['best_score_name']}={best_score:.4f})")

    return history



def evaluate_on_train(model, train_loader, device):
    """ 在整个 train_loader 上计算混淆矩阵 """
    model.eval()  # 评估模式
    all_labels = []
    all_preds = []

    with torch.no_grad():
        train_loader_tqdm = tqdm(train_loader, desc='Evaluate on Train set', leave=False)
        for inputs, labels, flight_ids, indices in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)  # 取最大概率类别
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", conf_matrix)


def plot_confusion_matrix(conf_matrix, class_names):
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()


def evaluate_save(model, val_loader, class_names=["Normal", "Abnormal"], device='cuda'):
    model.eval()  # 设置为评估模式
    model.to(device)
    correct_preds = 0
    total_preds = 0
    all_labels = []
    all_preds = []
    all_probs = []  # 用于存储预测概率（AUC需要）
    misclassified_samples = []  # 用于存储错误分类的样本信息

    with torch.no_grad():

        # 使用tqdm显示进度条
        val_loader_tqdm = tqdm(val_loader, leave=False)
        for inputs, labels, flight_ids, indices  in val_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到GPU
            outputs = model(inputs)
            
            # 获取预测结果
            _, predicted = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)  # 计算概率（适用于多分类）

            # 统计正确预测的样本数
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

            # 收集所有标签和预测结果
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())  # 存储概率
            
            # 找到预测错误的索引
            incorrect_indices = (predicted != labels).cpu().numpy()

            # 记录错误的 flight_id 和 index
            for i, incorrect in enumerate(incorrect_indices):
                if incorrect:
                    misclassified_samples.append((flight_ids[i].item(), indices[i].item(), labels[i].item(), predicted[i].item()))

    # 计算准确率
    accuracy = correct_preds / total_preds
    print(f"Accuracy: {accuracy:.4f}")

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(conf_matrix)

    # 绘制混淆矩阵热力图
    plot_confusion_matrix(conf_matrix, class_names)

    # 计算precision, recall, F1
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 计算AUC（仅适用于二分类）
    if len(np.unique(all_labels)) == 2:  # 确保是二分类
        auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])  # 取正类的概率
        print(f"AUC: {auc:.4f}")
    else:  # 多分类情况
        print("AUC is not calculated for multi-class classification.")

    # 打印预测错误的样本信息
    print("Misclassified Samples (flight_id, index, true_label, predicted_label):")
    for sample in misclassified_samples[:10]:  # 只打印前 10 个错误样本
        print(sample)

    return accuracy, misclassified_samples


def plot_detection_results(red_data_path, pred_save_path, min_index, max_index):
    """
    绘制突变点检测结果
    :param red_data_path: 原始数据路径
    :param pred_save_path: 预测结果保存路径
    :param min_index: 绘制的最小索引
    :param max_index: 绘制的最大索引
    """
    pred_df = pd.read_csv(pred_save_path) # 读取预测结果
    new_data = pd.read_csv(red_data_path, usecols=['status', 'VRTG']) # 读取原始数据

    new_data = new_data[pd.to_numeric(new_data['VRTG'], errors='coerce').notnull()]
    new_data['VRTG'] = pd.to_numeric(new_data['VRTG'])
    new_data.reset_index(drop=True, inplace=True)
    new_data['index'] = new_data.index  # 记录在该航段中的原始索引

    new_data['status'] = new_data['status'].apply(lambda x: 0 if x == 'normal' else 1) # 这里的1可能是attention

    # 合并原始数据和预测结果
    df = new_data.merge(pred_df, on=['index'])

    # 创建画布
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 绘制 y = 0.3, 1, 1.8 的超限参考线
    ax.axhline(1.8, color='gray', linestyle='-.', linewidth=1, label='y = 1.8')
    ax.axhline(1, color='gray', linestyle='-',  linewidth=1, label='y = 1')
    ax.axhline(0.3, color='gray', linestyle='-.', linewidth=1, label='y = 0.3')

    df = df[min_index: max_index]
    
    # Plot true normal points (green)
    true_normal = df[df['status'] == 0]
    plt.scatter(true_normal.index, true_normal['VRTG'], color='green', s=0.5, label='Normal')

    # Plot true abnormal points (blue)
    true_abnormal = df[df['status'] == 1]
    plt.scatter(true_abnormal.index, true_abnormal['VRTG'], color='blue', marker='x', s=3, label='Attention')

    # Plot predicted abnormal points (red)
    pred_abnormal = df[df['predicted_status'] == 1]
    plt.scatter(pred_abnormal.index, pred_abnormal['VRTG'], color='red', marker='o', s=1, label='Predicted Abnormal')

    # 设置密集的 x 轴刻度
    locator = MultipleLocator(5000)  # 每隔 5000 单位设置一个主刻度
    ax.xaxis.set_major_locator(locator)

    # 确保 y 轴包含 1 并手动设置刻度
    y_ticks = list(ax.yaxis.get_major_locator()())  # 获取默认的 y 轴刻度
    if 1 not in y_ticks:  # 如果 1 不在默认刻度中，添加它
        y_ticks.append(1)
        y_ticks = sorted(y_ticks)  # 保持刻度的顺序
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))

    # 设置标题和标签
    plt.title(f"{red_data_path}")
    plt.xlabel("Time Steps")
    plt.ylabel("VRTG Values")
    plt.legend()
    plt.show()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        p_t = torch.exp(-ce_loss)  # 计算正确分类的概率
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss  # 增强难分类样本的损失
        return focal_loss.mean()
