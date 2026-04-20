import os
import pickle
import pandas as pd

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from utils import compute_features, BiLSTM, load_whole_data
from predict_utils import test_model_on_flight, aggregate_metrics

from config import checkpoint_dir, plot_save_dir, results_save_dir, features, window_size, rolling_window_size, batch_size, input_size, hidden_size, output_size, num_layers

## --------------------------- 评估模型在各个数据集上的表现 -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
model = BiLSTM(input_size, hidden_size, output_size, num_layers)
train_results = {}
test_results = {}

## --------------------------- 加载训练数据 -------------------------
train_data, test_data = load_whole_data('data')

train_data = compute_features(train_data, rolling_window_size)
X_train, y_train = train_data[features].values, (train_data['status'] == 'abnormal').astype(int).values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
flight_ids_train = train_data['flight_id'].values
indices_train = train_data['index'].values
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"训练集 异常样本比例: {sum(y_train)/len(y_train):.4f}")
train_flight_ids = train_data['flight_id'].unique()
print("训练集 flight IDs:", train_flight_ids)


## --------------------------- 加载测试数据 ---------------------------
test_data = compute_features(test_data, rolling_window_size)
X_test, y_test = test_data[features].values, (test_data['status'] == 'abnormal').astype(int).values
X_test = scaler.transform(X_test)
flight_ids_test = test_data['flight_id'].values
indices_test = test_data['index'].values
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
print(f"训练集 异常样本比例: {sum(y_test)/len(y_test):.4f}")
test_flight_ids = test_data['flight_id'].unique()
print("测试集 flight IDs:", test_flight_ids)


## --------------------------- 评估模型在测试数据上的表现 ---------------------------
for flight_id in test_flight_ids:
    print(f"Testing flight ID: {flight_id}")
    results = test_model_on_flight(model, batch_size, f"{checkpoint_dir}/best_checkpoint.pth", test_data, flight_id, features, scaler, window_size, device, plot_save_dir)
    if results:
        test_results[flight_id] = results
        print(f"Flight ID {flight_id} 测试完成")
    else:
        print(f"Flight ID {flight_id} 测试失败或数据不足")

results_save_path = os.path.join(results_save_dir, "test_results_dict.pkl")
with open(results_save_path, 'wb') as f:
    pickle.dump(test_results, f)
print(f"test_results_dict 已保存到：{results_save_path}")



## --------------------------- 评估模型在训练数据上的表现 ---------------------------
for flight_id in train_flight_ids:
    print(f"Testing flight ID: {flight_id}")
    results = test_model_on_flight(model, batch_size, f"{checkpoint_dir}/best_checkpoint.pth", train_data, flight_id, features, scaler, window_size, device, plot_save_dir)
    if results:
        train_results[flight_id] = results
        print(f"Flight ID {flight_id} 测试完成")
    else:
        print(f"Flight ID {flight_id} 测试失败或数据不足")

results_save_path = os.path.join(results_save_dir, "train_results_dict.pkl")
with open(results_save_path, 'wb') as f:
    pickle.dump(train_results, f)
print(f"train_results_dict 已保存到：{results_save_path}")



# ---------------- 汇总计算 train/test 整体表现 ----------------
train_summary = aggregate_metrics(train_results)
test_summary = aggregate_metrics(test_results)

print("\n==== Train Overall Metrics ====")
for k, v in train_summary['metrics'].items():
    print(f"{k}: {v:.4f}")
print("Confusion matrix:\n", train_summary['confusion_matrix'])

print("\n==== Test Overall Metrics ====")
for k, v in test_summary['metrics'].items():
    print(f"{k}: {v:.4f}")
print("Confusion matrix:\n", test_summary['confusion_matrix'])

pickle.dump({'train': train_summary, 'test': test_summary},
            open(os.path.join(results_save_dir, 'overall_summary.pkl'), 'wb'))
