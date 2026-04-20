import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from utils import TimeSeriesDataset

def print_confusion_info(labels_flat, predictions_flat):
    cm = confusion_matrix(labels_flat, predictions_flat)
    print("\n混淆矩阵:")
    print(cm)
    TN = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
    FP = cm[0, 1] if cm.shape[1] > 1 else 0
    FN = cm[1, 0] if cm.shape[0] > 1 else 0
    TP = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
    print(f"TN: {TN}, FP: {FP}")
    print(f"FN: {FN}, TP: {TP}")
    return cm
    

def test_single_flight(test_data, flight_id, features, scaler):
    """
    测试模型在特定flight_id上的表现
    """
    # 筛选出指定flight_id的数据
    flight_data = test_data[test_data['flight_id'] == flight_id].copy()
    
    if len(flight_data) == 0:
        print(f"Flight ID {flight_id} not found in test data")
        return None
    
    print(f"Flight {flight_id} 数据量: {len(flight_data)}")
    
    # 准备特征和标签
    X_flight = flight_data[features].values
    y_flight = (flight_data['status'] == 'abnormal').astype(int).values
    
    # 标准化
    X_flight = scaler.transform(X_flight)
    
    # 获取flight_id和索引信息
    flight_ids = flight_data['flight_id'].values
    indices = flight_data['index'].values
    
    return X_flight, y_flight, flight_ids, indices


def evaluate_flight(model, data_loader, device, flight_id):
    """
    评估模型在特定航班上的表现 (many-to-one版本)
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad(): 
        for inputs, labels, batch_flight_ids, indices in tqdm(data_loader, desc="Predicting", unit="step"):
            inputs = inputs.to(device)
            
            # 前向传播
            outputs = model(inputs)
            # 对于many-to-one，outputs是(batch_size, 2)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # 收集结果
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    predictions_flat = np.array(all_predictions)
    labels_flat = np.array(all_labels)
    probabilities_flat = np.array(all_probabilities)
    error_mask = predictions_flat != labels_flat

    accuracy = accuracy_score(labels_flat, predictions_flat)
    precision = precision_score(labels_flat, predictions_flat, zero_division=0)
    recall = recall_score(labels_flat, predictions_flat, zero_division=0)
    f1 = f1_score(labels_flat, predictions_flat, zero_division=0)
    
    print(f"-------- Flight {flight_id} 测试结果 --------")
    print(f"预测错误样本数: {sum(error_mask)}")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"总样本数: {len(labels_flat)}")
    print(f"异常样本数: {sum(labels_flat)}")
    print(f"异常样本比例: {sum(labels_flat)/len(labels_flat):.4f}")
    cm = print_confusion_info(labels_flat, predictions_flat)
    
    return {
        'predictions': predictions_flat,
        'labels': labels_flat,
        'probabilities': probabilities_flat,
        'confusion_matrix': cm,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    }


def plot_miscls_points(mismatches, X_flight, y_flight, flight_id, plot_save_dir):
    """
    绘制某航班的VRTG时间序列图，并标出模型预测错误的位置。

    参数：
    - mismatches: list[int]，预测错误的点的局部索引（针对该航班）
    - X_flight: np.ndarray，形状为 (T, D)，D 维特征，其中第一列是 VRTG
    - y_flight: np.ndarray，形状为 (T,)，对应每个时间点的真实标签
    - flight_id: str 或 int，航班标识
    - plot_save_dir: str，图像保存路径
    """

    vrtg_values = X_flight[:, 0]  # 假设第0列是 VRTG
    time_steps = np.arange(len(vrtg_values))

    # 分类
    normal_idx = np.where(y_flight == 0)[0]
    abnormal_idx = np.where(y_flight == 1)[0]


    plt.figure(figsize=(11, 3))
    plt.axhline(1.8, color='gray', linestyle='-.', linewidth=1)
    plt.axhline(1, color='gray', linestyle='-',  linewidth=1)
    plt.axhline(0.3, color='gray', linestyle='-.', linewidth=1)
    if len(mismatches) > 0:
        plt.scatter(mismatches, vrtg_values[mismatches], color='red', marker='x', s=12, label='misclassified')
    plt.scatter(normal_idx, vrtg_values[normal_idx], color='green', s=1, label='normal')
    plt.scatter(abnormal_idx, vrtg_values[abnormal_idx], color='blue', s=1, label='abnormal')


    plt.title(f"Misclassified Points - Flight {flight_id}")
    plt.xlabel("Time Step")
    plt.ylabel("VRTG")
    plt.legend(loc='upper right')

    save_path = os.path.join(plot_save_dir, f"flight_{flight_id}_misclassified.png")
    plt.savefig(save_path, dpi=300)
    print(f"图像已保存到: {save_path}")

    plt.show()


def test_model_on_flight(model, batch_size, model_path, test_data, flight_id, features, scaler, window_size, device, plot_save_dir):
    """
    单个航班测试流程
    """
    # 1. 加载模型
    print(f"** 加载模型从 {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"模型已加载，使用设备: {device}")

    # 2. 准备数据
    flight_test_data = test_single_flight(test_data, flight_id, features, scaler)
    
    if not flight_test_data:
        return None
    
    X_flight, y_flight, flight_ids, indices = flight_test_data
    
    # 3. 创建数据加载器
    flight_dataset = TimeSeriesDataset(X_flight, y_flight, flight_ids, indices, window_size)
    flight_loader = torch.utils.data.DataLoader(flight_dataset, batch_size=batch_size, shuffle=False)
    print(f"准备测试数据完成，共 {len(flight_dataset)} 个样本")

    # 4. 评估
    results = evaluate_flight(model, flight_loader, device, flight_id)

    # 可视化
    plot_miscls_points(np.where(results['predictions'] != results['labels'])[0],
                       X_flight,
                       y_flight,
                       flight_id, 
                       plot_save_dir)
    return results


def aggregate_metrics(results_dict):
    """汇总多个航段的预测结果，计算总体metrics"""
    all_preds = []
    all_labels = []
    all_probs = []

    for fid, res in results_dict.items():
        all_preds.extend(res['predictions'])
        all_labels.extend(res['labels'])
        all_probs.extend(res['probabilities'])

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    cm = confusion_matrix(all_labels, all_preds)
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0)
    }

    return {'confusion_matrix': cm, 'metrics': metrics}
