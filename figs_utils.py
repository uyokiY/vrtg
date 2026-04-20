import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedLocator
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

def plot_detection_results_pkl(red_data_path, min_index=0, max_index=200000):
    """
    绘制突变点检测结果
    :param red_data_path: 原始数据路径
    :param min_index: 绘制的最小索引
    :param max_index: 绘制的最大索引
    """
    df = pd.read_csv(red_data_path, usecols=['status', 'VRTG']) # 读取原始数据

    df = df[pd.to_numeric(df['VRTG'], errors='coerce').notnull()]
    df['VRTG'] = pd.to_numeric(df['VRTG'])
    df.reset_index(drop=True, inplace=True)
    df['index'] = df.index  # 记录在该航段中的原始索引

    df['status'] = df['status'].apply(lambda x: 0 if x == 'normal' else 1) # 这里的1可能是attention

    fig, ax = plt.subplots(figsize=(4, 3))
    
    # 绘制 y = 0.3, 1, 1.8 的超限参考线
    ax.axhline(1.8, color='gray', linestyle='-.', linewidth=1)
    ax.axhline(1, color='gray', linestyle='-',  linewidth=1)
    ax.axhline(0.3, color='gray', linestyle='-.', linewidth=1)

    df = df[min_index: max_index]
    
    # Plot true normal points (green)
    true_normal = df[df['status'] == 0]
    plt.scatter(true_normal.index, true_normal['VRTG'], color='green', s=0.5, label='Normal')

    # Plot true abnormal points (blue)
    true_abnormal = df[df['status'] == 1]
    plt.scatter(true_abnormal.index, true_abnormal['VRTG'], color='blue', marker='x', s=3, label='Attention')

    plt.tick_params(axis='both', which='major', labelsize=7)  # 调整刻度字号

    plt.title(f"{red_data_path}")
    plt.xlabel("Time Steps")
    plt.ylabel("VRTG Values")
    # plt.legend(fontsize=7)
    plt.show()




def load_v_data(data_folder, sp_type='vshape'):
    df_info = pd.read_excel('问题类型记录.xlsx')
    df_info['csv_name'] = df_info['csv_name'].apply(lambda x: str(x) + '.csv' if not str(x).endswith('.csv') else str(x))
    flight_files = list(zip(df_info['flight_id'], df_info['csv_name']))
    target_flight_ids = df_info[df_info['type'] == sp_type]['flight_id'].tolist()
    print("target_flight_ids: ", target_flight_ids)

    all_data = []
    for flight_id, filename in flight_files:
        if flight_id not in target_flight_ids:
            continue
        file_path = os.path.join(data_folder, filename)
        print(f"Loading flight_id: {flight_id}, filename: {filename}")
        df = pd.read_csv(file_path, usecols=['status', 'VRTG'])
        df['flight_id'] = flight_id
        df['index'] = df.index  # 记录在该航段中的原始索引，从 0 开始
        all_data.append(df)

    data = pd.concat(all_data, ignore_index=True)
    print(f"Train size: {len(data)}")
    return data

# 计算斜率、拐点、深度等特征
def extract_v_shape_params(sample):
    n = len(sample)
    p = np.argmin(sample)  # 最低点位置
    left_slope = (sample[p] - sample[0]) / p
    right_slope = (sample[-1] - sample[p]) / (n - p - 1)
    dip = sample[0] - sample[p]  # 深度
    return {
        'length': n,
        'p': p,
        'left_slope': left_slope,
        'right_slope': right_slope,
        'dip': dip
    }

def generate_v_shape(length=60, dip=1.2, noise_level=0.05, asymmetry=0.5):
    # 拐点位置
    p = int(length * asymmetry)
    left = np.linspace(0, -dip, p)
    right = np.linspace(-dip, 0, length - p)
    v_shape = np.concatenate([left, right])
    # 加噪
    noise = np.random.normal(0, noise_level, size=length)
    return v_shape + noise


# data = load_v_data('data')
# data_v1 = data[data['flight_id'] == 132]
# data_v2 = data[data['flight_id'] == 133]
# sample_v1 = data_v1.iloc[5700:5800]
# sample_v2 = data_v2.iloc[6600:6900]


def plot_detection_results(red_data_path, min_index=0, max_index=200000):
    """
    绘制突变点检测结果
    :param red_data_path: 原始数据路径
    :param min_index: 绘制的最小索引
    :param max_index: 绘制的最大索引
    """
    df = pd.read_csv(red_data_path, usecols=['status', 'VRTG']) # 读取原始数据

    df = df[pd.to_numeric(df['VRTG'], errors='coerce').notnull()]
    df['VRTG'] = pd.to_numeric(df['VRTG'])
    df.reset_index(drop=True, inplace=True)
    df['index'] = df.index  # 记录在该航段中的原始索引

    df['status'] = df['status'].apply(lambda x: 0 if x == 'normal' else 1) # 这里的1可能是attention

    fig, ax = plt.subplots(figsize=(10, 3))
    
    # 绘制 y = 0.3, 1, 1.8 的超限参考线
    ax.axhline(1.8, color='gray', linestyle='-.', linewidth=1)
    ax.axhline(1, color='gray', linestyle='-',  linewidth=1)
    ax.axhline(0.3, color='gray', linestyle='-.', linewidth=1)

    df = df[min_index: max_index]
    
    # Plot true normal points (green)
    true_normal = df[df['status'] == 0]
    plt.scatter(true_normal.index, true_normal['VRTG'], color='green', s=0.5, label='Normal')

    # Plot true abnormal points (blue)
    true_abnormal = df[df['status'] == 1]
    plt.scatter(true_abnormal.index, true_abnormal['VRTG'], color='blue', marker='x', s=3, label='Attention')

    plt.tick_params(axis='both', which='major', labelsize=7)  # 调整刻度字号

    plt.title(f"{red_data_path}")
    plt.xlabel("Time Steps")
    plt.ylabel("VRTG Values")
    # plt.legend(fontsize=7)
    plt.show()