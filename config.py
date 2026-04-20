import os
import matplotlib.pyplot as plt

date = "1021"
platform = "local"  # "local" or "server"

# 路径配置
if platform == "server":
    checkpoint_dir = f"/mnt/checkpoints/{date}"
    plot_save_dir = f"/mnt/plots/{date}"
    results_save_dir = f"/mnt/results/{date}"
else:
    checkpoint_dir = f"checkpoints/{date}" 
    plot_save_dir = "plots"
    results_save_dir = 'results'

# 自动创建目录
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(plot_save_dir, exist_ok=True)
os.makedirs(results_save_dir, exist_ok=True)

# 模型参数
model_load = False
features = ['VRTG', 'VRTG_diff1', 'VRTG_diff2', 'VRTG_diff_median', 'VRTG_median', 'VRTG_std', 'VRTG_diff_10']
window_size = 100           # 前后各 window_size 个，共 2 * window_size 个数据点
rolling_window_size = 96    # 滑动窗口前后共 rolling_window_size 【中位数，96,bs:256】

num_epochs = 5

input_size = 7                 # 特征维度, X_train.shape[1] 
batch_size = 256
hidden_size = 256              # 隐藏层大小
output_size = 2                # 分类任务有两个输出：normal 和 abnormal
num_layers = 3

# 绘图参数
plt.rcParams['figure.dpi'] = 150