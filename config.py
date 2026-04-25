import os
import matplotlib.pyplot as plt


DEFAULT_DATE = "1021"
DEFAULT_PLATFORM = "local"  # "local" or "server"

# Keep defaults aligned with the features currently produced by utils.compute_features.
DEFAULT_FEATURES = [
    "VRTG",
    "VRTG_diff1",
    "VRTG_diff_10",
    "VRTG_robust_z",
]

DEFAULT_WINDOW_SIZE = 100
DEFAULT_ROLLING_WINDOW_SIZE = 96
DEFAULT_NUM_EPOCHS = 5
DEFAULT_BATCH_SIZE = 256
DEFAULT_HIDDEN_SIZE = 256
DEFAULT_OUTPUT_SIZE = 2
DEFAULT_NUM_LAYERS = 3
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_SAMPLE_IDS = [29, 30, 32, 40, 46, 47, 51, 62, 79, 92]
DEFAULT_VAL_SAMPLE_IDS = [34, 35, 36, 98, 99, 100]


def build_runtime_paths(date=DEFAULT_DATE, platform=DEFAULT_PLATFORM):
    if platform == "server":
        checkpoint_dir = f"/mnt/checkpoints/{date}"
        plot_save_dir = f"/mnt/plots/{date}"
        results_save_dir = f"/mnt/results/{date}"
    else:
        checkpoint_dir = f"checkpoints/{date}"
        plot_save_dir = "plots"
        results_save_dir = "results"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_save_dir, exist_ok=True)
    os.makedirs(results_save_dir, exist_ok=True)

    return checkpoint_dir, plot_save_dir, results_save_dir


# 绘图参数
plt.rcParams["figure.dpi"] = 150
