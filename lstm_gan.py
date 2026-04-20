
# -*- coding: utf-8 -*-

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict

import os, json, copy

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.optim import Adam

from tqdm.auto import tqdm

from utils import load_VAE_data, compute_features

# -------------------------------
# Utils
# -------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MinMaxScaler:
    """
    Per-feature min-max scaling to [0,1] using stats from training data.
    """
    def __init__(self):
        self.min_: Optional[torch.Tensor] = None
        self.max_: Optional[torch.Tensor] = None

    def fit(self, x: torch.Tensor):
        # x: [N, F]
        self.min_ = torch.min(x, dim=0).values
        self.max_ = torch.max(x, dim=0).values
        # Avoid zero division
        eps = 1e-8
        self.max_ = torch.where((self.max_ - self.min_) < eps, self.min_ + eps, self.max_)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.min_) / (self.max_ - self.min_)

    def inverse_transform(self, x_scaled: torch.Tensor) -> torch.Tensor:
        return x_scaled * (self.max_ - self.min_) + self.min_


# -------------------------------
# Dataset
# -------------------------------

class SlidingWindowADSBDataset(Dataset):
    """
    Slice a long multivariate time series into sliding windows.
    Each window X_in: [K, F] and target is next-step reconstruction setup.
    We use teacher-forcing reconstruction of the *shifted* sequence:
      input:   x_1 ... x_{K}
      target:  x_2 ... x_{K+1}
    If there is no x_{K+1}, we drop the last window.
    """

    def __init__(self, series: torch.Tensor, window: int):
        """
        Args:
            series: [T, F] tensor (assumed already scaled & cleaned)
            flight_ids: List of flight IDs corresponding to each time step
            window: K length
        """
        assert series.ndim == 2
        self.X = series
        self.window = window

        # number of valid windows with next-step target available
        self.n = max(0, len(self.X) - (self.window))

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # X_in : [K, F]
        x_in = self.X[idx: idx + self.window, :]
        # y_out: [K, F] — target is next-step shift of x_in
        y_out = self.X[idx + 1: idx + 1 + self.window, :]
        t_idx = torch.arange(idx + 1, idx + 1 + self.window)  # 对应 y_out 的全局时刻
        return x_in, y_out, t_idx


# ============ LSTM-GAN: 编码/解码/判别器 与封装 ============

class LSTMEncoder(nn.Module):
    """G1: X->[z]. 取最后层 hidden 拼接双向。"""
    def __init__(self, in_dim, hidden=64, layers=1, z_dim=20, bidir=True):
        super().__init__()
        self.bidir = bidir
        self.rnn = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True, bidirectional=bidir)
        out_h = hidden * (2 if bidir else 1)
        self.to_z = spectral_norm(nn.Linear(out_h, z_dim))
    def forward(self, x):                      # x: (B,T,F)
        _, (h, _) = self.rnn(x)               # h: (L*dir,B,H)
        h_last = torch.cat([h[-2], h[-1]], -1) if self.bidir else h[-1]
        return self.to_z(h_last)              # (B,z_dim)

class LSTMDecoder(nn.Module):
    def __init__(self, out_dim, hidden=64, layers=1, z_dim=20):
        super().__init__()
        self.out_dim = out_dim                              # ← 新增，保存输出/输入维
        self.init_h = spectral_norm(nn.Linear(z_dim, hidden))
        self.init_c = spectral_norm(nn.Linear(z_dim, hidden))
        self.rnn = nn.LSTM(input_size=out_dim, hidden_size=hidden,   # 保持 input_size=out_dim
                           num_layers=layers, batch_first=True)
        self.to_x = spectral_norm(nn.Linear(hidden, out_dim))

    def forward(self, z, T):
        B = z.size(0)
        h0 = torch.tanh(self.init_h(z)).unsqueeze(0)
        c0 = torch.tanh(self.init_c(z)).unsqueeze(0)
        zeros = z.new_zeros(B, T, self.out_dim)             # ← 这里从 1 改为 self.out_dim
        out, _ = self.rnn(zeros, (h0, c0))                  # (B, T, H)
        x_rec = self.to_x(out)                              # (B, T, F)
        return x_rec


class DiscriminatorX(nn.Module):
    """Dx: 时序判别（越大=越像真实）。"""
    def __init__(self, in_dim, hidden=64, bidir=True):
        super().__init__()
        self.bidir = bidir
        self.rnn = nn.LSTM(in_dim, hidden, num_layers=1, batch_first=True, bidirectional=bidir)
        out_h = hidden * (2 if bidir else 1)
        self.head = spectral_norm(nn.Linear(out_h, 1))
    def forward(self, x):
        _, (h, _) = self.rnn(x)
        h_last = torch.cat([h[-2], h[-1]], -1) if self.bidir else h[-1]
        return self.head(h_last).squeeze(-1)  # (B,)

class DiscriminatorZ(nn.Module):
    """Dz: 潜变量判别（越大=越像先验 N(0,1)）。"""
    def __init__(self, z_dim=20, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(z_dim, hidden)), nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(hidden, hidden)), nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(hidden, 1)),
        )
    def forward(self, z):
        return self.net(z).squeeze(-1)       # (B,)

class LSTMGAN(nn.Module):
    """
    对外暴露：
      - enc/dec/dx/dz
      - anomaly_score(x, alpha, calib): (B,) 分数（越大越异常）
    """
    def __init__(self, feat_dim, z_dim=20, hidden=64, layers=1, bidir=True):
        super().__init__()
        self.enc = LSTMEncoder(feat_dim, hidden, layers, z_dim, bidir)
        self.dec = LSTMDecoder(feat_dim, hidden, 1, z_dim)
        self.dx  = DiscriminatorX(feat_dim, hidden, bidir)
        self.dz  = DiscriminatorZ(z_dim, hidden)

    @torch.no_grad()
    def anomaly_score(self, x, alpha=0.5, calib=None):
        # 1) 重构误差
        z = self.enc(x)                      # (B,z)
        x_rec = self.dec(z, T=x.size(1))     # (B,T,F)
        rl = F.mse_loss(x_rec, x, reduction="none").mean(dim=(1,2))  # (B,)
        # 2) 判别器翻转并校准
        s = self.dx(x)                       # 越大越“真”
        if calib and "dx_min" in calib and "dx_max" in calib and calib["dx_max"] > calib["dx_min"]:
            dl = (calib["dx_max"] - s) / (calib["dx_max"] - calib["dx_min"])
        else:
            dl = (-s).sigmoid()              # 无校准时的平滑近似
        return alpha * rl + (1 - alpha) * dl # (B,)

# (可选) 若主程序仍写着 TransformerVAE(...)，解除侵入：
# TransformerVAE = LSTMGAN




# -------------------------------
# Training / Evaluation
# -------------------------------

@dataclass
class TrainConfig:
    window: int = 50
    features: List[str] = field(default_factory=lambda: ['VRTG', 'VRTG_diff1', 'VRTG_diff2', 'VRTG_diff_median', 'VRTG_median', 'VRTG_std', 'VRTG_diff_10'])
    batch_size: int = 64
    lr: float = 1e-3
    warmup_steps: int = 4000  # Noam-like warmup (simple linear here)
    max_epochs: int = 50
    beta_kl: float = 0.1  # KL weight (lambda in paper)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "./checkpoints"
    z_dim: int = 20
    hidden: int = 64
    layers: int = 1
    bidir: bool = True
    alpha: float = 0.5  # ADScore 融合权重
    n_critic: int = 5   # 判别器更新步
    lr_g: float = 1e-3
    lr_d: float = 1e-3
    beta_rl: float = 1.0 # 重构项权重
    train_model: bool = False
    fit_collect_errors: bool = False

class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    """
    A very simple Noam-like scheduler: lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
    Here we do not tie to d_model; just emulate warmup behavior.
    """
    def __init__(self, optimizer, warmup_steps=4000, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch + 1, 1)
        scale = (self.warmup_steps ** 0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        return [base_lr * scale for base_lr in self.base_lrs]

# ============ WGAN 损失与训练步 ============

from torch.optim import Adam

def _wgan_d_loss(real_score, fake_score):
    # max(real-fake) 等价于 min(-(real-fake))
    return -(real_score.mean() - fake_score.mean())

def _wgan_g_loss(fake_score):
    # max(fake_score) 等价于 min(-fake_score)
    return -fake_score.mean()

def train_epoch_gan(model, train_loader, cfg, device):
    """
    train/val 都是“正常”序列。这里对 train_loader 做对抗训练。
    你现有的 DataLoader/滑窗保持不变：batch -> (x_in, y_tgt, start_idx/ids/…)
    """
    model.train()
    opt_g  = Adam(list(model.enc.parameters()) + list(model.dec.parameters()),
                  lr=getattr(cfg, "lr_g", 1e-3), betas=(0.5, 0.9))
    opt_dx = Adam(model.dx.parameters(), lr=getattr(cfg, "lr_d", 1e-3), betas=(0.5, 0.9))
    opt_dz = Adam(model.dz.parameters(), lr=getattr(cfg, "lr_d", 1e-3), betas=(0.5, 0.9))
    n_critic = getattr(cfg, "n_critic", 5)
    beta_rl  = getattr(cfg, "beta_rl", 1.0)

    log = {"dx":0.0, "dz":0.0, "g":0.0, "rl":0.0}
    steps = 0

    for batch in tqdm(train_loader, desc="Training GAN epoch", unit="batch"):
        # —— 兼容：你的 batch 可能是 (x,y,id,...)，我们只取第一个是 x —— #
        x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        B, T, C = x.size()

        # 1) 判别器若干步
        for _ in range(n_critic):
            z_real = model.enc(x)
            z_fake = torch.randn_like(z_real)

            with torch.no_grad():
                x_fake = model.dec(z_fake, T=T)
            dx_real = model.dx(x)
            dx_fake = model.dx(x_fake)
            loss_dx = _wgan_d_loss(dx_real, dx_fake)
            opt_dx.zero_grad(set_to_none=True); loss_dx.backward(); opt_dx.step()

            dz_real = model.dz(z_real.detach())
            dz_fake = model.dz(z_fake)
            loss_dz = _wgan_d_loss(dz_real, dz_fake)
            opt_dz.zero_grad(set_to_none=True); loss_dz.backward(); opt_dz.step()

        # 2) 生成器一步（对抗 + 重构）
        z_real = model.enc(x)
        x_rec  = model.dec(z_real, T=T)
        z_fake = torch.randn_like(z_real)
        x_fake = model.dec(z_fake, T=T)
        g_adv  = _wgan_g_loss(model.dx(x_fake)) + _wgan_g_loss(model.dz(z_real))
        rl     = F.mse_loss(x_rec, x)
        loss_g = g_adv + beta_rl * rl

        opt_g.zero_grad(set_to_none=True); loss_g.backward()
        nn.utils.clip_grad_norm_(list(model.enc.parameters()) + list(model.dec.parameters()), 1.0)
        opt_g.step()

        # meters
        log["dx"] += float(loss_dx.detach().cpu())
        log["dz"] += float(loss_dz.detach().cpu())
        log["g"]  += float(loss_g.detach().cpu())
        log["rl"] += float(rl.detach().cpu())
        steps += 1

    for k in log: log[k] /= max(1, steps)
    return log


# ============ 校准 & 高斯拟合（val：正常） ============

@torch.no_grad()
def calibrate_dx_on_val(model, val_loader, device):
    model.eval()
    ss = []
    for batch in tqdm(val_loader, desc="Calibrating Dx", unit="step"):
        x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        ss.append(model.dx(x).detach().cpu())
    if not ss:
        return {"dx_min": 0.0, "dx_max": 1.0}
    s = torch.cat(ss, dim=0)
    return {"dx_min": float(s.min()), "dx_max": float(s.max())}

@torch.no_grad()
def fit_gaussian_on_val(model, val_loader, device, alpha=0.5, calib=None):
    """
    用 LSTM-GAN 的 AD 分数在 val(正常) 上拟合高斯，输出 (mu, sigma)。
    """
    model.eval()
    all_scores = []
    for batch in tqdm(val_loader, desc="Fitting Gaussian", unit="step"):
        x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        sc = model.anomaly_score(x, alpha=alpha, calib=calib)   # (B,)
        all_scores.append(sc.cpu())
    if not all_scores:
        return 0.0, 1.0
    s = torch.cat(all_scores, dim=0).numpy().astype("float64")
    mu, sigma = float(s.mean()), float(s.std() + 1e-8)
    return mu, sigma


def nll_scores(errors: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Negative log-likelihood for Gaussian with mean mu and std sigma.
    NLL(e) = 0.5*log(2pi*sigma^2) + (e-mu)^2 / (2 sigma^2)
    """
    var = sigma ** 2
    const = 0.5 * math.log(2 * math.pi * var)
    return const + (errors - mu) ** 2 / (2 * var)

# ============ 用 GAN 的分数替换 VAE “窗口误差” ============

@torch.no_grad()
def collect_step_errors(model, loader, T, device, alpha=0.5, calib=None):
    """
    与原函数同名同参：返回长度 T 的 time_scores 与覆盖计数 counts。
    依赖：DataLoader 的 batch 顺序与 start_idx 对应（测试阶段 shuffle=False）。
    """
    model.eval()
    import numpy as np
    scores = np.zeros((T,), dtype=np.float32)
    counts = np.zeros((T,), dtype=np.int64)

    for batch in tqdm(loader, desc="Collecting step errors", unit="step"):
        # —— 兼容不同 batch 形态 —— #
        if isinstance(batch, (list, tuple)):
            x = batch[0].to(device)
            # start_idx 可能在第 3 或第 4 个位置（视你的 Dataset 而定）
            if len(batch) >= 3 and torch.is_tensor(batch[2]):
                start_idx = batch[2]
            elif len(batch) >= 4 and torch.is_tensor(batch[3]):
                start_idx = batch[3]
            else:
                # 后备：按顺序回填（要求 loader 顺序与窗口索引对应）
                # 若你的 Dataset 明确返回起点索引，请删掉此分支，使用显式索引更稳妥
                if not hasattr(collect_step_errors, "_cursor"):
                    collect_step_errors._cursor = 0
                cur = collect_step_errors._cursor
                B, K, _ = x.size()
                start_idx = torch.arange(cur, cur + B, device=x.device)
                collect_step_errors._cursor += B
        else:
            x = batch.to(device)
            raise RuntimeError("请确保 Dataset 返回 start_idx 以便精确回填。")

        sc = model.anomaly_score(x, alpha=alpha, calib=calib).detach().cpu().numpy()  # (B,)
        if torch.is_tensor(start_idx): start_idx = start_idx.cpu().numpy().astype(int)

        K = x.size(1)
        for s, t0 in zip(sc, start_idx):
            t0 = int(t0); t1 = min(T, t0 + K)
            scores[t0:t1] += s
            counts[t0:t1] += 1

    return scores, counts



def choose_threshold_grid(scores: np.ndarray, labels: Optional[np.ndarray] = None, n_points: int = 100) -> float:
    """
    Choose threshold on anomaly *scores* (higher => more abnormal).
    If labels is provided (0=normal,1=abnormal), choose threshold maximizing F1.
    Else fallback to percentile (e.g., 99th).
    """
    if labels is None:
        return float(np.quantile(scores, 0.99))

    lo, hi = float(scores.min()), float(scores.max())
    grid = np.linspace(lo, hi, n_points)
    best_thr, best_f1 = lo, -1.0
    for t in grid:
        pred = (scores > t).astype(np.int32)
        tp = int(np.sum((pred == 1) & (labels == 1)))
        fp = int(np.sum((pred == 1) & (labels == 0)))
        fn = int(np.sum((pred == 0) & (labels == 1)))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = t
    return float(best_thr)


def confusion_and_prf(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    precision = tp / (tp + fp + 1e-12)
    recall    = tp / (tp + fn + 1e-12)
    f1        = 2 * precision * recall / (precision + recall + 1e-12)
    acc       = (tp + tn) / max(1, tp + tn + fp + fn)
    cm = np.array([[tn, fp],[fn, tp]], dtype=int)
    return {
        "cm": cm, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": acc,
        "support_pos": int(np.sum(y_true == 1)),
        "support_all": int(len(y_true))
    }



def run_eval_and_plot(
    y_test, test_flight_ids, vrtg_values,
    test_time_scores, test_counts,  # time-axis scores & coverage
    choose_threshold_grid,
    plot_save_dir,
    title_prefix="TEST"
):
    # 1) 时间轴掩码 & 阈值与预测
    labels_time = np.asarray(y_test)                  # [T]
    test_mask   = (test_counts > 0)
    scores_use  = test_time_scores[test_mask]         # 仅保留覆盖到的时刻
    labels_use  = labels_time[test_mask]

    thr = choose_threshold_grid(scores_use, labels_use, n_points=100)
    pred = (scores_use > thr).astype(np.int32)

    # 2) Overall 指标
    overall = confusion_and_prf(labels_use, pred)
    print(f"\n=== {title_prefix} Overall (time-axis, masked) ===")
    print("Confusion Matrix [[TN,FP],[FN,TP]]:\n", overall["cm"])
    print(f"P={overall['precision']:.4f} R={overall['recall']:.4f} "
          f"F1={overall['f1']:.4f} Acc={overall['accuracy']:.4f}  thr={thr:.6f}")
    print(f"TP={overall['tp']} FP={overall['fp']} FN={overall['fn']} TN={overall['tn']}  "
          f"(pos={overall['support_pos']}, N={overall['support_all']})")

    # 3) 映射回全局索引，准备分航段评估与画图
    mask_idx = np.flatnonzero(test_mask)   # 压缩→全局索引
    pred_global = np.full_like(labels_time, fill_value=-1, dtype=np.int32)
    pred_global[mask_idx] = pred

    # 4) 每个 flight 的指标 + 误分类点可视化
    unique_fids = np.unique(test_flight_ids)
    per_flight = {}

    for fid in unique_fids:
        fid_idx = np.where(test_flight_ids == fid)[0]
        covered = (pred_global[fid_idx] != -1)
        if not np.any(covered):
            continue
        y_f = labels_time[fid_idx][covered]
        p_f = pred_global[fid_idx][covered]
        s_f = test_time_scores[fid_idx][covered]   # 分数（可用于调试/可视化）

        m = confusion_and_prf(y_f, p_f)
        per_flight[int(fid)] = m

        # ---- 误分类点可视化 ----
        mismatches = fid_idx[covered][(p_f != y_f)]          # 该航段中的误分类“全局索引”
        normal_idx_fid   = fid_idx[labels_time[fid_idx] == 0]
        abnormal_idx_fid = fid_idx[labels_time[fid_idx] == 1]

        plt.figure(figsize=(11, 3))
        # 误分类点
        if mismatches.size > 0:
            plt.scatter(mismatches, vrtg_values[mismatches], marker='x', s=12, label='misclassified', color='red')
        # 正常/异常的 VRTG
        plt.scatter(normal_idx_fid,   vrtg_values[normal_idx_fid],   s=1, label='normal', color='green')
        plt.scatter(abnormal_idx_fid, vrtg_values[abnormal_idx_fid], s=1, label='abnormal', color='blue')

        plt.title(f'Flight {fid} — misclassified points '
                  f'(P={m["precision"]:.3f}, R={m["recall"]:.3f}, F1={m["f1"]:.3f})')
        plt.xlabel('time index')
        plt.ylabel('VRTG')
        plt.legend(loc='best')
        plt.tight_layout()
        save_path = os.path.join(plot_save_dir, f"flight_{fid}_misclassified.png")
        plt.savefig(save_path, dpi=150)
        print(f"图像已保存到: {save_path}")
        plt.show()

    # 5) 分航段汇总（macro/micro）
    if len(per_flight) > 0:
        macro_P = np.mean([m["precision"] for m in per_flight.values()])
        macro_R = np.mean([m["recall"]    for m in per_flight.values()])
        macro_F1 = np.mean([m["f1"]       for m in per_flight.values()])
        TP = sum(m["tp"] for m in per_flight.values())
        FP = sum(m["fp"] for m in per_flight.values())
        FN = sum(m["fn"] for m in per_flight.values())
        TN = sum(m["tn"] for m in per_flight.values())
        micro_P = TP / (TP + FP + 1e-12)
        micro_R = TP / (TP + FN + 1e-12)
        micro_F1 = 2 * micro_P * micro_R / (micro_P + micro_R + 1e-12)
        micro_Acc = (TP + TN) / max(1, TP + TN + FP + FN)
        print("\n=== Per-flight summary ===")
        print(f"Flights counted: {len(per_flight)}")
        print(f"Macro  P={macro_P:.4f} R={macro_R:.4f} F1={macro_F1:.4f}")
        print(f"Micro  P={micro_P:.4f} R={micro_R:.4f} F1={micro_F1:.4f} Acc={micro_Acc:.4f}")

    return overall, per_flight



def run_demo(seed: int = 42):
    """
    Train a small model on synthetic "normal" -> validate -> test on attacked sequence.
    This is JUST a smoke test of the full pipeline.
    """
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = TrainConfig(
        window=100,
        batch_size=256,
        lr=1e-3,
        warmup_steps=8000,
        max_epochs=5,
        beta_kl=0.1,
        device=device,
        z_dim=20,
        hidden=64,
        layers=1,
        bidir=True,
        alpha=0.5,   # ADScore 融合权重
        n_critic=5,  # 判别器更新步
        lr_g=1e-3,
        lr_d=1e-3,
        beta_rl=1.0, # 重构项权重
        checkpoint_dir="checkpoints/GAN/1113",
        # checkpoint_dir="/mnt/checkpoints/GAN/1113",
        train_model=True,
        fit_collect_errors=True,
        
    )
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # data
    train_data, val_data, test_data = load_VAE_data('data')

    train_flight_ids = train_data['flight_id'].values
    val_flight_ids = val_data['flight_id'].values
    test_flight_ids = test_data['flight_id'].values

    train_data = compute_features(train_data, rolling_window_size=cfg.window)
    val_data = compute_features(val_data, rolling_window_size=cfg.window)
    test_data = compute_features(test_data, rolling_window_size=cfg.window)

    X_train, y_train = train_data[cfg.features].values, (train_data['status'] == 'abnormal').astype(int).values
    X_val, y_val = val_data[cfg.features].values, (val_data['status'] == 'abnormal').astype(int).values
    X_test, y_test = test_data[cfg.features].values, (test_data['status'] == 'abnormal').astype(int).values
    print(f"训练集 异常样本比例: {sum(y_train)/len(y_train):.4f}")
    print(f"验证集 异常样本比例: {sum(y_val)/len(y_val):.4f}")
    print(f"测试集 异常样本比例: {sum(y_test)/len(y_test):.4f}")
    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    # ensure arrays are torch tensors before scaling
    X_train = torch.as_tensor(X_train, dtype=torch.float32)
    X_val = torch.as_tensor(X_val, dtype=torch.float32)
    X_test = torch.as_tensor(X_test, dtype=torch.float32)

    # fit scaler on "normal" train
    scaler = MinMaxScaler()
    # fit scaler on tensor data (redo if fit was called earlier with numpy)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    train_loader = DataLoader(SlidingWindowADSBDataset(X_train, cfg.window), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(SlidingWindowADSBDataset(X_val, cfg.window), batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(SlidingWindowADSBDataset(X_test, cfg.window), batch_size=cfg.batch_size, shuffle=False)

    # 查看loader数据
    for x_in, y_tgt, t_idx in train_loader:
        print(f"Train loader batch x_in: {x_in.shape}, y_tgt: {y_tgt.shape}, t_idx: {t_idx.shape}")
        break

    for x_in, y_tgt, t_idx in val_loader:
        print(f"Val loader batch x_in: {x_in.shape}, y_tgt: {y_tgt.shape}, t_idx: {t_idx.shape}")
        break

    for x_in, y_tgt, t_idx in test_loader:
        print(f"Test loader batch x_in: {x_in.shape}, y_tgt: {y_tgt.shape}, t_idx: {t_idx.shape}")
        break

    # ---- 模型 ----
    model = LSTMGAN(
        feat_dim=len(cfg.features),
        z_dim=getattr(cfg, "z_dim", 20),
        hidden=getattr(cfg, "hidden", 64),
        layers=getattr(cfg, "layers", 1),
        bidir=getattr(cfg, "bidir", True)
    ).to(cfg.device)

    # 这两个对象保留（即便 GAN 训练没有直接使用），用于兼容你现有的 checkpoint 恢复逻辑
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = NoamLR(optim, warmup_steps=16000)



    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    device = cfg.device
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())  # 先用初始权重兜底
    best_calib = {"dx_min": 0.0, "dx_max": 1.0}
    best_mu_sigma = (0.0, 1.0)

    best_ckpt_path = os.path.join(cfg.checkpoint_dir, "best_checkpoint.pth")
    raw_state_path = os.path.join(cfg.checkpoint_dir, "best_lstm_gan.pt")
    gauss_path = os.path.join(cfg.checkpoint_dir, "gaussian_mu_sigma.npy")
    calib_path = os.path.join(cfg.checkpoint_dir, "calib_dx.json")
    scores_path = os.path.join(cfg.checkpoint_dir, "test_time_scores.npy")
    counts_path = os.path.join(cfg.checkpoint_dir, "counts.npy")

    if cfg.train_model:
        for epoch in range(1, cfg.max_epochs + 1):
            # -------- 训练一个 epoch（正常序列）--------
            meters = train_epoch_gan(model, train_loader, cfg, device)
            print(f"[Epoch {epoch:02d}] "
                f"dx={meters['dx']:.4f} dz={meters['dz']:.4f} g={meters['g']:.4f} rl={meters['rl']:.4f}")

            # -------- 在 val(正常) 上校准 Dx 分数并计算 val 指标 --------
            calib = calibrate_dx_on_val(model, val_loader, device)  # {"dx_min":..., "dx_max":...}

            # val 平均 AD 分数（越小越好）作为早停指标
            model.eval()
            total_ad, total_cnt = 0.0, 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Evaluating on val", unit="step"):
                    x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                    B, T, C = x.size()
                    z = model.enc(x)
                    x_rec = model.dec(z, T=T)
                    rl = F.mse_loss(x_rec, x, reduction="none").mean(dim=(1, 2))   # (B,)

                    s = model.dx(x)  # 越大越“真”
                    dx_min, dx_max = calib["dx_min"], calib["dx_max"]
                    if dx_max > dx_min:
                        dl = (dx_max - s) / (dx_max - dx_min)
                    else:
                        dl = (-s).sigmoid()

                    ad = cfg.alpha * rl + (1.0 - cfg.alpha) * dl                   # (B,)
                    total_ad += float(ad.sum().item())
                    total_cnt += int(ad.numel())
            val_metric = total_ad / max(1, total_cnt)
            print(f"           val_ad(mean)={val_metric:.6f}  "
                f"[calib dx_min={dx_min:.4f}, dx_max={dx_max:.4f}]")

            # 为了后续 NLL 评估的一致性：在 val(正常) 上拟合 (mu, sigma)
            mu, sigma = fit_gaussian_on_val(model, val_loader, device, alpha=cfg.alpha, calib=calib)

            # -------- 早停/保留最优 --------
            if val_metric < best_val:
                best_val = val_metric
                best_state = copy.deepcopy(model.state_dict())
                best_calib = dict(calib)
                best_mu_sigma = (float(mu), float(sigma))

                # 同时落盘两份：1) 纯 state_dict（兼容简单加载） 2) 完整 checkpoint（含配置/优化器等）
                torch.save(best_state, raw_state_path)
                torch.save({
                    "epoch": epoch,
                    "best_val": best_val,
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optim.state_dict(),  # 为兼容你已有的加载分支
                    "calib": best_calib,
                    "mu": best_mu_sigma[0],
                    "sigma": best_mu_sigma[1],
                    "cfg": getattr(cfg, "__dict__", dict(cfg=1)),  # 粗存 cfg
                }, best_ckpt_path)
                # 也把校准与高斯参数单独保存，便于后续独立加载
                with open(calib_path, "w", encoding="utf-8") as f:
                    json.dump(best_calib, f, ensure_ascii=False, indent=2)
                np.save(gauss_path, np.array(best_mu_sigma, dtype=np.float64))

        # -------- 训练结束：加载最优权重 --------
        model.load_state_dict(best_state)
        model.to(device)

    else:
        # Restore from checkpoint if available
        checkpoint_path = best_ckpt_path
        ckpt = torch.load(checkpoint_path, map_location=cfg.device)
        model.load_state_dict(ckpt["model_state_dict"])
        optim.load_state_dict(ckpt.get("optimizer_state_dict", optim.state_dict()))
        # move optimizer states to correct device
        for state in optim.state.values():
            for k, v in list(state.items()):
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(cfg.device)
        print(f"Loaded checkpoint from {checkpoint_path}")
        # 尝试载入校准与高斯参数（若不存在则退化）
        if os.path.exists(calib_path):
            with open(calib_path, "r", encoding="utf-8") as f:
                best_calib = json.load(f)
        if os.path.exists(gauss_path):
            tmp = np.load(gauss_path)
            best_mu_sigma = (float(tmp[0]), float(tmp[1]))

    # ====== STEP 2: 产出测试分数并保存 / 或载入已保存分数 ======
    if cfg.fit_collect_errors:
        # 若你希望严格使用“最优模型对应的校准/高斯参数”，可用 best_calib / best_mu_sigma；
        # 也可以按当前模型重新估计（默认采用重新估计，以降低数据分布漂移的影响）
        calib = calibrate_dx_on_val(model, val_loader, cfg.device)
        mu, sigma = fit_gaussian_on_val(model, val_loader, cfg.device, alpha=cfg.alpha, calib=calib)

        time_scores, counts = collect_step_errors(
            model, test_loader, T=len(y_test),
            device=cfg.device, alpha=cfg.alpha, calib=calib
        )
        mask = counts > 0
        time_scores = np.where(mask, time_scores / np.maximum(counts, 1), 0.0)
        test_time_scores = nll_scores(time_scores, mu, sigma)   # 若你保留 NLL 路线
        test_counts = counts

        # 保存：时间步 NLL 分数、覆盖次数、以及高斯与校准参数
        np.save(scores_path, test_time_scores.astype(np.float32))
        np.save(counts_path, test_counts.astype(np.int64))
        np.save(gauss_path, np.array([mu, sigma], dtype=np.float64))
        with open(calib_path, "w", encoding="utf-8") as f:
            json.dump(calib, f, ensure_ascii=False, indent=2)

        print(f"Saved test_time_scores -> {scores_path}")
        print(f"Saved counts          -> {counts_path}")
        print(f"Saved gaussian mu/sig -> {gauss_path}")
        print(f"Saved calib (Dx)      -> {calib_path}")

    else:
        test_time_scores = np.load(scores_path)
        print(f"Loaded test_time_scores ({test_time_scores.shape}) from {scores_path}")

        test_counts = np.load(counts_path)
        print(f"Loaded test_counts ({test_counts.shape}) from {counts_path}")

        if os.path.exists(gauss_path):
            tmp = np.load(gauss_path)
            mu, sigma = float(tmp[0]), float(tmp[1])
            print(f"Loaded Gaussian parameters: mu={mu:.6f}, sigma={sigma:.6f}")
        else:
            # 兜底：用当前 val 重新估计
            calib = calibrate_dx_on_val(model, val_loader, cfg.device)
            mu, sigma = fit_gaussian_on_val(model, val_loader, cfg.device, alpha=cfg.alpha, calib=calib)
            np.save(gauss_path, np.array([mu, sigma], dtype=np.float64))
            print(f"[Fallback] Saved Gaussian parameters: mu={mu:.6f}, sigma={sigma:.6f}")

    # ====== 评估与作图（保持你原逻辑） ======
    overall, per_flight = run_eval_and_plot(
        y_test=y_test,
        test_flight_ids=test_flight_ids,
        vrtg_values=test_data['VRTG'].values,
        test_time_scores=test_time_scores,       # 长度 T
        test_counts=test_counts,                 # 长度 T
        choose_threshold_grid=choose_threshold_grid,
        plot_save_dir=os.path.join(cfg.checkpoint_dir, 'test_plots'),
        title_prefix="TEST"
    )

    return overall, per_flight


# -------------------------------
# Main guard for quick demo
# -------------------------------

if __name__ == "__main__":
    run_demo()
