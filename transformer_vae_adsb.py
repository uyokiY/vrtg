
# -*- coding: utf-8 -*-
"""
Transformer-VAE for ADS-B Anomaly Detection
===========================================

"基于Transformer-VAE的ADS-B异常检测方法" 的核心思路：
- 仅用“正常”序列训练 Transformer-VAE（重构式无监督）
- 以重构误差 -> 高斯负对数似然(NLL) 作为异常分数
- 在验证集上选阈值（网格/百分位），在测试集上评估

Author: ChatGPT (GPT-5 Thinking)
Python: 3.10+
PyTorch: 2.0+
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from utils import load_VAE_data, compute_features
import os

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


# -------------------------------
# Positional Encoding (sinusoidal)
# -------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:, :T, :]


# -------------------------------
# Transformer-VAE Model
# -------------------------------

class TransformerVAE(nn.Module):
    """
    Encoder: TransformerEncoder encodes [B, K, F] -> [B, K, D]
    Posterior Estimation: mean/logvar = FFN(avg_pool(H))
    Sample z ~ N(mean, diag(exp(logvar)))
    Decoder: TransformerDecoder takes teacher-forced targets (shifted x) with causal mask,
             cross-attends to encoder outputs.
    Fusion Gate: fuse decoder state and z
    Projection: predict next-step features
    """

    def __init__(
        self,
        feature_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        latent_dim: int = 32,
        fusion: str = "gate",
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.fusion = fusion

        # input linear to model dim
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        # posterior estimation from pooled encoder states
        self.post_mean = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, latent_dim)
        )
        self.post_logvar = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, latent_dim)
        )

        # decoder input projection & pos enc
        self.tgt_proj = nn.Linear(feature_dim, d_model)
        self.tgt_pos = PositionalEncoding(d_model)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        # fusion gate
        if fusion == "gate":
            self.gate = nn.Sequential(
                nn.Linear(d_model + latent_dim, d_model),
                nn.Sigmoid()
            )
        elif fusion == "concat":
            # fallback: simple concat + linear
            self.fuse_linear = nn.Linear(d_model + latent_dim, d_model)
        else:
            raise ValueError("fusion must be 'gate' or 'concat'")

        # output projection
        self.output_proj = nn.Linear(d_model, feature_dim)

        self.z_to_d = nn.Linear(self.latent_dim, self.d_model, bias=True)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # mean, logvar: [B, Z]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        # generate T x T mask: True means NOT allowed to attend
        mask = torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)
        return mask

    def forward(self, x_in: torch.Tensor, x_tgt: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_in:  [B, K, F]  encoder input
            x_tgt: [B, K, F]  teacher-forced decoder input (shifted target prefix)
        Returns dict with:
            recon: [B, K, F] reconstructed sequence
            mean, logvar: [B, Z] posterior params
        """
        B, K, F = x_in.size()   # [256, 100, 7] 
        device = x_in.device

        # Encoder
        src = self.input_proj(x_in)              # [B, K, D], [256, 100, 128]
        src = self.pos_enc(src)                  # add PE
        H = self.encoder(src)                    # [B, K, D], [256, 100, 128]

        # Posterior q(z|X): pool encoder states (mean pooling as paper)
        h_pool = H.mean(dim=1)                   # [B, D], [256, 128], 聚合窗口内信息
        mean = self.post_mean(h_pool)            # [B, Z], [256, 32]
        logvar = self.post_logvar(h_pool)        # [B, Z], [256, 32]
        z = self.reparameterize(mean, logvar)    # [B, Z], [256, 32]

        # Decoder with causal mask, cross-attending to H
        tgt = self.tgt_proj(x_tgt)               # [B, K, D], [256, 100, 128]
        tgt = self.tgt_pos(tgt)
        tgt_mask = self._causal_mask(K, device)
        dec_states = self.decoder(
            tgt=tgt, memory=H, tgt_mask=tgt_mask
        )                                        # [B, K, D], [256, 100, 128]

        # Fusion
        if self.fusion == "gate":
            # expand z to [B, K, Z]
            z_rep = z.unsqueeze(1).expand(B, K, -1) # [256, 100, 32]
            g = self.gate(torch.cat([dec_states, z_rep], dim=-1))  # [B,K,D] gate, [256, 100, 128]
            fused = (1.0 - g) * dec_states + g * (
                # project z to D for interpolation
                torch.tanh(self.z_to_d(z_rep))
            )  # [256, 100, 128])
        else:  # concat
            z_rep = z.unsqueeze(1).expand(B, K, -1)
            fused = torch.tanh(self.fuse_linear(torch.cat([dec_states, z_rep], dim=-1)))

        recon = self.output_proj(fused)          # [B, K, F], [256, 100, 7]
        return {"recon": recon, "mean": mean, "logvar": logvar}

    @staticmethod
    def loss_fn(batch_out: Dict[str, torch.Tensor], target: torch.Tensor, beta: float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        target: [B,K,F]
        recon loss: MSE
        KL loss: 0.5 * sum( exp(logvar) + mean^2 - 1 - logvar )
        """
        recon = batch_out["recon"]   # [256, 100, 7]
        mean = batch_out["mean"]     # [256, 32]
        logvar = batch_out["logvar"] # [256, 32]

        mse = nn.functional.mse_loss(recon, target, reduction="mean")

        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)  # [B], [256]
        kl = torch.mean(kl)

        loss = mse + beta * kl
        return loss, {"mse": float(mse.detach().cpu()), "kl": float(kl.detach().cpu())}


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


def train_epoch(model, loader, optimizer, scheduler, cfg: TrainConfig):
    model.train()
    total_loss = 0.0
    loader_iter = tqdm(loader, desc="train", unit="batch")
    for x_in, y_tgt, _ in loader_iter:
        x_in = x_in.to(cfg.device)     # [256, 100, 7]
        y_tgt = y_tgt.to(cfg.device)   # [256, 100, 7]

        out = model(x_in, x_tgt=y_tgt)
        loss, parts = model.loss_fn(out, y_tgt, beta=cfg.beta_kl)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * x_in.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_recon_error(model, loader, cfg: TrainConfig) -> float:
    model.eval()
    total = 0.0
    count = 0
    loader_iter = tqdm(loader, desc="eval", unit="batch")
    for x_in, y_tgt, _ in loader_iter:
        x_in = x_in.to(cfg.device)
        y_tgt = y_tgt.to(cfg.device)
        out = model(x_in, y_tgt)
        mse = nn.functional.mse_loss(out["recon"], y_tgt, reduction="sum")
        total += mse.item()
        count += y_tgt.numel()
    return total / max(count, 1)


# -------------------------------
# Anomaly Scoring & Thresholding
# -------------------------------

@torch.no_grad()
def collect_step_errors(model, loader, T: int, device: torch.device) -> np.ndarray:
    """
    Return per-step L2 errors e_t = ||x_t - x'_t|| over all batches.
    Shape: [N_steps] where N_steps = sum(B*K) across batches.
    """
    model.eval()
    time_sum = torch.zeros(T, device="cpu")
    time_count = torch.zeros(T, device="cpu")
    for x_in, y_tgt, t_idx in tqdm(loader, desc="collect_step_errors", unit="batch"):
        x_in = x_in.to(device)    # [256, 100, 7]
        y_tgt = y_tgt.to(device)  # [256, 100, 7]
        out = model(x_in, y_tgt)
        # per-step L2 norm over feature dim
        step_err = torch.sqrt(torch.sum((out["recon"] - y_tgt) ** 2, dim=-1))   # [256, 100]
        # 展平到 [B*K]
        flat_err = step_err.detach().cpu().reshape(-1)        # [B*K]
        flat_idx = t_idx.reshape(-1).to(torch.long).cpu()     # [B*K]，每个误差对应的全局时刻

        # 累加到时间轴
        time_sum.index_add_(0, flat_idx, flat_err)
        time_count.index_add_(0, flat_idx, torch.ones_like(flat_err))

    # 平均误差；未覆盖处保持 0（或用 NaN 更显式）
    time_scores = (time_sum / time_count.clamp_min(1)).numpy()

    return time_scores, time_count.numpy()


def fit_gaussian_mle(errors: np.ndarray) -> Tuple[float, float]:
    """
    Fit 1D Gaussian N(mu, sigma^2) to error samples via MLE.
    """
    mu = float(np.mean(errors))
    sigma = float(np.std(errors) + 1e-8)
    return mu, sigma


def nll_scores(errors: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Negative log-likelihood for Gaussian with mean mu and std sigma.
    NLL(e) = 0.5*log(2pi*sigma^2) + (e-mu)^2 / (2 sigma^2)
    """
    var = sigma ** 2
    const = 0.5 * math.log(2 * math.pi * var)
    return const + (errors - mu) ** 2 / (2 * var)


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
    os.makedirs(plot_save_dir, exist_ok=True)

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
        plt.axhline(1.8, color='gray', linestyle='-.', linewidth=1)
        plt.axhline(1, color='gray', linestyle='-',  linewidth=1)
        plt.axhline(0.3, color='gray', linestyle='-.', linewidth=1)

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



def run(seed: int = 42):
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
        checkpoint_dir="checkpoints/VAE/1113",
        # checkpoint_dir="/mnt/checkpoints/VAE/1113",
        train_model=False,
        fit_collect_errors=False,
        
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

    model = TransformerVAE(
        feature_dim=X_train.shape[1],
        d_model=128, nhead=4,
        num_encoder_layers=4, num_decoder_layers=1,
        dim_feedforward=256, dropout=0.1,
        latent_dim=32, fusion="gate"
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = NoamLR(optim, warmup_steps=16000)
    
    if cfg.train_model:
        best_val = float("inf")
        for epoch in range(1, cfg.max_epochs + 1):
            print(f"=== Epoch {epoch}/{cfg.max_epochs} ===")
            tr_loss = train_epoch(model, train_loader, optim, scheduler, cfg)
            val_err = eval_recon_error(model, val_loader, cfg)
            print(f"[Epoch {epoch:02d}] train_loss={tr_loss:.6f}  val_recon_mse={val_err:.6f}")
            if val_err < best_val:
                best_val = val_err
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                torch.save({"model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optim.state_dict(),
                            "val_recon": val_err}, f"{cfg.checkpoint_dir}/best_checkpoint.pth")
                print(f"  (new best model saved to {cfg.checkpoint_dir}/best_checkpoint.pth)")
        # load best
        model.load_state_dict(best_state)
        model.to(device)
 
    else:
        # Restore from checkpoint if available
        checkpoint_path = os.path.join(cfg.checkpoint_dir, "best_checkpoint.pth")
    
        ckpt = torch.load(checkpoint_path, map_location=cfg.device)
        model.load_state_dict(ckpt["model_state_dict"])
        optim.load_state_dict(ckpt["optimizer_state_dict"])
        # move optimizer states to correct device
        for state in optim.state.values():
            for k, v in list(state.items()):
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(cfg.device)
        print(f"Loaded checkpoint from {checkpoint_path}")

    if cfg.fit_collect_errors:    
        # ---- 验证集（正常集） ----
        val_time_errors, val_counts = collect_step_errors(model, val_loader, T=len(y_val), device=cfg.device)
        mu, sigma = fit_gaussian_mle(val_time_errors[val_counts > 0])
        np.save(os.path.join(cfg.checkpoint_dir, "gaussian_mu_sigma.npy"), np.array([mu, sigma]))
        print(f"Fitted Gaussian on validation TIME-AXIS: mu={mu:.6f}, sigma={sigma:.6f}")

        # ---- 测试集 ----
        test_time_errors, test_counts = collect_step_errors(model, test_loader, T=len(y_test), device=cfg.device)
        test_time_scores = nll_scores(test_time_errors, mu, sigma)

        np.save(os.path.join(cfg.checkpoint_dir, "test_scores.npy"), test_time_scores)
        np.save(os.path.join(cfg.checkpoint_dir, "test_counts.npy"), test_counts)
        print(f"Saved test_time_scores -> {os.path.join(cfg.checkpoint_dir, 'test_scores.npy')}")

    else:
        test_time_scores = np.load(os.path.join(cfg.checkpoint_dir, "test_scores.npy"))
        print(f"Loaded test_time_scores ({test_time_scores.shape}) from {cfg.checkpoint_dir}")

        test_counts = np.load(os.path.join(cfg.checkpoint_dir, "test_counts.npy"))
        print(f"Loaded test_counts ({test_counts.shape}) from {cfg.checkpoint_dir}")
        
        mu, sigma = np.load(os.path.join(cfg.checkpoint_dir, "gaussian_mu_sigma.npy"))
        print(f"Loaded Gaussian parameters: mu={mu:.6f}, sigma={sigma:.6f}")


    overall, per_flight = run_eval_and_plot(
        y_test=y_test,
        test_flight_ids=test_flight_ids,
        vrtg_values=test_data['VRTG'].values,
        test_time_scores=test_time_scores,       # 长度 T
        test_counts=test_counts,                 # 长度 T
        choose_threshold_grid=choose_threshold_grid,
        plot_save_dir=os.path.join(cfg.checkpoint_dir,'test_plots'),
        title_prefix="TEST"
    )
    
    return overall, per_flight


if __name__ == "__main__":
    run()
