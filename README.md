# VRTG Anomaly Detection

这个仓库用于基于航段 VRTG 时序数据做异常检测实验，当前主要包含两条路线：

- `whole_train.py`: 使用 BiLSTM 训练逐点异常分类模型
- `whole_predict.py`: 加载训练好的 checkpoint，对训练集和测试集做逐航段评估并保存结果

另外仓库里还有一些探索性脚本和 notebook，例如：

- `rule_detect.py`: 基于规则的异常检测探索
- `transformer_vae_adsb.py`: Transformer VAE 实验
- `lstm_gan.py`: LSTM GAN 实验

## 目录说明

- `data/`: 原始 CSV 数据目录，默认已被 Git 忽略
- `checkpoints/`: 训练输出的模型权重目录，默认已被 Git 忽略
- `results/`: 评估结果输出目录，默认已被 Git 忽略
- `plots/`: 可视化图像输出目录，默认已被 Git 忽略
- `问题类型记录.xlsx`: 航段与 `csv_name`、`split_seq` 等元数据映射表

## 环境依赖

项目代码依赖以下 Python 包：

- `torch`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `tqdm`
- `openpyxl`

可按需自行安装，例如：

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn tqdm openpyxl
```

## 数据准备

将数据文件手动放到 `data/` 目录下。代码默认会：

- 从 `问题类型记录.xlsx` 读取每个航段对应的 `csv_name`
- 在 `data/<csv_name>` 位置查找 CSV 文件
- 从每个 CSV 中读取 `status` 和 `VRTG` 两列

## 关键配置

主要配置集中在 [`config.py`](/root/yyq/vrtg/config.py)：

- `date`: 控制 checkpoint 子目录
- `platform`: `local` 或 `server`
- `features`: 训练使用的特征列
- `window_size`: 时序窗口半径
- `rolling_window_size`: 滚动统计特征窗口
- `num_epochs`, `batch_size`, `hidden_size`, `num_layers`: 模型训练参数

## 使用方式

训练模型：

```bash
python whole_train.py
```

也可以显式传参：

```bash
python whole_train.py \
  --data-dir data \
  --date 1021 \
  --epochs 5 \
  --batch-size 256 \
  --window-size 100
```

运行评估：

```bash
python whole_predict.py
```

如果要指定某次训练的输出目录或 checkpoint：

```bash
python whole_predict.py \
  --checkpoint-dir checkpoints/1021 \
  --checkpoint-path checkpoints/1021/best_checkpoint.pth
```

运行后默认会产出：

- `checkpoints/<date>/best_checkpoint.pth`
- `checkpoints/<date>/scaler.pkl`
- `checkpoints/<date>/train_args.json`
- `results/train_results_dict.pkl`
- `results/test_results_dict.pkl`
- `results/overall_summary.pkl`
- `plots/` 下的逐航段误分类可视化图片

## 规则检测

[`rule_detect.py`](/root/yyq/vrtg/rule_detect.py) 提供了一套基于阈值和跳变幅度的规则检测，用于识别 VRTG 序列中的 jump 异常。

规则大致分三步：

- 先找出 `VRTG < 0.5` 或 `VRTG > 1.8` 的连续异常区间
- 再要求异常段进入和退出时都存在足够大的跳变，默认 `jump_thresh = 0.5`
- 最后过滤掉疑似“渐变红色事件”的片段，避免把缓慢漂移误判成 jump

常用函数包括：

- `load_flight_data()`: 按 `flight_id` 读取单个航段数据
- `detect_jump_segments()`: 返回检测出的 jump 区间列表
- `plot_jump_segments()`: 绘制正常点、异常点和检测到的 jump 点
- `inspect_flight()`: 单航段一站式检测和画图

单个航段检测：

```bash
python rule_detect.py --flight-id 35 --data-dir data
```

如果当前终端没有图形界面，建议直接保存图像：

```bash
python rule_detect.py \
  --flight-id 35 \
  --data-dir data \
  --save-path plots/rule_detect_f35.png \
  --no-show
```

批量检测指定航段：

```bash
python rule_detect.py \
  --flight-ids 35 36 98 \
  --data-dir data \
  --save-dir plots/rule_batch \
  --no-show
```

这会为每个航段分别生成一张图，例如：

- `plots/rule_batch/flight_35_jump_detection.png`
- `plots/rule_batch/flight_36_jump_detection.png`
- `plots/rule_batch/flight_98_jump_detection.png`

可调参数包括：

- `--low-thresh`
- `--high-thresh`
- `--jump-thresh`
- `--min-abnormal-len`
- `--pre-window`
- `--min-index` / `--max-index`

## 当前代码特点

- 主流程是脚本式组织，不是包结构
- notebook 与实验脚本较多，适合继续按“主流程 + 探索实验”方式维护
- 数据和模型产物已经通过 `.gitignore` 隔离，避免大文件进入版本控制

## 建议

如果后面我们继续整理，下一步最值得做的是：

1. 把依赖固定到 `requirements.txt`
2. 把训练/评估入口参数化，减少直接改源码配置
3. 把 `utils.py` 中的数据加载和特征工程再拆分成独立模块
