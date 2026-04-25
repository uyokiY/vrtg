import argparse
import json
import os
import pickle

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATE,
    DEFAULT_FEATURES,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_NUM_LAYERS,
    DEFAULT_OUTPUT_SIZE,
    DEFAULT_PLATFORM,
    DEFAULT_ROLLING_WINDOW_SIZE,
    DEFAULT_SAMPLE_IDS,
    DEFAULT_VAL_SAMPLE_IDS,
    DEFAULT_WINDOW_SIZE,
    build_runtime_paths,
)
from utils import BiLSTM, TimeSeriesDataset, build_split_summary, compute_features, load_bilstm_data, train


def parse_args():
    parser = argparse.ArgumentParser(description="Train a BiLSTM VRTG anomaly detector.")
    parser.add_argument("--data-dir", default="data", help="Directory containing flight CSV files.")
    parser.add_argument("--metadata-path", default="问题类型记录.xlsx", help="Flight metadata Excel path.")
    parser.add_argument(
        "--date",
        default=DEFAULT_DATE,
        help="Run identifier used to build the default checkpoint directory.",
    )
    parser.add_argument(
        "--platform",
        choices=["local", "server"],
        default=DEFAULT_PLATFORM,
        help="Controls the default output directories.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Override checkpoint output directory.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
        help="Feature columns used for training.",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        type=int,
        default=DEFAULT_SAMPLE_IDS,
        help="Extra sampled flight IDs merged into the training split.",
    )
    parser.add_argument(
        "--val-sample-ids",
        nargs="+",
        type=int,
        default=DEFAULT_VAL_SAMPLE_IDS,
        help="Representative normal flight IDs merged into the validation split.",
    )
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--rolling-window-size", type=int, default=DEFAULT_ROLLING_WINDOW_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--output-size", type=int, default=DEFAULT_OUTPUT_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--max-flights",
        type=int,
        default=None,
        help="Limit the number of training flights for quick smoke tests.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example 'cpu' or 'cuda'.",
    )
    return parser.parse_args()


def ensure_features_exist(dataframe, features):
    missing = [feature for feature in features if feature not in dataframe.columns]
    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}. "
            f"Available columns: {sorted(dataframe.columns.tolist())}"
        )


def limit_flights(dataframe, max_flights):
    if max_flights is None:
        return dataframe

    selected_flight_ids = dataframe["flight_id"].drop_duplicates().head(max_flights).tolist()
    limited = dataframe[dataframe["flight_id"].isin(selected_flight_ids)].copy()
    print(f"Limiting training to first {len(selected_flight_ids)} flights: {selected_flight_ids}")
    return limited


def save_training_artifacts(args, checkpoint_dir, scaler, train_data, val_data, test_data):
    scaler_path = os.path.join(checkpoint_dir, "scaler.pkl")
    with open(scaler_path, "wb") as handle:
        pickle.dump(scaler, handle)

    run_config = {
        "data_dir": args.data_dir,
        "metadata_path": args.metadata_path,
        "date": args.date,
        "platform": args.platform,
        "checkpoint_dir": checkpoint_dir,
        "features": args.features,
        "split_strategy": "bilstm_module_whole_vs_sampled_normal",
        "sample_ids": args.sample_ids,
        "val_sample_ids": args.val_sample_ids,
        "window_size": args.window_size,
        "rolling_window_size": args.rolling_window_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "output_size": args.output_size,
        "learning_rate": args.learning_rate,
        "max_flights": args.max_flights,
        "device": args.device,
        "input_size": len(args.features),
        "train_flight_ids": sorted(train_data["flight_id"].unique().tolist()),
        "val_flight_ids": sorted(val_data["flight_id"].unique().tolist()),
        "test_flight_ids": sorted(test_data["flight_id"].unique().tolist()),
    }

    config_path = os.path.join(checkpoint_dir, "train_args.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, ensure_ascii=False, indent=2)

    print(f"Scaler saved to: {scaler_path}")
    print(f"Training config saved to: {config_path}")


def save_split_summary(checkpoint_dir, train_data, val_data, test_data):
    summary = build_split_summary({
        "train": train_data,
        "val": val_data,
        "test": test_data,
    })
    summary_path = os.path.join(checkpoint_dir, "split_summary.csv")
    summary.to_csv(summary_path, index=False)

    totals = summary.groupby("split", as_index=False)[
        ["normal_points", "abnormal_points", "total_points"]
    ].sum()
    totals["abnormal_ratio"] = totals["abnormal_points"] / totals["total_points"]
    totals_path = os.path.join(checkpoint_dir, "split_totals.csv")
    totals.to_csv(totals_path, index=False)

    print(f"Split summary saved to: {summary_path}")
    print(f"Split totals saved to: {totals_path}")


def main():
    args = parse_args()
    checkpoint_dir, _, _ = build_runtime_paths(args.date, args.platform)
    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    train_data, val_data, test_data = load_bilstm_data(
        args.data_dir,
        metadata_path=args.metadata_path,
        train_sample_ids=args.sample_ids,
        val_sample_ids=args.val_sample_ids,
    )
    if train_data.empty or val_data.empty:
        raise ValueError("Train/val data is empty. Please verify --data-dir and metadata mappings.")

    train_data = compute_features(train_data, args.rolling_window_size)
    val_data = compute_features(val_data, args.rolling_window_size)
    train_data = limit_flights(train_data, args.max_flights)
    ensure_features_exist(train_data, args.features)
    ensure_features_exist(val_data, args.features)

    x_train = train_data[args.features].values
    y_train = (train_data["status"] == "abnormal").astype(int).values
    x_val = val_data[args.features].values
    y_val = (val_data["status"] == "abnormal").astype(int).values

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    flight_ids_train = train_data["flight_id"].values
    indices_train = train_data["index"].values
    flight_ids_val = val_data["flight_id"].values
    indices_val = val_data["index"].values

    print(f"X_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"训练集异常样本比例: {sum(y_train) / len(y_train):.4f}")
    print(f"X_val: {x_val.shape}, y_val: {y_val.shape}")
    print(f"验证集异常样本比例: {sum(y_val) / len(y_val):.4f}")

    train_dataset = TimeSeriesDataset(
        x_train, y_train, flight_ids_train, indices_train, args.window_size
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_dataset = TimeSeriesDataset(
        x_val, y_val, flight_ids_val, indices_val, args.window_size
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )

    for batch_x, batch_y, _, _ in train_loader:
        print("Batch X shape:", batch_x.shape)
        print("Batch Y shape:", batch_y.shape)
        break

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")

    device = torch.device(args.device)
    print("device:", device)

    model = BiLSTM(len(args.features), args.hidden_size, args.output_size, args.num_layers)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    history = train(
        model,
        train_loader,
        epochs=args.epochs,
        start_epoch=0,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=checkpoint_dir,
        val_loader=val_loader,
    )

    history_path = os.path.join(checkpoint_dir, "history.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"Training history saved to: {history_path}")

    save_training_artifacts(args, checkpoint_dir, scaler, train_data, val_data, test_data)
    save_split_summary(checkpoint_dir, train_data, val_data, test_data)


if __name__ == "__main__":
    main()
