import argparse
import json
import os
import pickle

import torch
from sklearn.preprocessing import StandardScaler

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATE,
    DEFAULT_FEATURES,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_NUM_LAYERS,
    DEFAULT_OUTPUT_SIZE,
    DEFAULT_PLATFORM,
    DEFAULT_ROLLING_WINDOW_SIZE,
    DEFAULT_SAMPLE_IDS,
    DEFAULT_VAL_SAMPLE_IDS,
    DEFAULT_WINDOW_SIZE,
    build_runtime_paths,
)
from predict_utils import aggregate_metrics, load_checkpoint_into_model, test_model_on_flight
from utils import BiLSTM, compute_features, load_bilstm_data


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained BiLSTM VRTG anomaly detector.")
    parser.add_argument("--data-dir", default="data", help="Directory containing flight CSV files.")
    parser.add_argument("--metadata-path", default=None, help="Flight metadata Excel path.")
    parser.add_argument(
        "--date",
        default=DEFAULT_DATE,
        help="Run identifier used to build default checkpoint/results directories.",
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
        help="Override checkpoint directory.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Explicit checkpoint path. Defaults to <checkpoint-dir>/best_checkpoint.pth.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Override results output directory.",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Override plot output directory.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Feature columns used for evaluation. Defaults to train_args.json or config defaults.",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        type=int,
        default=DEFAULT_SAMPLE_IDS,
        help="Extra sampled flight IDs merged into the training split when fitting fallback scaler.",
    )
    parser.add_argument(
        "--val-sample-ids",
        nargs="+",
        type=int,
        default=DEFAULT_VAL_SAMPLE_IDS,
        help="Representative normal flight IDs merged into the validation split.",
    )
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--rolling-window-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--output-size", type=int, default=None)
    parser.add_argument(
        "--max-flights",
        type=int,
        default=None,
        help="Limit the number of evaluated flights per split for quick smoke tests.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example 'cpu' or 'cuda'.",
    )
    return parser.parse_args()


def load_json_if_exists(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def ensure_features_exist(dataframe, features):
    missing = [feature for feature in features if feature not in dataframe.columns]
    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}. "
            f"Available columns: {sorted(dataframe.columns.tolist())}"
        )


def load_or_fit_scaler(checkpoint_dir, train_data, features):
    scaler_path = os.path.join(checkpoint_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as handle:
            scaler = pickle.load(handle)
        print(f"Loaded scaler from: {scaler_path}")
        return scaler

    print("No saved scaler found. Fitting a new scaler from the training split.")
    scaler = StandardScaler()
    scaler.fit(train_data[features].values)
    return scaler


def limit_flights(dataframe, max_flights, split_name):
    if max_flights is None:
        return dataframe

    selected_flight_ids = dataframe["flight_id"].drop_duplicates().head(max_flights).tolist()
    limited = dataframe[dataframe["flight_id"].isin(selected_flight_ids)].copy()
    print(f"Limiting {split_name} evaluation to first {len(selected_flight_ids)} flights: {selected_flight_ids}")
    return limited


def evaluate_split(split_name, flight_ids, data, model, args, scaler, features, window_size, device, plot_dir):
    results = {}
    for flight_id in flight_ids:
        print(f"Testing {split_name} flight ID: {flight_id}")
        flight_results = test_model_on_flight(
            model,
            args.batch_size,
            data,
            flight_id,
            features,
            scaler,
            window_size,
            device,
            plot_dir,
        )
        if flight_results:
            results[flight_id] = flight_results
            print(f"Flight ID {flight_id} 测试完成")
        else:
            print(f"Flight ID {flight_id} 测试失败或数据不足")
    return results


def main():
    args = parse_args()
    checkpoint_dir, default_plot_dir, default_results_dir = build_runtime_paths(
        args.date, args.platform
    )
    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    plot_dir = args.plot_dir or default_plot_dir
    results_dir = args.results_dir or default_results_dir
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    train_args = load_json_if_exists(os.path.join(checkpoint_dir, "train_args.json"))
    features = args.features or train_args.get("features") or DEFAULT_FEATURES
    metadata_path = args.metadata_path or train_args.get("metadata_path") or "问题类型记录.xlsx"
    sample_ids = train_args.get("sample_ids") or args.sample_ids
    val_sample_ids = train_args.get("val_sample_ids") or args.val_sample_ids
    window_size = args.window_size or train_args.get("window_size") or DEFAULT_WINDOW_SIZE
    rolling_window_size = (
        args.rolling_window_size
        or train_args.get("rolling_window_size")
        or DEFAULT_ROLLING_WINDOW_SIZE
    )
    hidden_size = args.hidden_size or train_args.get("hidden_size") or DEFAULT_HIDDEN_SIZE
    num_layers = args.num_layers or train_args.get("num_layers") or DEFAULT_NUM_LAYERS
    output_size = args.output_size or train_args.get("output_size") or DEFAULT_OUTPUT_SIZE

    checkpoint_path = args.checkpoint_path or os.path.join(checkpoint_dir, "best_checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    train_data, val_data, test_data = load_bilstm_data(
        args.data_dir,
        metadata_path=metadata_path,
        train_sample_ids=sample_ids,
        val_sample_ids=val_sample_ids,
    )
    if train_data.empty or val_data.empty or test_data.empty:
        raise ValueError("Train/val/test data is empty. Please verify --data-dir and metadata mappings.")

    train_data = compute_features(train_data, rolling_window_size)
    val_data = compute_features(val_data, rolling_window_size)
    test_data = compute_features(test_data, rolling_window_size)
    train_data = limit_flights(train_data, args.max_flights, "train")
    val_data = limit_flights(val_data, args.max_flights, "val")
    test_data = limit_flights(test_data, args.max_flights, "test")
    ensure_features_exist(train_data, features)
    ensure_features_exist(val_data, features)
    ensure_features_exist(test_data, features)

    scaler = load_or_fit_scaler(checkpoint_dir, train_data, features)

    x_train = train_data[features].values
    y_train = (train_data["status"] == "abnormal").astype(int).values
    x_val = val_data[features].values
    y_val = (val_data["status"] == "abnormal").astype(int).values
    x_test = test_data[features].values
    y_test = (test_data["status"] == "abnormal").astype(int).values

    print(f"X_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"训练集异常样本比例: {sum(y_train) / len(y_train):.4f}")
    print("训练集 flight IDs:", train_data["flight_id"].unique())

    print(f"X_val: {x_val.shape}, y_val: {y_val.shape}")
    print(f"验证集异常样本比例: {sum(y_val) / len(y_val):.4f}")
    print("验证集 flight IDs:", val_data["flight_id"].unique())

    print(f"X_test: {x_test.shape}, y_test: {y_test.shape}")
    print(f"测试集异常样本比例: {sum(y_test) / len(y_test):.4f}")
    print("测试集 flight IDs:", test_data["flight_id"].unique())

    device = torch.device(args.device)
    print("device:", device)
    model = BiLSTM(len(features), hidden_size, output_size, num_layers)
    model = load_checkpoint_into_model(model, checkpoint_path, device)

    test_results = evaluate_split(
        "test",
        test_data["flight_id"].unique(),
        test_data,
        model,
        args,
        scaler,
        features,
        window_size,
        device,
        plot_dir,
    )
    train_results = evaluate_split(
        "train",
        train_data["flight_id"].unique(),
        train_data,
        model,
        args,
        scaler,
        features,
        window_size,
        device,
        plot_dir,
    )
    val_results = evaluate_split(
        "val",
        val_data["flight_id"].unique(),
        val_data,
        model,
        args,
        scaler,
        features,
        window_size,
        device,
        plot_dir,
    )

    test_results_path = os.path.join(results_dir, "test_results_dict.pkl")
    with open(test_results_path, "wb") as handle:
        pickle.dump(test_results, handle)
    print(f"test_results_dict 已保存到：{test_results_path}")

    train_results_path = os.path.join(results_dir, "train_results_dict.pkl")
    with open(train_results_path, "wb") as handle:
        pickle.dump(train_results, handle)
    print(f"train_results_dict 已保存到：{train_results_path}")

    val_results_path = os.path.join(results_dir, "val_results_dict.pkl")
    with open(val_results_path, "wb") as handle:
        pickle.dump(val_results, handle)
    print(f"val_results_dict 已保存到：{val_results_path}")

    train_summary = aggregate_metrics(train_results)
    val_summary = aggregate_metrics(val_results)
    test_summary = aggregate_metrics(test_results)

    print("\n==== Train Overall Metrics ====")
    for key, value in train_summary["metrics"].items():
        print(f"{key}: {value:.4f}")
    print("Confusion matrix:\n", train_summary["confusion_matrix"])

    print("\n==== Val Overall Metrics ====")
    for key, value in val_summary["metrics"].items():
        print(f"{key}: {value:.4f}")
    print("Confusion matrix:\n", val_summary["confusion_matrix"])

    print("\n==== Test Overall Metrics ====")
    for key, value in test_summary["metrics"].items():
        print(f"{key}: {value:.4f}")
    print("Confusion matrix:\n", test_summary["confusion_matrix"])

    summary_path = os.path.join(results_dir, "overall_summary.pkl")
    with open(summary_path, "wb") as handle:
        pickle.dump({"train": train_summary, "val": val_summary, "test": test_summary}, handle)
    print(f"overall_summary 已保存到：{summary_path}")


if __name__ == "__main__":
    main()
