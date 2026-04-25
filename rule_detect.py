import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_METADATA_PATH = "问题类型记录.xlsx"
DEFAULT_DATA_DIR = "data"


def load_metadata(metadata_path=DEFAULT_METADATA_PATH):
    metadata = pd.read_excel(metadata_path).copy()
    metadata["csv_name"] = metadata["csv_name"].apply(
        lambda value: f"{value}.csv" if not str(value).endswith(".csv") else str(value)
    )
    return metadata


def get_flight_file_map(metadata):
    return dict(zip(metadata["flight_id"], metadata["csv_name"]))


def get_flight_ids_by_split(metadata, split_seqs):
    if "split_seq" not in metadata.columns:
        raise ValueError("Metadata is missing the 'split_seq' column.")
    return metadata[metadata["split_seq"].isin(split_seqs)]["flight_id"].tolist()


def load_flight_data(flight_id, data_dir=DEFAULT_DATA_DIR, metadata_path=DEFAULT_METADATA_PATH):
    metadata = load_metadata(metadata_path)
    flight_files = get_flight_file_map(metadata)
    if flight_id not in flight_files:
        raise KeyError(f"Unknown flight_id: {flight_id}")

    file_path = Path(data_dir) / flight_files[flight_id]
    if not file_path.exists():
        raise FileNotFoundError(f"Flight data not found: {file_path}")

    flight_data = pd.read_csv(file_path, usecols=["status", "VRTG"]).copy()
    flight_data["flight_id"] = flight_id
    flight_data["index"] = range(len(flight_data))
    return flight_data


def is_gradual_red_event(vrtg_series, start, pre_window=10, normal_range=(0.8, 1.2), max_normal_ratio=0.2):
    """
    Heuristic filter for events that drift away from the normal range before
    crossing the abnormal threshold, rather than jumping abruptly.
    """
    if start < pre_window:
        return False

    pre_values = vrtg_series.iloc[start - pre_window:start]
    in_normal_range = pre_values.between(normal_range[0], normal_range[1])
    return in_normal_range.mean() <= max_normal_ratio


def detect_jump_segments(
    vrtg_series,
    low_thresh=0.5,
    high_thresh=1.8,
    jump_thresh=0.5,
    min_abnormal_len=1,
    pre_window=10,
    normal_range=(0.8, 1.2),
):
    """
    Detect abrupt jump segments by:
    1. locating contiguous out-of-range regions,
    2. requiring both entry and exit jumps to be large enough,
    3. filtering out gradual red events.
    """
    vrtg = pd.Series(vrtg_series).reset_index(drop=True)
    abnormal_mask = (vrtg < low_thresh) | (vrtg > high_thresh)
    segments = []
    i = 0

    while i < len(vrtg):
        if not abnormal_mask.iloc[i]:
            i += 1
            continue

        start = i
        while i < len(vrtg) and abnormal_mask.iloc[i]:
            i += 1
        end = i - 1

        if end - start + 1 < min_abnormal_len:
            continue

        pre_value = vrtg.iloc[start - 1] if start > 0 else None
        post_value = None
        for j in range(end + 1, len(vrtg)):
            if low_thresh <= vrtg.iloc[j] <= high_thresh:
                post_value = vrtg.iloc[j]
                break

        jump_in = abs(vrtg.iloc[start] - pre_value) if pre_value is not None else 0.0
        jump_out = abs(post_value - vrtg.iloc[end]) if post_value is not None else 0.0

        if jump_in <= jump_thresh or jump_out <= jump_thresh:
            continue

        if is_gradual_red_event(
            vrtg,
            start,
            pre_window=pre_window,
            normal_range=normal_range,
        ):
            continue

        segments.append((start, end))

    return segments


def segment_points(segments, min_index=0, max_index=None):
    points = []
    upper = float("inf") if max_index is None else max_index
    for start, end in segments:
        points.extend(point for point in range(start, end + 1) if min_index <= point < upper)
    return points


def plot_jump_segments(
    flight_data,
    segments,
    min_index=0,
    max_index=None,
    title=None,
    save_path=None,
    show=True,
):
    """
    Plot one flight with normal points, abnormal points, and detected jump points.
    """
    plot_data = flight_data.iloc[min_index:max_index].copy()
    if plot_data.empty:
        raise ValueError("No data left to plot after applying min_index/max_index.")

    normal_points = plot_data[plot_data["status"] == "normal"]
    abnormal_points = plot_data[plot_data["status"] != "normal"]
    jump_points = segment_points(
        segments,
        min_index=min_index,
        max_index=len(flight_data) if max_index is None else max_index,
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(1.8, color="gray", linestyle="-.", linewidth=1, label="upper threshold")
    ax.axhline(1.0, color="gray", linestyle="-", linewidth=1, label="center line")
    ax.axhline(0.3, color="gray", linestyle="-.", linewidth=1, label="lower reference")

    ax.scatter(normal_points.index, normal_points["VRTG"], color="green", s=1, label="normal")
    ax.scatter(abnormal_points.index, abnormal_points["VRTG"], color="blue", s=2, label="abnormal")

    if jump_points:
        jump_values = flight_data.loc[jump_points, "VRTG"]
        ax.scatter(jump_points, jump_values, color="red", marker="x", s=10, label="detected jump")

    ax.set_title(title or f"Flight {plot_data['flight_id'].iloc[0]} jump detection")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("VRTG")
    ax.legend(loc="upper right")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def inspect_flight(
    flight_id,
    data_dir=DEFAULT_DATA_DIR,
    metadata_path=DEFAULT_METADATA_PATH,
    min_index=0,
    max_index=None,
    save_path=None,
    show=True,
    **rule_kwargs,
):
    flight_data = load_flight_data(
        flight_id=flight_id,
        data_dir=data_dir,
        metadata_path=metadata_path,
    )
    segments = detect_jump_segments(flight_data["VRTG"], **rule_kwargs)
    plot_jump_segments(
        flight_data=flight_data,
        segments=segments,
        min_index=min_index,
        max_index=max_index,
        save_path=save_path,
        show=show,
    )
    return segments


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Rule-based VRTG jump detection for one flight.")
    parser.add_argument("--flight-id", type=int, default=None, help="Single flight ID to inspect.")
    parser.add_argument(
        "--flight-ids",
        nargs="+",
        type=int,
        default=None,
        help="Multiple flight IDs to inspect in one run.",
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory containing CSV files.")
    parser.add_argument(
        "--metadata-path",
        default=DEFAULT_METADATA_PATH,
        help="Excel file containing flight_id -> csv_name metadata.",
    )
    parser.add_argument("--min-index", type=int, default=0, help="Plot start index.")
    parser.add_argument("--max-index", type=int, default=None, help="Plot end index.")
    parser.add_argument("--save-path", default=None, help="Optional image output path.")
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Optional directory for batch image outputs. One file will be created per flight.",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not open the plot window.")
    parser.add_argument("--low-thresh", type=float, default=0.5)
    parser.add_argument("--high-thresh", type=float, default=1.8)
    parser.add_argument("--jump-thresh", type=float, default=0.5)
    parser.add_argument("--min-abnormal-len", type=int, default=1)
    parser.add_argument("--pre-window", type=int, default=10)
    return parser


def main():
    args = build_arg_parser().parse_args()
    flight_ids = []
    if args.flight_id is not None:
        flight_ids.append(args.flight_id)
    if args.flight_ids:
        flight_ids.extend(args.flight_ids)

    if not flight_ids:
        raise SystemExit("Please provide --flight-id or --flight-ids.")

    # Preserve input order while removing duplicates.
    flight_ids = list(dict.fromkeys(flight_ids))

    for flight_id in flight_ids:
        save_path = args.save_path
        if args.save_dir:
            save_path = str(Path(args.save_dir) / f"flight_{flight_id}_jump_detection.png")

        segments = inspect_flight(
            flight_id=flight_id,
            data_dir=args.data_dir,
            metadata_path=args.metadata_path,
            min_index=args.min_index,
            max_index=args.max_index,
            save_path=save_path,
            show=not args.no_show,
            low_thresh=args.low_thresh,
            high_thresh=args.high_thresh,
            jump_thresh=args.jump_thresh,
            min_abnormal_len=args.min_abnormal_len,
            pre_window=args.pre_window,
        )
        print(f"Detected jump segments for flight {flight_id}: {segments}")


if __name__ == "__main__":
    main()
