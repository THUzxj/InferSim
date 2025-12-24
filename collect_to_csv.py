import json
import csv
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import matplotlib.pyplot as plt


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten nested dict like {"a": {"b": 1}} -> {"a.b": 1}
    """
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def parse_filename_meta(filename: str) -> Dict[str, Any]:
    """
    Parse input_length and batch_size from filename like:
    input_length128_batch_size4.json
    """
    m = re.match(r"input_length(\d+)_batch_size(\d+)\.json$", filename)
    if not m:
        return {}
    input_len, batch_size = m.groups()
    return {
        "input_length": int(input_len),
        "batch_size": int(batch_size),
    }


def collect_json_to_csv(
    folder: Path,
    output_csv: Path,
) -> None:
    folder = folder.resolve()
    json_files: List[Path] = sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix == ".json"]
    )

    rows: List[Dict[str, Any]] = []
    for jf in json_files:
        with jf.open("r", encoding="utf-8") as f:
            data = json.load(f)

        flat = flatten_dict(data)
        meta = parse_filename_meta(jf.name)
        flat.update(meta)
        flat["filename"] = jf.name
        rows.append(flat)

    if not rows:
        print(f"No JSON files found in {folder}")
        return

    # Collect all keys for CSV header
    header_keys: List[str] = sorted({k for row in rows for k in row.keys()})

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {output_csv}")


def sort_and_plot(csv_path: Path) -> None:
    """
    - 按照 batch_size, config.target_isl 排序并覆盖 csv
    - 画两组多条折线图：
        1）x=batch_size, 每个 config.target_isl 一条线，y=decoding.tpot_ms
        2）x=config.target_isl, 每个 batch_size 一条线，y=decoding.tpot_ms
    """
    df = pd.read_csv(csv_path)

    # 确保需要的列存在
    required_cols = ["batch_size", "config.target_isl", "decoding.tpot_ms"]
    for col in required_cols:
        if col not in df.columns:
            print(f"Column '{col}' not found in CSV, skip plotting.")
            return

    # 按 batch_size 和 config.target_isl 排序
    df_sorted = df.sort_values(by=["batch_size", "config.target_isl"])
    df_sorted.to_csv(csv_path, index=False)

    # 1）以 batch_size 为横轴：每个 config.target_isl 一条线
    plt.figure()
    for isl, sub in df_sorted.groupby("config.target_isl"):
        sub = sub.sort_values("batch_size")
        plt.plot(
            sub["batch_size"],
            sub["decoding.tpot_ms"],
            marker="o",
            label=f"target_isl={isl}",
        )
    plt.xlabel("batch_size")
    plt.ylabel("decoding.tpot_ms")
    plt.title("decoding.tpot_ms vs batch_size (per config.target_isl)")
    plt.grid(True)
    plt.legend(title="config.target_isl")
    bs_fig = csv_path.with_name(csv_path.stem + "_vs_batch_size.png")
    plt.savefig(bs_fig, bbox_inches="tight")
    plt.close()

    # 2）以 config.target_isl 为横轴：每个 batch_size 一条线
    plt.figure()
    for bs, sub in df_sorted.groupby("batch_size"):
        sub = sub.sort_values("config.target_isl")
        plt.plot(
            sub["config.target_isl"],
            sub["decoding.tpot_ms"],
            marker="o",
            label=f"batch_size={bs}",
        )
    plt.xlabel("config.target_isl")
    plt.ylabel("decoding.tpot.ms")
    plt.title("decoding.tpot_ms vs config.target_isl (per batch_size)")
    plt.grid(True)
    plt.legend(title="batch_size")
    isl_fig = csv_path.with_name(csv_path.stem + "_vs_target_isl.png")
    plt.savefig(isl_fig, bbox_inches="tight")
    plt.close()

    # 3）以 batch_size * config.target_isl 为横轴：按 batch_size 分组多条折线
    df_sorted["bs_times_isl"] = df_sorted["batch_size"] * \
        df_sorted["config.target_isl"]
    plt.figure()
    for bs, sub in df_sorted.groupby("batch_size"):
        sub = sub.sort_values("bs_times_isl")
        plt.plot(
            sub["bs_times_isl"],
            sub["decoding.tpot_ms"],
            marker="o",
            label=f"batch_size={bs}",
        )
    plt.xlabel("batch_size * config.target_isl")
    plt.ylabel("decoding.tpot.ms")
    plt.title(
        "decoding.tpot_ms vs (batch_size * config.target_isl) (per batch_size)")
    plt.grid(True)
    plt.legend(title="batch_size")
    prod_fig = csv_path.with_name(
        csv_path.stem + "_vs_bs_times_target_isl.png")
    plt.savefig(prod_fig, bbox_inches="tight")
    plt.close()

    print(f"Sorted CSV saved to {csv_path}")
    print(f"Saved plot: {bs_fig}")
    print(f"Saved plot: {isl_fig}")
    print(f"Saved plot: {prod_fig}")


def main():
    """
    Usage:
        python collect_to_csv.py /path/to/json_folder [output.csv]
    """
    if len(sys.argv) < 2:
        print(
            "Usage: python collect_to_csv.py /path/to/json_folder [output.csv]")
        sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"Error: {folder} is not a directory")
        sys.exit(1)

    if len(sys.argv) >= 3:
        output_csv = Path(sys.argv[2])
    else:
        output_csv = folder / "all_experiment_summary.csv"

    collect_json_to_csv(folder, output_csv)

    # 生成排序后的 csv 和折线图
    sort_and_plot(output_csv)


if __name__ == "__main__":
    main()
