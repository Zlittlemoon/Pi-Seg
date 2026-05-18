import os
import re
import csv
import glob
import random
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np


DLRSD_THING_CLASSES = [
    "airplane",
    "buildings",
    "cars",
    "court",
    "dock",
    "mobile home",
    "ship",
    "tanks",
]

DLRSD_STUFF_CLASSES = [
    "bare soil",
    "chaparral",
    "field",
    "grass",
    "pavement",
    "sand",
    "sea",
    "trees",
    "water",
]

DLRSD_ALL_CLASSES = [
    "airplane",
    "bare soil",
    "buildings",
    "cars",
    "chaparral",
    "court",
    "dock",
    "field",
    "grass",
    "mobile home",
    "pavement",
    "sand",
    "sea",
    "ship",
    "tanks",
    "trees",
    "water",
]


def norm01_torch(x, eps=1e-6):
    x = x.float()
    return (x - x.amin()) / (x.amax() - x.amin() + eps)


def normalize_class_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def parse_class_list(s):
    if s is None or str(s).strip() == "":
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def get_preset_classes(preset):
    if preset == "thing":
        return DLRSD_THING_CLASSES
    if preset == "stuff":
        return DLRSD_STUFF_CLASSES
    if preset == "all":
        return DLRSD_ALL_CLASSES
    return []


def find_class_id(class_names, query):
    query_norm = normalize_class_name(query)

    for i, name in enumerate(class_names):
        if normalize_class_name(name) == query_norm:
            return i

    for i, name in enumerate(class_names):
        name_norm = normalize_class_name(name)
        if query_norm in name_norm or name_norm in query_norm:
            return i

    return None


def infer_dlrsd_class_from_filename(file_name, candidate_classes=None):
    if candidate_classes is None:
        candidate_classes = DLRSD_ALL_CLASSES

    base = os.path.basename(str(file_name))
    stem = os.path.splitext(base)[0]
    stem_norm = normalize_class_name(stem)

    sorted_classes = sorted(
        candidate_classes,
        key=lambda x: len(normalize_class_name(x)),
        reverse=True,
    )

    for cls_name in sorted_classes:
        cls_norm = normalize_class_name(cls_name)
        if stem_norm.startswith(cls_norm):
            return cls_name

    return None


def resize_score_to_target(score, target_hw, mode="bilinear"):
    """
    score: Tensor, shape (H, W)
    target_hw: (H_target, W_target)
    """
    score = score.float()[None, None]
    if mode == "nearest":
        out = F.interpolate(score, size=target_hw, mode="nearest")
    else:
        out = F.interpolate(score, size=target_hw, mode="bilinear", align_corners=False)
    return out[0, 0]


def compute_gap(score, gt_mask, valid_mask, eps=1e-6):
    """
    score: Tensor, shape (H, W), normalized to [0, 1]
    gt_mask: bool Tensor, foreground of current class
    valid_mask: bool Tensor, non-ignore region
    """
    fg = gt_mask & valid_mask
    bg = (~gt_mask) & valid_mask

    fg_n = int(fg.sum().item())
    bg_n = int(bg.sum().item())

    if fg_n == 0 or bg_n == 0:
        return None

    inside = score[fg].mean().item()
    outside = score[bg].mean().item()
    gap = inside - outside
    ratio = inside / (outside + eps)

    return {
        "inside": inside,
        "outside": outside,
        "gap": gap,
        "ratio": ratio,
        "fg_pixels": fg_n,
        "bg_pixels": bg_n,
    }


def analyze_one_file(
    pt_path,
    target_classes,
    file_class_only=False,
    ignore_value=255,
    min_fg_pixels=10,
):
    item = torch.load(pt_path, map_location="cpu")

    if item.get("target", None) is None:
        print(f"[WARN] skip no target: {pt_path}")
        return []

    target = item["target"].long()
    corr = item["corr"].float()      # (T, Hc, Wc)
    lazy = item["lazy"].float()      # (Hc, Wc)
    class_names = [str(x) for x in item["class_names"]]

    file_name = item.get("file_name", os.path.basename(pt_path))

    Ht, Wt = target.shape[-2], target.shape[-1]

    lazy_up = resize_score_to_target(lazy, (Ht, Wt), mode="bilinear")
    lazy_up = norm01_torch(lazy_up)

    valid_mask = target != ignore_value

    # 决定分析哪些类别
    if file_class_only:
        file_cls = infer_dlrsd_class_from_filename(
            file_name,
            candidate_classes=target_classes if len(target_classes) > 0 else DLRSD_ALL_CLASSES,
        )
        if file_cls is None:
            return []
        class_list = [file_cls]
    else:
        class_list = target_classes

    rows = []

    for cls_name in class_list:
        cid = find_class_id(class_names, cls_name)
        if cid is None:
            continue

        gt_mask = target == cid
        fg_pixels = int((gt_mask & valid_mask).sum().item())

        if fg_pixels < min_fg_pixels:
            continue

        raw = corr[cid]
        raw_up = resize_score_to_target(raw, (Ht, Wt), mode="bilinear")
        raw_up = norm01_torch(raw_up)

        # 与当前可视化脚本保持一致：Lazy × text = norm01(lazy_up * raw_cost_up)
        lazy_text = norm01_torch(lazy_up * raw_up)

        # 也额外统计 lazy 本身的目标区分度
        raw_metric = compute_gap(raw_up, gt_mask, valid_mask)
        lazy_metric = compute_gap(lazy_up, gt_mask, valid_mask)
        lazy_text_metric = compute_gap(lazy_text, gt_mask, valid_mask)

        if raw_metric is None or lazy_metric is None or lazy_text_metric is None:
            continue

        row = {
            "file": str(file_name),
            "pt_path": pt_path,
            "class": cls_name,
            "class_id": cid,
            "fg_pixels": fg_pixels,

            "raw_inside": raw_metric["inside"],
            "raw_outside": raw_metric["outside"],
            "raw_gap": raw_metric["gap"],
            "raw_ratio": raw_metric["ratio"],

            "lazy_inside": lazy_metric["inside"],
            "lazy_outside": lazy_metric["outside"],
            "lazy_gap": lazy_metric["gap"],
            "lazy_ratio": lazy_metric["ratio"],

            "lazy_text_inside": lazy_text_metric["inside"],
            "lazy_text_outside": lazy_text_metric["outside"],
            "lazy_text_gap": lazy_text_metric["gap"],
            "lazy_text_ratio": lazy_text_metric["ratio"],

            "delta_gap": lazy_text_metric["gap"] - raw_metric["gap"],
            "lazy_better": int(lazy_text_metric["gap"] > raw_metric["gap"]),
        }

        rows.append(row)

    return rows


def mean(xs):
    if len(xs) == 0:
        return float("nan")
    return float(np.mean(xs))


def summarize_by_class(rows):
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["class"]].append(r)

    summary = []

    for cls_name, rs in sorted(buckets.items()):
        n = len(rs)

        raw_gap = [r["raw_gap"] for r in rs]
        lazy_gap = [r["lazy_gap"] for r in rs]
        lazy_text_gap = [r["lazy_text_gap"] for r in rs]
        delta_gap = [r["delta_gap"] for r in rs]
        lazy_better = [r["lazy_better"] for r in rs]

        summary.append({
            "class": cls_name,
            "n": n,
            "raw_gap_mean": mean(raw_gap),
            "lazy_gap_mean": mean(lazy_gap),
            "lazy_text_gap_mean": mean(lazy_text_gap),
            "delta_gap_mean": mean(delta_gap),
            "lazy_better_rate": mean(lazy_better),
            "raw_inside_mean": mean([r["raw_inside"] for r in rs]),
            "raw_outside_mean": mean([r["raw_outside"] for r in rs]),
            "lazy_text_inside_mean": mean([r["lazy_text_inside"] for r in rs]),
            "lazy_text_outside_mean": mean([r["lazy_text_outside"] for r in rs]),
        })

    return summary


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def print_summary(summary):
    print("\n========== Summary by class ==========")
    print(
        f"{'class':16s} {'n':>5s} "
        f"{'raw_gap':>10s} {'lazy_gap':>10s} {'lazy_text':>12s} "
        f"{'delta':>10s} {'better%':>9s}"
    )

    for r in summary:
        print(
            f"{r['class']:16s} {int(r['n']):5d} "
            f"{r['raw_gap_mean']:10.4f} "
            f"{r['lazy_gap_mean']:10.4f} "
            f"{r['lazy_text_gap_mean']:12.4f} "
            f"{r['delta_gap_mean']:10.4f} "
            f"{100.0 * r['lazy_better_rate']:8.1f}%"
        )

    all_delta = [r["delta_gap_mean"] for r in summary if not np.isnan(r["delta_gap_mean"])]
    if len(all_delta) > 0:
        print("\n========== Overall ==========")
        print(f"mean class delta_gap = {np.mean(all_delta):.4f}")
        print("delta_gap = lazy_text_gap - raw_gap")
        print("delta_gap < 0 表示 Lazy × text 的前景/背景区分度弱于 Raw cost")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dump", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument(
        "--preset",
        default="all",
        choices=["all", "thing", "stuff", "custom"],
        help="分析 DLRSD 全部类别、thing 类、stuff 类，或自定义类别。",
    )

    parser.add_argument(
        "--classes",
        default="",
        help="preset=custom 时使用，逗号分隔类别名。",
    )

    parser.add_argument(
        "--file-class-only",
        action="store_true",
        help="每张图只分析文件名前缀对应的类别。DLRSD 推荐打开。",
    )

    parser.add_argument("--ignore-value", type=int, default=255)
    parser.add_argument("--min-fg-pixels", type=int, default=10)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.preset == "custom":
        target_classes = parse_class_list(args.classes)
    else:
        target_classes = get_preset_classes(args.preset)

    if len(target_classes) == 0:
        raise ValueError("target_classes is empty")

    pt_files = sorted(glob.glob(os.path.join(args.dump, "*.pt")))

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(pt_files)

    if args.max_files and args.max_files > 0:
        pt_files = pt_files[: args.max_files]

    print(f"[INFO] found {len(pt_files)} dump files")
    print(f"[INFO] target classes: {target_classes}")
    print(f"[INFO] file_class_only: {args.file_class_only}")

    all_rows = []

    for i, p in enumerate(pt_files):
        rows = analyze_one_file(
            p,
            target_classes=target_classes,
            file_class_only=args.file_class_only,
            ignore_value=args.ignore_value,
            min_fg_pixels=args.min_fg_pixels,
        )

        all_rows.extend(rows)

        if (i + 1) % 500 == 0:
            print(f"[INFO] processed {i+1}/{len(pt_files)}, valid rows={len(all_rows)}")

    if len(all_rows) == 0:
        raise RuntimeError(
            "没有得到有效统计结果。请检查 target 是否存在、类别名是否匹配、min_fg_pixels 是否过大。"
        )

    summary = summarize_by_class(all_rows)

    os.makedirs(args.out, exist_ok=True)

    detail_fields = [
        "file",
        "pt_path",
        "class",
        "class_id",
        "fg_pixels",

        "raw_inside",
        "raw_outside",
        "raw_gap",
        "raw_ratio",

        "lazy_inside",
        "lazy_outside",
        "lazy_gap",
        "lazy_ratio",

        "lazy_text_inside",
        "lazy_text_outside",
        "lazy_text_gap",
        "lazy_text_ratio",

        "delta_gap",
        "lazy_better",
    ]

    summary_fields = [
        "class",
        "n",
        "raw_gap_mean",
        "lazy_gap_mean",
        "lazy_text_gap_mean",
        "delta_gap_mean",
        "lazy_better_rate",
        "raw_inside_mean",
        "raw_outside_mean",
        "lazy_text_inside_mean",
        "lazy_text_outside_mean",
    ]

    write_csv(
        os.path.join(args.out, "detail_per_image.csv"),
        all_rows,
        detail_fields,
    )

    write_csv(
        os.path.join(args.out, "summary_by_class.csv"),
        summary,
        summary_fields,
    )

    print_summary(summary)

    print(f"\n[DONE] detail saved to: {os.path.join(args.out, 'detail_per_image.csv')}")
    print(f"[DONE] summary saved to: {os.path.join(args.out, 'summary_by_class.csv')}")


if __name__ == "__main__":
    main()