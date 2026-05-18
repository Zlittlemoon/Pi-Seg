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


def norm01_torch(x, valid_mask=None, eps=1e-6):
    x = x.float()

    if valid_mask is not None:
        valid_vals = x[valid_mask]
        if valid_vals.numel() == 0:
            return torch.zeros_like(x)
        x_min = valid_vals.amin()
        x_max = valid_vals.amax()
    else:
        x_min = x.amin()
        x_max = x.amax()

    return (x - x_min) / (x_max - x_min + eps)


def resize_score_to_target(score, target_hw, mode="bilinear"):
    score = score.float()[None, None]
    if mode == "nearest":
        out = F.interpolate(score, size=target_hw, mode="nearest")
    else:
        out = F.interpolate(score, size=target_hw, mode="bilinear", align_corners=False)
    return out[0, 0]


def compute_gap(score, gt_mask, valid_mask, eps=1e-6):
    fg = gt_mask & valid_mask
    bg = (~gt_mask) & valid_mask

    if int(fg.sum().item()) == 0 or int(bg.sum().item()) == 0:
        return None

    inside = score[fg].mean().item()
    outside = score[bg].mean().item()

    return {
        "inside": inside,
        "outside": outside,
        "gap": inside - outside,
        "ratio": inside / (outside + eps),
    }


def compute_best_iou(score, gt_mask, valid_mask, thresholds):
    """
    score: normalized score map, shape (H, W)
    gt_mask: bool, current class mask
    valid_mask: bool, non-ignore region
    """
    gt = gt_mask & valid_mask

    best_iou = 0.0
    best_thr = 0.0

    for thr in thresholds:
        pred = (score >= thr) & valid_mask

        inter = (pred & gt).sum().item()
        union = (pred | gt).sum().item()

        if union == 0:
            iou = 0.0
        else:
            iou = inter / union

        if iou > best_iou:
            best_iou = iou
            best_thr = thr

    return best_iou, best_thr


def make_score_variants(raw, lazy, valid_mask):
    """
    raw/lazy: normalized to [0, 1], shape (H, W)

    Variants:
      raw:
        原始 Pi-Seg cost map

      raw_x_lazy:
        你现在验证过的直接乘法版本，通常会变差

      raw_res_c005/c010/c020:
        centered residual gate，LazyScore 只做弱残差调制

      raw_unc_u010/u020:
        uncertainty-aware gate，只在 raw 不够确定的位置加 LazyScore
    """
    lazy_mean = lazy[valid_mask].mean() if valid_mask.sum() > 0 else lazy.mean()
    lazy_centered = lazy - lazy_mean

    variants = {}

    variants["raw"] = raw

    # 原始直接乘法
    variants["raw_x_lazy"] = norm01_torch(raw * lazy, valid_mask)

    # 弱 residual，避免破坏 Raw cost
    for lam in [0.05, 0.10, 0.20]:
        score = raw * (1.0 + lam * lazy_centered)
        variants[f"raw_res_c{int(lam * 1000):03d}"] = norm01_torch(score, valid_mask)

    # 只在 raw 不确定区域使用 LazyScore
    # uncertainty = 1 - raw，raw 高的地方基本不动，raw 低/中等的地方略增强
    uncertainty = 1.0 - raw
    for lam in [0.10, 0.20]:
        score = raw + lam * uncertainty * raw * lazy
        variants[f"raw_unc_u{int(lam * 1000):03d}"] = norm01_torch(score, valid_mask)

    # 纯 add residual，作为补充
    for lam in [0.05, 0.10]:
        score = raw + lam * raw * lazy
        variants[f"raw_add_l{int(lam * 1000):03d}"] = norm01_torch(score, valid_mask)

    return variants


def analyze_one_file(
    pt_path,
    target_classes,
    file_class_only=False,
    ignore_value=255,
    min_fg_pixels=20,
    thresholds=None,
):
    if thresholds is None:
        thresholds = [i / 100.0 for i in range(5, 96, 5)]

    item = torch.load(pt_path, map_location="cpu")

    if item.get("target", None) is None:
        return []

    target = item["target"].long()
    corr = item["corr"].float()      # (T, Hc, Wc)
    lazy = item["lazy"].float()      # (Hc, Wc)
    class_names = [str(x) for x in item["class_names"]]

    file_name = item.get("file_name", os.path.basename(pt_path))

    Ht, Wt = target.shape[-2], target.shape[-1]
    valid_mask = target != ignore_value

    lazy_up = resize_score_to_target(lazy, (Ht, Wt), mode="bilinear")
    lazy_up = norm01_torch(lazy_up, valid_mask)

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
        raw_up = norm01_torch(raw_up, valid_mask)

        variants = make_score_variants(raw_up, lazy_up, valid_mask)

        # raw baseline metrics
        raw_gap_metric = compute_gap(variants["raw"], gt_mask, valid_mask)
        raw_iou, raw_thr = compute_best_iou(
            variants["raw"], gt_mask, valid_mask, thresholds
        )

        if raw_gap_metric is None:
            continue

        for vname, score in variants.items():
            gap_metric = compute_gap(score, gt_mask, valid_mask)
            if gap_metric is None:
                continue

            best_iou, best_thr = compute_best_iou(
                score, gt_mask, valid_mask, thresholds
            )

            row = {
                "file": str(file_name),
                "pt_path": pt_path,
                "class": cls_name,
                "class_id": cid,
                "variant": vname,
                "fg_pixels": fg_pixels,

                "inside": gap_metric["inside"],
                "outside": gap_metric["outside"],
                "gap": gap_metric["gap"],
                "ratio": gap_metric["ratio"],

                "best_iou": best_iou,
                "best_thr": best_thr,

                "raw_gap": raw_gap_metric["gap"],
                "raw_best_iou": raw_iou,
                "raw_best_thr": raw_thr,

                "delta_gap": gap_metric["gap"] - raw_gap_metric["gap"],
                "delta_iou": best_iou - raw_iou,

                "better_gap": int(gap_metric["gap"] > raw_gap_metric["gap"]),
                "better_iou": int(best_iou > raw_iou),
            }

            rows.append(row)

    return rows


def mean(xs):
    if len(xs) == 0:
        return float("nan")
    return float(np.mean(xs))


def summarize(rows, group_keys):
    buckets = defaultdict(list)

    for r in rows:
        key = tuple(r[k] for k in group_keys)
        buckets[key].append(r)

    summary = []

    for key, rs in sorted(buckets.items()):
        out = {}
        for k, v in zip(group_keys, key):
            out[k] = v

        out["n"] = len(rs)
        out["gap_mean"] = mean([r["gap"] for r in rs])
        out["raw_gap_mean"] = mean([r["raw_gap"] for r in rs])
        out["delta_gap_mean"] = mean([r["delta_gap"] for r in rs])
        out["better_gap_rate"] = mean([r["better_gap"] for r in rs])

        out["best_iou_mean"] = mean([r["best_iou"] for r in rs])
        out["raw_best_iou_mean"] = mean([r["raw_best_iou"] for r in rs])
        out["delta_iou_mean"] = mean([r["delta_iou"] for r in rs])
        out["better_iou_rate"] = mean([r["better_iou"] for r in rs])

        out["inside_mean"] = mean([r["inside"] for r in rs])
        out["outside_mean"] = mean([r["outside"] for r in rs])

        summary.append(out)

    return summary


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def print_overall(summary):
    print("\n========== Overall by variant ==========")
    print(
        f"{'variant':16s} {'n':>7s} "
        f"{'gap':>9s} {'d_gap':>9s} {'betterG%':>9s} "
        f"{'bestIoU':>9s} {'d_iou':>9s} {'betterI%':>9s}"
    )

    # raw 放第一，其它按 delta_gap 排序
    raw_rows = [r for r in summary if r["variant"] == "raw"]
    other_rows = [r for r in summary if r["variant"] != "raw"]
    other_rows = sorted(other_rows, key=lambda x: x["delta_gap_mean"], reverse=True)

    for r in raw_rows + other_rows:
        print(
            f"{r['variant']:16s} {int(r['n']):7d} "
            f"{r['gap_mean']:9.4f} "
            f"{r['delta_gap_mean']:9.4f} "
            f"{100.0 * r['better_gap_rate']:8.1f}% "
            f"{r['best_iou_mean']:9.4f} "
            f"{r['delta_iou_mean']:9.4f} "
            f"{100.0 * r['better_iou_rate']:8.1f}%"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dump", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument(
        "--preset",
        default="all",
        choices=["all", "thing", "stuff", "custom"],
    )

    parser.add_argument(
        "--classes",
        default="",
        help="preset=custom 时使用，逗号分隔类别名。",
    )

    parser.add_argument(
        "--file-class-only",
        action="store_true",
        help="DLRSD 推荐打开：每张图只分析文件名前缀对应的类别。",
    )

    parser.add_argument("--ignore-value", type=int, default=255)
    parser.add_argument("--min-fg-pixels", type=int, default=20)

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
        raise RuntimeError("没有有效统计结果，请检查 dump/target/classes/min_fg_pixels。")

    os.makedirs(args.out, exist_ok=True)

    detail_fields = [
        "file",
        "pt_path",
        "class",
        "class_id",
        "variant",
        "fg_pixels",
        "inside",
        "outside",
        "gap",
        "ratio",
        "best_iou",
        "best_thr",
        "raw_gap",
        "raw_best_iou",
        "raw_best_thr",
        "delta_gap",
        "delta_iou",
        "better_gap",
        "better_iou",
    ]

    summary_class_variant = summarize(all_rows, ["class", "variant"])
    summary_variant = summarize(all_rows, ["variant"])

    summary_fields_class_variant = [
        "class",
        "variant",
        "n",
        "gap_mean",
        "raw_gap_mean",
        "delta_gap_mean",
        "better_gap_rate",
        "best_iou_mean",
        "raw_best_iou_mean",
        "delta_iou_mean",
        "better_iou_rate",
        "inside_mean",
        "outside_mean",
    ]

    summary_fields_variant = [
        "variant",
        "n",
        "gap_mean",
        "raw_gap_mean",
        "delta_gap_mean",
        "better_gap_rate",
        "best_iou_mean",
        "raw_best_iou_mean",
        "delta_iou_mean",
        "better_iou_rate",
        "inside_mean",
        "outside_mean",
    ]

    write_csv(
        os.path.join(args.out, "detail_per_image_variant.csv"),
        all_rows,
        detail_fields,
    )

    write_csv(
        os.path.join(args.out, "summary_by_class_variant.csv"),
        summary_class_variant,
        summary_fields_class_variant,
    )

    write_csv(
        os.path.join(args.out, "overall_by_variant.csv"),
        summary_variant,
        summary_fields_variant,
    )

    print_overall(summary_variant)

    print(f"\n[DONE] saved detail: {os.path.join(args.out, 'detail_per_image_variant.csv')}")
    print(f"[DONE] saved class summary: {os.path.join(args.out, 'summary_by_class_variant.csv')}")
    print(f"[DONE] saved overall summary: {os.path.join(args.out, 'overall_by_variant.csv')}")


if __name__ == "__main__":
    main()