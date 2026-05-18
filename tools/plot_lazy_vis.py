import os
import argparse
import glob
import re
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# DLRSD class presets
# ============================================================

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

DEFAULT_DLRSD_CLASSES_CSV = ",".join(DLRSD_ALL_CLASSES)


# ============================================================
# Basic utils
# ============================================================

def norm01(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.min()) / (x.max() - x.min() + eps)


def normalize_class_name(name):
    """
    用于类别名和文件名前缀匹配。

    Examples:
        'Bare Soil'    -> 'baresoil'
        'bare soil'    -> 'baresoil'
        'MobileHome00' -> 'mobilehome00'
        'Airplane00'   -> 'airplane00'
    """
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def parse_class_list(s):
    if s is None:
        return []
    s = str(s).strip()
    if s == "":
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def get_preset_classes(preset):
    if preset == "thing":
        return DLRSD_THING_CLASSES
    if preset == "stuff":
        return DLRSD_STUFF_CLASSES
    if preset == "all":
        return DLRSD_ALL_CLASSES
    return []


def tensor_image_to_rgb(img):
    """
    Pi-Seg / Detectron2 里的 image 通常是 CHW, BGR, 0-255。
    """
    if img is None:
        return None

    img = img.detach().cpu().float()

    if img.dim() == 3:
        img = img.permute(1, 2, 0).numpy()
    elif img.dim() == 2:
        img = img.numpy()
    else:
        raise ValueError(f"Unsupported image shape: {tuple(img.shape)}")

    if img.max() <= 2.0:
        img = norm01(img) * 255.0

    img = np.clip(img, 0, 255).astype(np.uint8)

    # CHW -> HWC 后，如果是 3 通道，Detectron2 通常是 BGR，所以转 RGB
    if img.ndim == 3 and img.shape[-1] == 3:
        img = img[..., ::-1]

    return img


def resize_map(m, size, mode="bilinear"):
    """
    Args:
        m: Tensor or numpy, shape (H, W) / (1, H, W)
        size: (H_img, W_img)
    """
    if not torch.is_tensor(m):
        m = torch.from_numpy(np.asarray(m))

    m = m.detach().cpu()

    if m.dim() == 3 and m.shape[0] == 1:
        m = m[0]
    elif m.dim() == 3 and m.shape[-1] == 1:
        m = m[..., 0]

    if m.dim() != 2:
        raise ValueError(f"resize_map expects 2D map, got shape={tuple(m.shape)}")

    m = m.float()[None, None]

    if mode == "nearest":
        out = F.interpolate(m, size=size, mode="nearest")
    else:
        out = F.interpolate(m, size=size, mode="bilinear", align_corners=False)

    return out[0, 0].cpu().numpy()


def safe_torch_load(path):
    return torch.load(path, map_location="cpu")


# ============================================================
# DLRSD filename-prefix sampling
# ============================================================

def infer_dlrsd_class_from_filename(file_name, candidate_classes=None):
    """
    根据 DLRSD 文件名前缀推断类别。

    Examples:
        Airplane00.jpg   -> airplane
        BareSoil03.jpg   -> bare soil
        MobileHome12.jpg -> mobile home
        Buildings08.jpg  -> buildings
        Ship01.jpg       -> ship
    """
    if candidate_classes is None:
        candidate_classes = DLRSD_ALL_CLASSES

    base = os.path.basename(str(file_name))

    # 如果 dump 文件名类似 Airplane00.jpg_lazy_00000.pt，也没关系
    stem = os.path.splitext(base)[0]
    stem_norm = normalize_class_name(stem)

    # 长类别优先，避免 mobile home 这类被错误短匹配
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


def load_dump_file_name(pt_path):
    """
    只读取 dump 里的 file_name，用来按类别抽样。
    如果 dump 里没有 file_name，则退回到 .pt 文件名。
    """
    try:
        item = safe_torch_load(pt_path)
        return item.get("file_name", os.path.basename(pt_path))
    except Exception:
        return os.path.basename(pt_path)


def select_files_by_dlrsd_prefix(
    pt_files,
    file_classes,
    samples_per_class=5,
    max_files=50,
    seed=42,
):
    """
    根据 DLRSD 文件名前缀，每个类别随机抽 samples_per_class 张。
    """
    rng = random.Random(seed)
    buckets = defaultdict(list)

    for p in pt_files:
        file_name = load_dump_file_name(p)
        cls_name = infer_dlrsd_class_from_filename(
            file_name,
            candidate_classes=file_classes,
        )

        if cls_name is not None:
            buckets[cls_name].append(p)

    selected = []

    print("[INFO] DLRSD filename-prefix buckets:")
    for cls_name in file_classes:
        files = buckets.get(cls_name, [])
        print(f"  {cls_name:12s}: {len(files)}")

        if len(files) == 0:
            continue

        rng.shuffle(files)
        selected.extend(files[:samples_per_class])

    rng.shuffle(selected)

    if max_files is not None and max_files > 0:
        selected = selected[:max_files]

    return selected


# ============================================================
# Class selection
# ============================================================

def find_class_ids(class_names, focus_classes):
    ids = []

    raw_names = [str(x) for x in class_names]
    lower_names = [x.lower() for x in raw_names]
    norm_names = [normalize_class_name(x) for x in raw_names]

    for q in focus_classes:
        q = q.strip()
        if not q:
            continue

        q_lower = q.lower()
        q_norm = normalize_class_name(q)

        hit = None

        # 1. 精确 lower 匹配
        for i, name in enumerate(lower_names):
            if name == q_lower:
                hit = i
                break

        # 2. 归一化精确匹配，比如 bare soil / BareSoil
        if hit is None:
            for i, name in enumerate(norm_names):
                if name == q_norm:
                    hit = i
                    break

        # 3. 子串匹配，兼容 building/buildings, car/cars 等
        if hit is None:
            for i, name in enumerate(norm_names):
                if q_norm in name or name in q_norm:
                    hit = i
                    break

        if hit is not None and hit not in ids:
            ids.append(hit)

    return ids


def select_classes(item, focus_classes, max_classes=6, ignore_value=255):
    class_names = item["class_names"]
    ids = find_class_ids(class_names, focus_classes)

    target = item.get("target", None)

    # 如果指定类别找不到，就从 GT 里自动选当前图出现的类别
    if len(ids) == 0 and target is not None:
        vals = torch.unique(target)
        vals = [int(v.item()) for v in vals if int(v.item()) != ignore_value]
        ids = [v for v in vals if 0 <= v < len(class_names)]

    # 如果还没有，就从预测里选 top 类
    if len(ids) == 0:
        pred = item["pred"]  # (T, H, W)
        score = pred.flatten(1).mean(dim=1)
        ids = score.topk(min(max_classes, score.numel())).indices.tolist()

    return ids[:max_classes]


# ============================================================
# Plot
# ============================================================

def plot_one(
    item,
    save_path,
    focus_classes,
    max_classes=6,
    ignore_value=255,
    title_prefix="",
):
    image = tensor_image_to_rgb(item.get("image", None))
    if image is None:
        raise ValueError(
            "Dump 里没有 image，不能画原图。"
            "请确认 dump 时 TEST.SLIDING_WINDOW=False，且 sem_seg_head 接收了 input_images。"
        )

    H_img, W_img = image.shape[:2]

    target = item.get("target", None)
    corr = item["corr"]      # (T, Hc, Wc)
    lazy = item["lazy"]      # (Hc, Wc)
    pred = item["pred"]      # (T, Hp, Wp)
    class_names = item["class_names"]

    lazy_up = resize_map(lazy, (H_img, W_img))
    lazy_up = norm01(lazy_up)

    # final prediction class map
    pred_up = F.interpolate(
        pred[None].float(),
        size=(H_img, W_img),
        mode="bilinear",
        align_corners=False,
    )[0]
    pred_cls = pred_up.argmax(dim=0).cpu().numpy()

    if target is not None:
        gt = target.cpu()
        gt_np = resize_map(gt, (H_img, W_img), mode="nearest").astype(np.int64)
    else:
        gt_np = None

    class_ids = select_classes(
        item,
        focus_classes=focus_classes,
        max_classes=max_classes,
        ignore_value=ignore_value,
    )

    if len(class_ids) == 0:
        print(f"[WARN] no valid class found for {item.get('file_name', '')}")
        return False

    nrows = len(class_ids)
    ncols = 6

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 3.0 * max(1, nrows)),
        squeeze=False,
    )

    file_name = str(item.get("file_name", "unknown"))

    for row, cid in enumerate(class_ids):
        cname = str(class_names[cid])

        raw_cost = corr[cid]
        raw_cost_up = resize_map(raw_cost, (H_img, W_img))
        raw_cost_up = norm01(raw_cost_up)

        lazy_x_text = norm01(lazy_up * raw_cost_up)

        pred_bin = (pred_cls == cid).astype(np.float32)

        if gt_np is not None:
            gt_bin = (gt_np == cid).astype(np.float32)
        else:
            gt_bin = np.zeros((H_img, W_img), dtype=np.float32)

        axes[row, 0].imshow(image)
        axes[row, 0].set_title("Image")

        axes[row, 1].imshow(gt_bin, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(f"GT: {cname}")

        axes[row, 2].imshow(raw_cost_up, cmap="jet", vmin=0, vmax=1)
        axes[row, 2].set_title(f"Raw cost: {cname}")

        axes[row, 3].imshow(lazy_up, cmap="jet", vmin=0, vmax=1)
        axes[row, 3].set_title("LazyScore")

        axes[row, 4].imshow(lazy_x_text, cmap="jet", vmin=0, vmax=1)
        axes[row, 4].set_title(f"Lazy × text: {cname}")

        axes[row, 5].imshow(pred_bin, cmap="gray", vmin=0, vmax=1)
        axes[row, 5].set_title(f"Pred: {cname}")

        for col in range(ncols):
            axes[row, col].axis("off")

    if title_prefix:
        fig.suptitle(f"{title_prefix} | {file_name}", fontsize=12)
    else:
        fig.suptitle(file_name, fontsize=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return True


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dump", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument(
        "--classes",
        default=DEFAULT_DLRSD_CLASSES_CSV,
        help="画哪些类别。默认是 DLRSD 17 类。",
    )

    parser.add_argument("--max-files", type=int, default=50)
    parser.add_argument("--max-classes", type=int, default=6)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ignore-value", type=int, default=255)

    parser.add_argument(
        "--dlrsd-preset",
        default="",
        choices=["", "thing", "stuff", "all"],
        help="按 DLRSD 文件名前缀抽样：thing / stuff / all。",
    )

    parser.add_argument(
        "--file-classes",
        default="",
        help=(
            "用于按文件名前缀抽样的类别列表。"
            "如果为空，则使用 --dlrsd-preset。"
        ),
    )

    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=5,
        help="每个 DLRSD 类别抽多少张。",
    )

    parser.add_argument(
        "--draw-file-class-only",
        action="store_true",
        help="每张图只画文件名对应的那个类别，避免一张图画很多行。",
    )

    args = parser.parse_args()

    random.seed(args.seed)

    pt_files_all = sorted(glob.glob(os.path.join(args.dump, "*.pt")))
    print(f"[INFO] found {len(pt_files_all)} dump files in {args.dump}")

    if len(pt_files_all) == 0:
        raise FileNotFoundError(f"No .pt files found in: {args.dump}")

    # --------------------------------------------------------
    # Decide whether to use DLRSD prefix sampling
    # --------------------------------------------------------
    use_dlrsd_prefix_sampling = (
        args.dlrsd_preset != "" or args.file_classes.strip() != ""
    )

    file_classes = parse_class_list(args.file_classes)
    if len(file_classes) == 0 and args.dlrsd_preset != "":
        file_classes = get_preset_classes(args.dlrsd_preset)

    if use_dlrsd_prefix_sampling:
        if len(file_classes) == 0:
            raise ValueError(
                "使用 DLRSD 前缀抽样时，必须设置 --dlrsd-preset 或 --file-classes"
            )

        pt_files = select_files_by_dlrsd_prefix(
            pt_files_all,
            file_classes=file_classes,
            samples_per_class=args.samples_per_class,
            max_files=args.max_files,
            seed=args.seed,
        )

        # 如果用户没有显式改 --classes，则默认画当前 preset 的类别
        if args.classes.strip() == DEFAULT_DLRSD_CLASSES_CSV:
            focus_classes = file_classes
        else:
            focus_classes = parse_class_list(args.classes)

    else:
        pt_files = pt_files_all

        if args.shuffle:
            random.shuffle(pt_files)

        pt_files = pt_files[: args.max_files]
        focus_classes = parse_class_list(args.classes)

    print(f"[INFO] selected {len(pt_files)} dump files for plotting")

    if len(pt_files) == 0:
        raise RuntimeError(
            "没有选中任何文件。请检查 dump 里的 file_name 是否类似 Airplane00.jpg，"
            "或者检查 --dlrsd-preset / --file-classes 是否写对。"
        )

    os.makedirs(args.out, exist_ok=True)

    # --------------------------------------------------------
    # Plot selected files
    # --------------------------------------------------------
    success = 0

    for i, p in enumerate(pt_files):
        item = safe_torch_load(p)
        base = os.path.splitext(os.path.basename(p))[0]

        file_cls = infer_dlrsd_class_from_filename(
            item.get("file_name", base),
            candidate_classes=DLRSD_ALL_CLASSES,
        )

        if args.draw_file_class_only and file_cls is not None:
            cur_focus_classes = [file_cls]
            cur_max_classes = 1
            cls_tag = normalize_class_name(file_cls)

            save_subdir = os.path.join(args.out, cls_tag)
            os.makedirs(save_subdir, exist_ok=True)
            save_path = os.path.join(save_subdir, base + ".png")
            title_prefix = file_cls
        else:
            cur_focus_classes = focus_classes
            cur_max_classes = args.max_classes
            save_path = os.path.join(args.out, base + ".png")
            title_prefix = args.dlrsd_preset if args.dlrsd_preset else ""

        ok = plot_one(
            item,
            save_path=save_path,
            focus_classes=cur_focus_classes,
            max_classes=cur_max_classes,
            ignore_value=args.ignore_value,
            title_prefix=title_prefix,
        )

        if ok:
            success += 1
            print(f"[{i+1}/{len(pt_files)}] saved {save_path}")
        else:
            print(f"[{i+1}/{len(pt_files)}] skipped {p}")

    print(f"[DONE] saved {success}/{len(pt_files)} visualizations to {args.out}")


if __name__ == "__main__":
    main()