"""Scan the image folders and build data/metadata.csv with leakage-safe splits.

Expected folder layout (you fill these in Step 2):

    data/real/<any_name>.jpg                       -> label 0
    data/fake/<generator>/<any_name>.png           -> label 1, fully generated
    data/fake/<generator>_inpaint/<real_stem>__<anything>.png
                                                   -> label 1, inpainted edit of a real photo
    data/masks/<same filename as the inpainted image>  (optional, used in Step 10)

Key ideas you should be able to explain in an interview:

1. GROUPS: an inpainted image is a copy of a real photo with one region changed.
   If the real photo is in train and its edited copy is in test, the model has
   effectively "seen" the test image. So each real photo and all its edits share
   a group_id and always land in the same split.

2. HOLD-OUT GENERATOR: every image from one generator (config: holdout_generator)
   goes into a separate split "test_holdout". The model never trains on it, so it
   measures whether we detect AI images in general or just memorise one generator.

Run:  python -m src.build_metadata
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.utils import PROJECT_ROOT, load_config, set_seed

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _image_info(path: Path) -> dict:
    """Read basic properties without loading full pixels (used later in EDA)."""
    with Image.open(path) as img:
        width, height = img.size
        fmt = (img.format or path.suffix.lstrip(".")).lower()
        has_exif = bool(img.getexif())
    return {"width": width, "height": height, "file_format": fmt, "has_exif": has_exif}


def scan_images(cfg: dict) -> pd.DataFrame:
    root = PROJECT_ROOT
    real_dir = root / cfg["paths"]["real_dir"]
    fake_dir = root / cfg["paths"]["fake_dir"]
    rows = []

    # Real photos: each one starts its own group.
    for p in sorted(real_dir.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTS:
            rows.append({
                "path": p.relative_to(root).as_posix(), "label": 0,
                "source_type": "real", "generator": "none",
                "group_id": f"real::{p.stem}",
            })

    # Fakes: one subfolder per generator.
    if fake_dir.exists():
        for gen_dir in sorted(d for d in fake_dir.iterdir() if d.is_dir()):
            generator = gen_dir.name.lower()
            inpainted = "inpaint" in generator
            for p in sorted(gen_dir.rglob("*")):
                if p.suffix.lower() not in IMAGE_EXTS:
                    continue
                if inpainted:
                    # Filename convention <real_stem>__<suffix> ties the edit to its source photo.
                    source_stem = p.stem.split("__")[0]
                    group_id = f"real::{source_stem}"
                else:
                    group_id = f"{generator}::{p.stem}"
                rows.append({
                    "path": p.relative_to(root).as_posix(), "label": 1,
                    "source_type": "inpainted" if inpainted else "generated",
                    "generator": generator, "group_id": group_id,
                })

    if not rows:
        raise SystemExit(
            f"No images found under {real_dir} or {fake_dir}. Add data first (Step 2).")

    df = pd.DataFrame(rows)
    info = pd.DataFrame([_image_info(root / p) for p in df["path"]])
    df = pd.concat([df, info], axis=1)
    df.insert(0, "image_id", [f"img_{i:06d}" for i in range(len(df))])
    return df


def assign_splits(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Split by GROUP, not by image, and carve out the hold-out generator."""
    s = cfg["splits"]
    holdout = str(s.get("holdout_generator") or "").lower()
    rng = np.random.default_rng(cfg["seed"])

    df = df.copy()
    # A group is held out if it contains any image from the hold-out generator
    # (covers e.g. "flux" and "flux_inpaint").
    is_holdout_img = df["generator"].str.startswith(
        holdout) if holdout else pd.Series(False, index=df.index)
    holdout_groups = set(df.loc[is_holdout_img, "group_id"])

    split_of_group: dict[str, str] = {}
    # Assign remaining groups separately for real-rooted and fake-only groups,
    # which keeps the label ratio roughly equal across train/val/test (stratification).
    remaining = df[~df["group_id"].isin(holdout_groups)]
    for _, grp in remaining.groupby(remaining["group_id"].str.startswith("real::")):
        # sort first so the shuffle is reproducible
        groups = sorted(grp["group_id"].unique())
        rng.shuffle(groups)
        n = len(groups)
        n_train = int(round(n * s["train"]))
        n_val = int(round(n * s["val"]))
        for i, g in enumerate(groups):
            split_of_group[g] = "train" if i < n_train else "val" if i < n_train + \
                n_val else "test"

    df["split"] = df["group_id"].map(split_of_group).fillna("test_holdout")
    # The real photo behind a held-out inpaint goes to test_holdout as well (same group).
    return df


def check_no_leakage(df: pd.DataFrame) -> None:
    """Fail loudly if any group appears in more than one split."""
    splits_per_group = df.groupby("group_id")["split"].nunique()
    leaked = splits_per_group[splits_per_group > 1]
    assert leaked.empty, f"Leakage: {len(leaked)} groups span several splits, e.g. {list(leaked.index[:3])}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    df = assign_splits(scan_images(cfg), cfg)
    check_no_leakage(df)

    out = PROJECT_ROOT / cfg["paths"]["metadata_csv"]
    df.to_csv(out, index=False)

    print(f"Saved {len(df)} rows to {out.relative_to(PROJECT_ROOT)}\n")
    print("Images per split and label (0 = real, 1 = fake):")
    print(pd.crosstab(df["split"], df["label"], margins=True), "\n")
    print("Images per split and generator:")
    print(pd.crosstab(df["generator"], df["split"]))


if __name__ == "__main__":
    main()
