"""Step 2.2: check every real photo and write data/real_manifest.csv.

The manifest is an auditable list of ALL candidate real photos, one row each, saying
whether it is kept and, if not, why. Images are NOT copied here: on Kaggle the original
datasets stay attached as inputs, which avoids duplicating ~15 GB. Copying the chosen
photos happens once, when the final dataset is packaged (Step 2.5).

What it checks, and why (interview points):
1. Corrupt files     -> a broken file would crash training later.
2. Too small         -> shortest side < min_side has too little pixel detail.
3. Near-duplicates   -> the same photo twice could land in train AND test (leakage).
                        Perceptual hashes (pHash) of similar images differ in only a few
                        bits, so we compare Hamming distance instead of exact equality.
4. Inpaint sources   -> large photos from approved sources are flagged for Step 2.4.

Run on Kaggle (sources configured in configs/config.yaml):
    python -m src.clean_real
Run anywhere else by pointing at other folders:
    python -m src.clean_real --source vehide=path/to/a --source hitl=path/to/b
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from src.utils import PROJECT_ROOT, load_config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_photos(root: Path, exclude: list[str]) -> list[Path]:
    """All image files under root, skipping folders whose path contains an excluded word."""
    files = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTS and not any(word in str(p.parent) for word in exclude):
            files.append(p)
    return files


def inspect(path: Path) -> dict:
    """Open one photo: size, format, EXIF and perceptual hash. Never raises."""
    try:
        with Image.open(path) as img:
            width, height = img.size
            fmt = (img.format or "").lower()
            has_exif = bool(img.getexif())
            # draft() lets JPEG decode at reduced scale: much faster, and pHash
            # shrinks the image to 32x32 anyway, so nothing is lost.
            img.draft("RGB", (256, 256))
            phash = str(imagehash.phash(img.convert("RGB")))
        return {"width": width, "height": height, "format": fmt,
                "has_exif": has_exif, "phash": phash, "error": ""}
    except Exception as e:  # corrupt / truncated / unreadable file
        return {"width": 0, "height": 0, "format": "", "has_exif": False,
                "phash": "", "error": f"{type(e).__name__}: {e}"}


def popcount64(x: np.ndarray) -> np.ndarray:
    """Number of 1-bits in each uint64 (= Hamming distance after XOR)."""
    if hasattr(np, "bitwise_count"):          # NumPy >= 2.0
        return np.bitwise_count(x)
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return lut[x.view(np.uint8)].reshape(-1, 8).sum(axis=1)


def find_duplicates(hashes_hex: list[str], max_dist: int) -> list[int]:
    """Return, for each position, -1 if kept or the index of the photo it duplicates.

    Photos must already be in priority order (best first): the first photo of each
    duplicate group is kept and the later ones point back to it.
    """
    h = np.array([int(x, 16) for x in hashes_hex], dtype=np.uint64)
    dup_of = np.full(len(h), -1)
    for i in range(len(h)):
        if dup_of[i] != -1:
            continue                                   # already removed as someone's duplicate
        # distance from i to every later photo
        dist = popcount64(h[i + 1:] ^ h[i])
        later = np.nonzero((dist <= max_dist) & (
            dup_of[i + 1:] == -1))[0] + i + 1
        dup_of[later] = i
    return dup_of.tolist()


def build_manifest(sources: list[dict], cfg: dict, workers: int = 16) -> pd.DataFrame:
    ing = cfg["ingest"]

    # 1. List candidate photos, with a stable ID like vehide_000123
    rows = []
    for src in sources:
        files = list_photos(Path(src["root"]), src.get("exclude", []))
        print(f"{src['prefix']:<8} {len(files):6d} photos  ({src['root']})")
        for i, p in enumerate(files):
            rows.append({"file_id": f"{src['prefix']}_{i:06d}", "source": src["prefix"],
                         "original_path": str(p), "ext": p.suffix.lower(),
                         "src_inpaint": bool(src.get("inpaint_source", False))})
    if not rows:
        raise SystemExit(
            "No photos found. Check the source paths in configs/config.yaml.")
    df = pd.DataFrame(rows)

    # 2. Inspect every photo in parallel (I/O-bound -> threads)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        info = list(tqdm(pool.map(inspect, map(Path, df["original_path"])),
                         total=len(df), desc="inspecting"))
    df = pd.concat([df, pd.DataFrame(info)], axis=1)
    df["min_side"] = df[["width", "height"]].min(axis=1)

    # 3. Corrupt and too-small photos
    df["drop_reason"] = ""
    df.loc[df["error"] != "", "drop_reason"] = "corrupt"
    df.loc[(df["drop_reason"] == "") & (df["min_side"] <
                                        ing["min_side"]), "drop_reason"] = "too_small"

    # 4. Near-duplicates among the remaining photos. Priority = largest first,
    #    so when two copies exist we keep the higher-resolution one.
    ok = df[df["drop_reason"] == ""].sort_values(
        ["min_side", "file_id"], ascending=[False, True])
    dup_of = find_duplicates(ok["phash"].tolist(), ing["dedup_hash_distance"])
    ids = ok["file_id"].tolist()
    for row_idx, d in zip(ok.index, dup_of):
        if d != -1:
            df.at[row_idx, "drop_reason"] = f"duplicate_of:{ids[d]}"

    # 5. Final flags
    df["keep"] = df["drop_reason"] == ""
    df["inpaint_source"] = df["keep"] & df["src_inpaint"] & (
        df["min_side"] >= ing["inpaint_min_side"])
    return df.drop(columns=["src_inpaint"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", default=[],
                        help="override config sources: prefix=folder (repeatable)")
    parser.add_argument("--out", default=None,
                        help="output CSV (default: config ingest.manifest_csv)")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    cfg = load_config()
    if args.source:
        sources = []
        for s in args.source:
            prefix, root = s.split("=", 1)
            sources.append({"prefix": prefix, "root": root, "exclude": [
                           "masks"], "inpaint_source": True})
    else:
        sources = cfg["ingest"]["real_sources"]

    df = build_manifest(sources, cfg, workers=args.workers)

    out = Path(args.out) if args.out else PROJECT_ROOT / \
        cfg["ingest"]["manifest_csv"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    reasons = df["drop_reason"].str.split(":").str[0].replace("", "kept")
    print(f"\nSaved {len(df)} rows to {out}\n")
    print("Outcome per source:")
    print(pd.crosstab(df["source"], reasons, margins=True), "\n")
    print(f"Kept photos:        {df['keep'].sum()}")
    print(f"Inpainting sources: {df['inpaint_source'].sum()}")


if __name__ == "__main__":
    main()
