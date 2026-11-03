"""Step 2.5a: score every image with a small CLIP model: does it show a car, is it damaged,
is it a close-up?

    python -m src.clip_scores --input-root /kaggle/input --device cuda:0

Why (interview points):
1. Text-to-image models sometimes ignore the prompt: an abstract texture, a road with no
   car, a shiny undamaged car. A real claim photo is never like that. If such fakes stay
   in the dataset, the detector can learn "no damage = fake", a CONTENT shortcut that
   has nothing to do with AI traces. src/select_fakes.py removes them using these scores.
2. ZERO-SHOT: CLIP compares an image with text descriptions, so no labelled training data
   is needed. For each question we write positive prompts ("a photo of a car") and
   negative prompts ("an abstract texture"), take a softmax over all of them and add up
   the positive probabilities: p_car, p_damage, p_closeup, each between 0 and 1.
3. Real photos are scored too. The thresholds come from the reals ("at least as car-like
   as 95% of real claim photos"), so we never pick an arbitrary cut-off by hand.
4. The filter uses ViT-B/32, a small CLIP that is NOT one of the detectors benchmarked in
   Step 5. Filtering with the same model we later evaluate would bias that benchmark.
5. p_closeup is kept for EDA (Step 3): it checks that real and fake framing match.

Output: data/clip_scores.csv  (image_id, path, label, generator, p_car, p_damage, p_closeup)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from src.utils import PROJECT_ROOT, load_config, set_seed

# question -> (positive prompts, negative prompts). Each question gets its own softmax.
PROMPTS: dict[str, tuple[list[str], list[str]]] = {
    "car": (
        ["a photo of a car", "a close-up photo of part of a car",
            "a photo of a damaged car body panel"],
        # "3d render" is left out on purpose: realism is the detector's job, not the filter's.
        ["an abstract texture", "a photo of a wall or floor", "an illustration or drawing",
         "a photo of a street with no cars"],
    ),
    "damage": (
        ["a photo of a damaged car with dents or scratches", "a close-up of a dent or scratch on a car",
         "a car damaged in an accident"],
        ["a photo of a clean undamaged car",
            "a close-up of clean, shiny, undamaged car paint"],
    ),
    "closeup": (
        ["a close-up photo of a small part of a car"],
        ["a photo of a whole car"],
    ),
}
# CLIP's learned temperature (exp(4.6052) = 100 for OpenAI weights)
LOGIT_SCALE = 100.0


def group_probs(sims: np.ndarray, n_pos: int, scale: float = LOGIT_SCALE) -> np.ndarray:
    """sims: (n_images, n_prompts) cosine similarities; the first n_pos prompts are positive.
    Returns P(positive) per image = sum of softmax probabilities of the positive prompts."""
    logits = scale * sims
    # numerically stable softmax
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e[:, :n_pos].sum(axis=1) / e.sum(axis=1)


# ----------------------------------------------------------------------------- image list
def find_fake_logs(input_root: Path, generators: list[str]) -> dict[str, Path]:
    """Locate data/fake/<generator>/log.csv anywhere under input_root (Kaggle mounts each
    notebook output in its own folder). Fails if a generator is missing or found twice."""
    found: dict[str, list[Path]] = {g: [] for g in generators}
    for log in sorted(input_root.glob("**/fake/*/log.csv")):
        gen = log.parent.name
        if gen in found:
            found[gen].append(log)
    missing = [g for g, logs in found.items() if not logs]
    twice = {g: [str(p) for p in logs]
             for g, logs in found.items() if len(logs) > 1}
    if missing:
        raise SystemExit(
            f"No log.csv found for {missing}. Attach the notebook outputs that made them.")
    if twice:
        raise SystemExit(
            f"Generators found in several inputs (detach the extra ones): {twice}")
    return {g: logs[0] for g, logs in found.items()}


def list_all_images(cfg: dict, input_root: Path, manifest: pd.DataFrame | None = None) -> pd.DataFrame:
    """Kept real photos (from the manifest) + every logged fake that exists on disk."""
    if manifest is None:
        manifest = pd.read_csv(PROJECT_ROOT / cfg["ingest"]["manifest_csv"])
    reals = manifest[manifest["keep"].astype(bool)]
    rows = [{"image_id": r.file_id, "path": r.original_path, "label": 0, "generator": "none"}
            for r in reals.itertuples()]

    logs = find_fake_logs(input_root, list(cfg["generators"]))
    for gen, log_path in logs.items():
        log = pd.read_csv(log_path).drop_duplicates("file", keep="last")
        n_missing = 0
        for name in sorted(log["file"]):
            p = log_path.parent / name
            if not p.exists():
                n_missing += 1
                continue
            rows.append({"image_id": Path(name).stem,
                        "path": p.as_posix(), "label": 1, "generator": gen})
        if n_missing:
            print(
                f"warning: {gen}: {n_missing} logged files are missing on disk and were skipped")

    df = pd.DataFrame(rows)
    assert df["image_id"].is_unique, "image_id must be unique"
    return df


# ----------------------------------------------------------------------------- scoring (GPU)
def load_image(path: str) -> Image.Image:
    with Image.open(path) as img:
        # JPEG: decode at reduced size, much faster
        img.draft("RGB", (448, 448))
        # same orientation a person would see
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")


def score_images(df: pd.DataFrame, cfg: dict, device: str, batch_size: int = 128,
                 workers: int = 4) -> pd.DataFrame:
    import open_clip
    import torch
    from torch.utils.data import DataLoader, Dataset
    from tqdm.auto import tqdm

    fcfg = cfg["filter"]
    model, _, preprocess = open_clip.create_model_and_transforms(fcfg["clip_model"],
                                                                 pretrained=fcfg["clip_pretrained"])
    tokenizer = open_clip.get_tokenizer(fcfg["clip_model"])
    model = model.to(device).eval()

    class Images(Dataset):
        def __len__(self):
            return len(df)

        def __getitem__(self, i):
            return preprocess(load_image(df["path"].iat[i]))

    # One text embedding per prompt, computed once.
    order = [(q, pos + neg, len(pos)) for q, (pos, neg) in PROMPTS.items()]
    all_prompts = [t for _, texts, _ in order for t in texts]
    with torch.no_grad():
        text = model.encode_text(tokenizer(all_prompts).to(device)).float()
        text = text / text.norm(dim=-1, keepdim=True)

    feats = []
    loader = DataLoader(Images(), batch_size=batch_size, num_workers=workers)
    with torch.no_grad():
        for x in tqdm(loader, desc="CLIP"):
            # fp16 only inside the image encoder (speed); everything after it in float32
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                f = model.encode_image(x.to(device))
            feats.append(f.float().cpu().numpy())
    feats = np.concatenate(feats).astype(np.float64)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    # cosine similarities, float64
    sims = feats @ text.cpu().numpy().astype(np.float64).T

    out = df.copy()
    start = 0
    for q, texts, n_pos in order:
        out[f"p_{q}"] = group_probs(
            sims[:, start:start + len(texts)], n_pos).round(4)
        start += len(texts)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="/kaggle/input",
                        help="folder holding the generation outputs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None,
                        help="random sample of N images (quick test)")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    df = list_all_images(cfg, Path(args.input_root))
    print("Images to score:\n", df.groupby(
        "generator").size().to_string(), "\n")
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=cfg["seed"])

    scores = score_images(df, cfg, args.device)
    out = PROJECT_ROOT / cfg["filter"]["scores_csv"]
    scores.to_csv(out, index=False)
    print(f"\nSaved {len(scores)} rows to {out.relative_to(PROJECT_ROOT)}")
    print("Median scores per generator:")
    print(scores.groupby("generator")[
          ["p_car", "p_damage", "p_closeup"]].median().round(3))


if __name__ == "__main__":
    main()
