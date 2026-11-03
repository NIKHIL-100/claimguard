"""Step 2.4: make inpainted fakes by repainting part of a REAL close-up photo with AI.

    python -m src.inpaint --generator sd15_inpaint --device cuda:0
    python -m src.inpaint --generator sdxl_inpaint --device cuda:1 --limit 5     # quick test

This is the realistic insurance-fraud case: a genuine claim photo with damage added or
enlarged. For each image:
1. take a real VehiDE photo flagged as an inpainting source in data/real_manifest.csv
2. crop a square window, draw a random damage-shaped mask inside it (blob = dent or crack,
   thin streak = scratches), never touching the top band where the timestamp overlay sits
3. the inpainting model repaints the masked region with a damage prompt
4. paste ONLY the masked region (with a soft edge) back onto the ORIGINAL full photo, so
   every other pixel stays exactly as the camera captured it. A diffusion model re-encodes
   its whole input through its VAE, which would leave AI traces everywhere [DiffusionPrint];
   pasting back keeps the edit local, as a real fraudster's edit would be.
5. save the image, its mask (ground truth for localisation, Step 10) and a log row

Names link each fake to its source photo: vehide_000123__sd15_inpaint.png is an edit of
real photo vehide_000123, so build_metadata puts both in the same split (no leakage).
Each generator edits DIFFERENT source photos (disjoint slices of one seeded shuffle).
Same design as src/generate.py: reproducible per index, resumable, atomic writes, logged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps
from tqdm.auto import tqdm

from src.prompts import NEGATIVE
from src.utils import PROJECT_ROOT, load_config

# Damage to paint, matched to the mask shape.
BLOB_DAMAGES = ["a deep dent", "a large dent with creased metal", "a crack", "a crumpled dent with scratches",
                "chipped and scraped paint", "a dent with cracked paint"]
STREAK_DAMAGES = ["long deep scratches", "a long scrape through the paint", "several scratches",
                  "a deep key scratch"]

LOG_FIELDS = ["file", "generator", "index", "source_file_id", "source_path", "source_width", "source_height",
              "window_x", "window_y", "window", "model_res", "mask_shape", "mask_area_frac", "prompt",
              "negative_prompt", "image_seed", "steps", "guidance", "strength", "model_id", "seconds",
              "created_utc"]


def stable_seed(*parts) -> int:
    """A seed that depends only on the text of `parts` (same on every machine and run)."""
    return int.from_bytes(hashlib.sha256("-".join(map(str, parts)).encode()).digest()[:8], "little")


# ----------------------------------------------------------------------------------- sources
def assign_sources(manifest: pd.DataFrame, cfg: dict, generator: str) -> pd.DataFrame:
    """This generator's slice of the inpainting sources. Slices never overlap between generators."""
    src = manifest[manifest["inpaint_source"].astype(
        str).str.lower() == "true"]
    src = src.sort_values("file_id").reset_index(drop=True)
    order = np.random.default_rng(cfg["seed"]).permutation(len(src))
    src = src.iloc[order].reset_index(drop=True)

    start = 0
    # inpaint generators, in config order
    for name, g in cfg["generators"].items():
        if g["mode"] != "inpaint":
            continue
        if name == generator:
            return src.iloc[start:start + g["n_generate"]].reset_index(drop=True)
        start += g["n_generate"]
    raise ValueError(
        f"{generator} is not an inpainting generator in config.yaml")


def output_name(file_id: str, generator: str) -> str:
    return f"{file_id}__{generator}.png"


# ------------------------------------------------------------------------------------- masks
def _ellipse(h: int, w: int, cy: float, cx: float, ry: float, rx: float, angle: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    y, x = yy - cy, xx - cx
    c, s = np.cos(angle), np.sin(angle)
    return ((x * c + y * s) / rx) ** 2 + ((-x * s + y * c) / ry) ** 2 <= 1.0


def random_damage_mask(size: int, rng: np.random.Generator, area_range: tuple[float, float],
                       streak_share: float) -> tuple[np.ndarray, str]:
    """A damage-shaped mask (bool, size x size) inside the central part of the window.

    'blob'  : union of a few overlapping ellipses -> irregular dent / crack region
    'streak': a few long thin ellipses at similar angles -> scratches
    """
    shape = "streak" if rng.random() < streak_share else "blob"
    target = rng.uniform(*area_range) * size * size
    mask = np.zeros((size, size), bool)
    # keep the edit away from the edges
    cy, cx = rng.uniform(0.3, 0.7, size=2) * size
    base_angle = rng.uniform(0, np.pi)

    # add pieces until the area is reached
    for _ in range(40):
        if mask.sum() >= target:
            break
        if shape == "blob":
            r = np.sqrt(target / np.pi) * rng.uniform(0.35, 0.7)
            oy, ox = rng.normal(0, r * 0.6, size=2)
            mask |= _ellipse(size, size, cy + oy, cx + ox, r *
                             rng.uniform(0.6, 1.0), r, rng.uniform(0, np.pi))
        else:
            length = size * rng.uniform(0.2, 0.4)
            width = max(2.0, size * rng.uniform(0.008, 0.02))
            oy, ox = rng.normal(0, size * 0.04, size=2)
            mask |= _ellipse(size, size, cy + oy, cx + ox, width,
                             length, base_angle + rng.normal(0, 0.15))
    return mask, shape


# --------------------------------------------------------------------------------- one image
def inpaint_one(photo: Image.Image, gcfg: dict, icfg: dict, rng: np.random.Generator,
                call_model: Callable[[Image.Image, Image.Image, str], Image.Image]) -> tuple[Image.Image, Image.Image, dict]:
    """Edit one real photo. call_model(window_img, mask_img, prompt) -> repainted window (model_res square)."""
    photo = photo.convert("RGB")
    w, h = photo.size
    win = min(gcfg["window"], w, h)
    res = gcfg["model_res"]
    top = int(icfg["top_band_frac"] * h)

    # retry if the mask lands in the top band
    for _ in range(20):
        x0 = int(rng.integers(0, w - win + 1))
        y0 = int(rng.integers(0, h - win + 1))
        m, shape = random_damage_mask(win, rng, tuple(
            icfg["mask_area"]), icfg["streak_share"])
        full = np.zeros((h, w), bool)
        full[y0:y0 + win, x0:x0 + win] = m
        # never touch the timestamp band
        full[:top] = False
        m = full[y0:y0 + win, x0:x0 + win]
        if m.sum() > 0.5 * icfg["mask_area"][0] * win * win:
            break

    damage = rng.choice(STREAK_DAMAGES if shape == "streak" else BLOB_DAMAGES)
    prompt = f"close-up photo of {damage} on the car body, realistic, natural light"

    window = photo.crop((x0, y0, x0 + win, y0 + win))
    mask_img = Image.fromarray((m * 255).astype(np.uint8))
    if win != res:                                             # SDXL on smaller photos: scale up for the model
        model_in, model_mask = window.resize(
            (res, res), Image.LANCZOS), mask_img.resize((res, res), Image.NEAREST)
    else:
        model_in, model_mask = window, mask_img
    out = call_model(model_in, model_mask, prompt).convert("RGB")
    if out.size != (win, win):
        out = out.resize((win, win), Image.LANCZOS)

    # Soft-edged paste-back: alpha is 1 inside the mask, fades over feather_px, 0 elsewhere.
    alpha = np.asarray(mask_img.filter(
        ImageFilter.GaussianBlur(icfg["feather_px"])), np.float32) / 255.0
    # keep the band untouched after blurring too
    alpha[: max(0, top - y0)] = 0.0
    orig = np.asarray(window, np.float32)
    new = np.asarray(out, np.float32)
    blended = alpha[..., None] * new + (1 - alpha[..., None]) * orig
    blended = np.where(alpha[..., None] > 0, np.clip(
        np.rint(blended), 0, 255), orig).astype(np.uint8)

    result = photo.copy()
    result.paste(Image.fromarray(blended), (x0, y0))
    # ground truth: every pixel that changed
    gt = np.zeros((h, w), np.uint8)
    gt[y0:y0 + win, x0:x0 + win] = (alpha > 0) * 255
    info = {"window_x": x0, "window_y": y0, "window": win, "model_res": res, "mask_shape": shape,
            "mask_area_frac": round(float(m.mean()), 4), "prompt": prompt}
    return result, Image.fromarray(gt), info


# ---------------------------------------------------------------------------------- the loop
def finished_ids(generator: str, out_dir: Path) -> set[str]:
    """Source ids with both an image and a log row. Orphan images (crash before logging) are removed."""
    log_path = out_dir / "log.csv"
    logged = set(pd.read_csv(log_path)[
                 "source_file_id"]) if log_path.exists() else set()
    done = set()
    for png in out_dir.glob(f"*__{generator}.png"):
        sid = png.stem.split("__")[0]
        if sid in logged:
            done.add(sid)
        else:
            png.unlink()
    return done


def run_inpainting(generator: str, gcfg: dict, icfg: dict, seed: int, sources: pd.DataFrame,
                   out_dir: Path, mask_dir: Path, call_model) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.csv"
    done = finished_ids(generator, out_dir)
    todo = [(i, r) for i, r in sources.iterrows() if r["file_id"] not in done]
    print(f"{generator}: {len(done)} already done, {len(todo)} to generate -> {out_dir}")

    new_log = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=LOG_FIELDS, extrasaction="ignore")
        if new_log:
            writer.writeheader()
        for i, r in tqdm(todo, desc=generator):
            rng = np.random.default_rng(
                stable_seed(seed, generator, r["file_id"]))
            image_seed = int(rng.integers(2**32))
            t0 = time.time()
            with Image.open(r["original_path"]) as im:
                # same orientation a viewer would show
                photo = ImageOps.exif_transpose(im)
                result, gt, info = inpaint_one(photo, gcfg, icfg, rng,
                                               lambda a, b, p: call_model(a, b, p, image_seed))
            seconds = time.time() - t0

            name = output_name(r["file_id"], generator)
            # mask first, image last
            for img, folder in ((gt, mask_dir), (result, out_dir)):
                tmp = folder / (name + ".tmp")
                img.save(tmp, format="PNG")
                os.replace(tmp, folder / name)
            writer.writerow({**info, "file": name, "generator": generator, "index": i,
                             "source_file_id": r["file_id"], "source_path": r["original_path"],
                             "source_width": result.width, "source_height": result.height,
                             "negative_prompt": NEGATIVE, "image_seed": image_seed,
                             "steps": gcfg["steps"], "guidance": gcfg["guidance"], "strength": gcfg["strength"],
                             "model_id": gcfg["model_id"], "seconds": round(seconds, 2),
                             "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            f.flush()
    return {"done_before": len(done), "generated": len(todo), "total": len(done) + len(todo)}


# --------------------------------------------------------------------- real models (GPU only)
def build_call_model(generator: str, gcfg: dict, device: str):
    import torch
    from diffusers.utils import logging as dlogging

    dlogging.set_verbosity_error()
    torch.cuda.set_device(int(device.split(":")[1]) if ":" in device else 0)
    fp16 = torch.float16
    if generator == "sd15_inpaint":
        from diffusers import StableDiffusionInpaintPipeline
        pipe = StableDiffusionInpaintPipeline.from_pretrained(gcfg["model_id"], dtype=fp16,
                                                              safety_checker=None).to(device)
    elif generator == "sdxl_inpaint":
        from diffusers import StableDiffusionXLInpaintPipeline
        try:                                   # half-precision files, if the repo provides them
            pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                gcfg["model_id"], dtype=fp16, variant="fp16")
        except (OSError, ValueError):
            pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                gcfg["model_id"], dtype=fp16)
        pipe = pipe.to(device)
    else:
        raise ValueError(f"No inpainting loader for '{generator}'")
    pipe.set_progress_bar_config(disable=True)
    res = gcfg["model_res"]

    def call_model(window: Image.Image, mask: Image.Image, prompt: str, image_seed: int) -> Image.Image:
        g = torch.Generator(device).manual_seed(image_seed)
        return pipe(prompt=prompt, negative_prompt=NEGATIVE, image=window, mask_image=mask,
                    height=res, width=res, strength=gcfg["strength"], num_inference_steps=gcfg["steps"],
                    guidance_scale=gcfg["guidance"], generator=g).images[0]
    return call_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", required=True,
                        help="an inpaint key from config.yaml generators")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N source photos (testing)")
    args = parser.parse_args()

    cfg = load_config()
    gcfg = cfg["generators"][args.generator]
    if gcfg["mode"] != "inpaint":
        raise SystemExit(
            f"{args.generator} is a text-to-image generator: use src.generate.")
    manifest = pd.read_csv(PROJECT_ROOT / cfg["ingest"]["manifest_csv"])
    sources = assign_sources(manifest, cfg, args.generator)
    if args.limit:
        sources = sources.head(args.limit)

    out_dir = PROJECT_ROOT / cfg["paths"]["fake_dir"] / args.generator
    mask_dir = PROJECT_ROOT / cfg["paths"]["mask_dir"] / args.generator
    if set(sources["file_id"]) <= finished_ids(args.generator, out_dir):
        print(
            f"{args.generator}: all {len(sources)} images already exist. Nothing to do.")
        return
    call_model = build_call_model(args.generator, gcfg, args.device)
    stats = run_inpainting(args.generator, gcfg, cfg["inpainting"], cfg["seed"], sources,
                           out_dir, mask_dir, call_model)
    log = pd.read_csv(out_dir / "log.csv")
    print(
        f"\n{args.generator}: {stats}\nmean seconds/image: {log['seconds'].mean():.1f}")


if __name__ == "__main__":
    main()
