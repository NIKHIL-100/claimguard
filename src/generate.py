"""Step 2.3: generate fully AI-made fake photos with one text-to-image generator.

    python -m src.generate --generator sd15 --device cuda:0
    python -m src.generate --generator flux --device cuda:1 --limit 5     # quick test

Design (interview points):
1. REPRODUCIBLE: image i always uses prompt i and seed i from src/prompts.py, and every
   setting comes from configs/config.yaml. Any image can be regenerated exactly.
2. RESUMABLE: a Kaggle session can stop at any time. Images that already exist (and are
   logged) are skipped, so re-running carries on where it stopped.
3. CRASH-SAFE FILES: each image is written to a temporary file and then renamed. A rename
   is atomic, so a half-written PNG can never be mistaken for a finished one.
4. LOGGED: data/fake/<generator>/log.csv records prompt, seed, settings and seconds for
   every image, so the dataset is documented and auditable.
5. MEMORY: FLUX runs with 4-bit weights and tiled VAE decoding (peak 15.4 GB of a 15.8 GB
   T4 before tiling). PixArt-Sigma fits fully on the GPU (peak 14.9 GB); CPU offloading
   was measured and rejected because it doubled the time per image.

Images are saved as lossless PNG. Compression is applied identically to real and fake
images later (Step 4), so the file format cannot become a shortcut.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from src.prompts import make_prompts
from src.utils import PROJECT_ROOT, load_config

LOG_FIELDS = ["file", "generator", "index", "model_id", "prompt", "negative_prompt", "image_seed",
              "width", "height", "steps", "guidance", "seconds", "created_utc",
              "colour", "make", "car_type", "damage", "part", "framing", "scene", "lighting"]


def image_name(generator: str, index: int) -> str:
    return f"{generator}_{index:05d}.png"


def read_log(log_path: Path) -> pd.DataFrame:
    if not log_path.exists():
        return pd.DataFrame(columns=LOG_FIELDS)
    log = pd.read_csv(log_path)
    # a re-generated index keeps its newest row
    return log.drop_duplicates("index", keep="last")


def finished_indices(generator: str, out_dir: Path) -> set[int]:
    """Indices with BOTH a PNG and a log row. Orphan PNGs (crash between save and log) are removed."""
    logged = set(read_log(out_dir / "log.csv")["index"].astype(int))
    done = set()
    for png in out_dir.glob(f"{generator}_*.png"):
        idx = int(png.stem.rsplit("_", 1)[1])
        if idx in logged:
            done.add(idx)
        else:
            png.unlink()          # no record of how it was made -> regenerate it
    return done


def run_generation(generator: str, gcfg: dict, prompts: list[dict], out_dir: Path,
                   generate_fn: Callable[[dict], Image.Image]) -> dict:
    """Generate every prompt that is not finished yet. generate_fn(prompt_row) -> PIL image."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.csv"
    done = finished_indices(generator, out_dir)
    todo = [p for p in prompts if p["index"] not in done]
    print(f"{generator}: {len(done)} already done, {len(todo)} to generate -> {out_dir}")

    new_log = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=LOG_FIELDS, extrasaction="ignore")
        if new_log:
            writer.writeheader()
        for p in tqdm(todo, desc=generator):
            t0 = time.time()
            img = generate_fn(p)
            seconds = time.time() - t0

            name = image_name(generator, p["index"])
            tmp = out_dir / (name + ".tmp")
            img.save(tmp, format="PNG")
            # atomic: file is complete or absent
            os.replace(tmp, out_dir / name)

            writer.writerow({**p, "file": name, "generator": generator, "model_id": gcfg["model_id"],
                             "negative_prompt": p["negative_prompt"] if generator != "flux" else "",
                             "width": gcfg["width"], "height": gcfg["height"], "steps": gcfg["steps"],
                             "guidance": gcfg["guidance"], "seconds": round(seconds, 2),
                             "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            # keep the log current if the session dies
            f.flush()
    return {"done_before": len(done), "generated": len(todo), "total": len(done) + len(todo)}


# ---------------------------------------------------------------------------------------
# Real generators (need a GPU; imported lazily so the tests above run on any laptop)
# ---------------------------------------------------------------------------------------
def build_generate_fn(generator: str, gcfg: dict, device: str) -> Callable[[dict], Image.Image]:
    import torch
    from diffusers.utils import logging as dlogging

    # hide harmless deprecation / upcast warnings
    dlogging.set_verbosity_error()
    gpu_id = int(device.split(":")[1]) if ":" in device else 0
    torch.cuda.set_device(gpu_id)
    fp16 = torch.float16
    common = dict(width=gcfg["width"], height=gcfg["height"],
                  num_inference_steps=gcfg["steps"], guidance_scale=gcfg["guidance"])

    if generator == "sd15":
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(gcfg["model_id"], dtype=fp16,
                                                       safety_checker=None).to(device)
        extra = {}
    elif generator == "sdxl":
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(gcfg["model_id"], dtype=fp16, variant="fp16",
                                                         use_safetensors=True).to(device)
        extra = {}
    elif generator == "pixart_sigma":
        from diffusers import PixArtSigmaPipeline
        # Fully on the GPU (peak 14.9 GB of 15.8 GB in the smoke test, constant per image).
        # CPU offloading was tried in the quick check: it moved the 9.5 GB T5 encoder to the
        # GPU and back for every image and doubled the time (41 s vs 21 s per image).
        pipe = PixArtSigmaPipeline.from_pretrained(
            gcfg["model_id"], dtype=fp16).to(device)
        extra = {"clean_caption": False}
    elif generator == "flux":
        from diffusers import FluxPipeline
        from diffusers.quantizers import PipelineQuantizationConfig
        quant = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                          "bnb_4bit_compute_dtype": fp16},
            components_to_quantize=["transformer", "text_encoder_2"])
        pipe = FluxPipeline.from_pretrained(
            gcfg["model_id"], quantization_config=quant, dtype=fp16).to(device)
        pipe.vae.enable_tiling()          # decode the image in tiles: lower peak memory
        extra = {"max_sequence_length": 256}
    else:
        raise ValueError(f"No text-to-image loader for '{generator}'")

    pipe.set_progress_bar_config(disable=True)

    def generate_fn(p: dict) -> Image.Image:
        g = torch.Generator(device).manual_seed(p["image_seed"])
        kwargs = {**common, **extra}
        if generator != "flux":           # FLUX-schnell does not use a negative prompt
            kwargs["negative_prompt"] = p["negative_prompt"]
        return pipe(p["prompt"], generator=g, **kwargs).images[0]

    return generate_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", required=True,
                        help="a text2img key from config.yaml generators")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N prompts (for testing)")
    parser.add_argument("--out-root", default=None,
                        help="default: config paths.fake_dir")
    args = parser.parse_args()

    cfg = load_config()
    gcfg = cfg["generators"][args.generator]
    if gcfg["mode"] != "text2img":
        raise SystemExit(
            f"{args.generator} is an inpainting generator (Step 2.4), not text-to-image.")

    n = gcfg["n_generate"] if args.limit is None else min(
        args.limit, gcfg["n_generate"])
    prompts = make_prompts(n, seed=cfg["seed"])
    out_root = Path(
        args.out_root) if args.out_root else PROJECT_ROOT / cfg["paths"]["fake_dir"]
    out_dir = out_root / args.generator

    if len(finished_indices(args.generator, out_dir) & {p["index"] for p in prompts}) == n:
        print(f"{args.generator}: all {n} images already exist. Nothing to do.")
        return
    generate_fn = build_generate_fn(args.generator, gcfg, args.device)
    stats = run_generation(args.generator, gcfg, prompts, out_dir, generate_fn)

    log = read_log(out_dir / "log.csv")
    print(f"\n{args.generator}: {stats}")
    if len(log):
        print(f"mean seconds/image: {log['seconds'].mean():.1f}")


if __name__ == "__main__":
    main()
