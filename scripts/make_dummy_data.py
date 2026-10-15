"""Create tiny random images in the right folders, to test the pipeline before real data exists.

Usage:  python scripts/make_dummy_data.py [target_dir]
Default target is the project root. Delete data/real/*, data/fake/* and data/metadata.csv afterwards.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    __file__).resolve().parents[1]
rng = np.random.default_rng(0)


def make(rel_path: str, fmt: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(0, 255, (64, 80, 3),
                    dtype=np.uint8)).save(path, fmt)


for i in range(20):
    make(f"data/real/car_{i}.jpg", "JPEG")                     # real photos
for i in range(8):
    # fully generated
    make(f"data/fake/sdxl/s_{i}.png", "PNG")
for i in range(5):
    # hold-out generator
    make(f"data/fake/flux/f_{i}.png", "PNG")
for i in range(6):
    # edits of car_0 .. car_5
    make(f"data/fake/sd15_inpaint/car_{i}__dent.png", "PNG")
make("data/fake/flux_inpaint/car_10__scratch.png",
     "PNG")      # hold-out edit of car_10

print(f"Dummy data created under {root / 'data'}")
