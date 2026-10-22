"""Checks for src/clean_real.py. Run with:  pytest -q"""
import numpy as np
from PIL import Image

from src.clean_real import build_manifest, find_duplicates
from src.utils import load_config


def photo(path, seed, size=(640, 480), fmt="JPEG"):
    """A smooth random 'photo' (smooth so perceptual hashing behaves like on real images)."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (6, 8, 3), dtype=np.uint8)
    img = Image.fromarray(small).resize(size, Image.BICUBIC)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, fmt)
    return img


def test_find_duplicates_keeps_first_of_each_group():
    # hashes 0 and 1 differ by 1 bit; 2 is far away; 3 duplicates 2
    hashes = ["0000000000000000", "0000000000000001",
              "ffffffffffffffff", "fffffffffffffffe"]
    assert find_duplicates(hashes, max_dist=4) == [-1, 0, -1, 2]


def test_manifest_flags_each_problem(tmp_path):
    a, b = tmp_path / "vehide", tmp_path / "hitl"
    for i in range(5):
        photo(a / "image" / f"p{i}.jpg", seed=i, size=(1024, 768))
    img = photo(b / "img" / "big.png", seed=100, size=(700, 480), fmt="PNG")
    # near-duplicate, lower res
    img.resize((600, 411)).save(b / "img" / "big_copy_small.png")
    photo(b / "img" / "tiny.png", seed=200, size=(200, 150), fmt="PNG")
    (b / "img" / "broken.jpg").write_bytes(b"not really a jpeg")
    photo(b / "masks_human" / "m.png", seed=300,
          fmt="PNG")          # must be excluded

    cfg = load_config()
    sources = [{"prefix": "vehide", "root": str(a), "exclude": [], "inpaint_source": True},
               {"prefix": "hitl", "root": str(b), "exclude": ["masks"], "inpaint_source": False}]
    df = build_manifest(sources, cfg, workers=4).set_index("original_path")

    # masks never listed
    assert not any("masks" in p for p in df.index)
    assert df.loc[str(b / "img" / "broken.jpg"), "drop_reason"] == "corrupt"
    assert df.loc[str(b / "img" / "tiny.png"), "drop_reason"] == "too_small"
    # larger copy kept
    assert df.loc[str(b / "img" / "big.png"), "keep"]
    assert df.loc[str(b / "img" / "big_copy_small.png"),
                  "drop_reason"].startswith("duplicate_of:")
    assert df.loc[[str(a / "image" / f"p{i}.jpg")
                   for i in range(5)], "inpaint_source"].all()
    # hitl never an inpaint source
    assert not df.loc[str(b / "img" / "big.png"), "inpaint_source"]
