"""Checks for src/inpaint.py with a fake inpainting model (no GPU needed). Run with:  pytest -q"""
import numpy as np
import pandas as pd
from PIL import Image

from src.inpaint import assign_sources, inpaint_one, output_name, random_damage_mask, run_inpainting
from src.utils import load_config

CFG = load_config()
ICFG = CFG["inpainting"]


def fake_model(window, mask, prompt, seed=0):
    """Stands in for a diffusion model: paints the whole window solid red."""
    return Image.new("RGB", window.size, (255, 0, 0))


def smooth_photo(w, h, seed):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (6, 8, 3), dtype=np.uint8)
    return Image.fromarray(small).resize((w, h), Image.BICUBIC)


def test_masks_have_damage_like_size_and_shape():
    rng = np.random.default_rng(0)
    shapes = set()
    for _ in range(50):
        m, shape = random_damage_mask(512, rng, (0.03, 0.15), 0.4)
        shapes.add(shape)
        assert m.dtype == bool and m.shape == (512, 512)
        # a few % of the window, like real damage
        assert 0.01 < m.mean() < 0.25
    assert shapes == {"blob", "streak"}


def test_only_masked_pixels_change_and_top_band_is_untouched():
    photo = smooth_photo(1632, 1224, seed=1)
    gcfg = CFG["generators"]["sd15_inpaint"]
    result, gt, info = inpaint_one(
        photo, gcfg, ICFG, np.random.default_rng(3), fake_model)

    before, after, changed = np.asarray(
        photo), np.asarray(result), np.asarray(gt) > 0
    # full-resolution output
    assert result.size == photo.size
    assert changed.any()
    # everything else is the real photo
    assert (before[~changed] == after[~changed]).all()
    top = int(ICFG["top_band_frac"] * photo.height)
    # timestamp band never edited
    assert not changed[:top].any()


def test_small_photo_with_sdxl_window_is_scaled_for_the_model():
    # shorter than SDXL's 1024 window
    photo = smooth_photo(800, 600, seed=2)
    seen = []

    def model(window, mask, prompt, seed=0):
        seen.append(window.size)
        return fake_model(window, mask, prompt)
    result, gt, info = inpaint_one(
        photo, CFG["generators"]["sdxl_inpaint"], ICFG, np.random.default_rng(0), model)
    assert info["window"] == 600 and seen == [
        (1024, 1024)] and result.size == (800, 600)


def test_generators_get_disjoint_source_photos():
    n = 3000
    manifest = pd.DataFrame({"file_id": [f"vehide_{i:06d}" for i in range(n)],
                             "inpaint_source": [i % 10 != 0 for i in range(n)]})   # 10% not eligible
    a = assign_sources(manifest, CFG, "sd15_inpaint")
    b = assign_sources(manifest, CFG, "sdxl_inpaint")
    assert len(a) == CFG["generators"]["sd15_inpaint"]["n_generate"]
    assert len(b) == CFG["generators"]["sdxl_inpaint"]["n_generate"]
    # never the same photo twice
    assert not set(a["file_id"]) & set(b["file_id"])
    assert a["file_id"].isin(
        manifest.loc[manifest["inpaint_source"], "file_id"]).all()


def test_run_names_logs_and_resumes(tmp_path):
    rows = []
    for i in range(4):
        p = tmp_path / "src" / f"photo{i}.jpg"
        p.parent.mkdir(exist_ok=True)
        smooth_photo(800, 600, seed=i).save(p)
        rows.append({"file_id": f"vehide_{i:06d}", "original_path": str(p)})
    sources = pd.DataFrame(rows)
    out, masks = tmp_path / "fake", tmp_path / "masks"
    gcfg = CFG["generators"]["sd15_inpaint"]

    run_inpainting("sd15_inpaint", gcfg, ICFG, 42,
                   sources.head(2), out, masks, fake_model)
    stats = run_inpainting("sd15_inpaint", gcfg, ICFG,
                           42, sources, out, masks, fake_model)

    assert stats == {"done_before": 2, "generated": 2, "total": 4}
    names = sorted(p.name for p in out.glob("*.png"))
    assert names == [output_name(
        f"vehide_{i:06d}", "sd15_inpaint") for i in range(4)]
    assert sorted(p.name for p in masks.glob("*.png")
                  ) == names       # one mask per image
    # build_metadata links an edit to its source via the part before "__"
    assert [n.split("__")[0] for n in names] == list(sources["file_id"])
    assert len(pd.read_csv(out / "log.csv")) == 4
