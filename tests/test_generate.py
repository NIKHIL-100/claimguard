"""Checks for src/generate.py using a fake generator (no GPU needed). Run with:  pytest -q"""
import pandas as pd
from PIL import Image

from src.generate import image_name, run_generation
from src.prompts import make_prompts
from src.utils import load_config

GCFG = load_config()["generators"]["sd15"]


def fake_generator(calls):
    """Stands in for a diffusion model: returns a small solid image and records each call."""
    def generate_fn(p):
        calls.append(p["index"])
        return Image.new("RGB", (64, 48), (p["index"] % 255, 0, 0))
    return generate_fn


def test_generates_every_prompt_and_logs_it(tmp_path):
    calls, prompts = [], make_prompts(5)
    stats = run_generation("sd15", GCFG, prompts,
                           tmp_path, fake_generator(calls))

    assert stats == {"done_before": 0, "generated": 5, "total": 5}
    assert sorted(p.name for p in tmp_path.glob("*.png")
                  ) == [image_name("sd15", i) for i in range(5)]
    log = pd.read_csv(tmp_path / "log.csv")
    assert list(log["index"]) == [0, 1, 2, 3, 4]
    assert list(log["prompt"]) == [p["prompt"]
                                   for p in prompts]          # prompt recorded
    assert list(log["image_seed"]) == [p["image_seed"]
                                       for p in prompts]  # seed recorded
    # no half-written files
    assert not list(tmp_path.glob("*.tmp"))


def test_resume_skips_finished_images(tmp_path):
    run_generation("sd15", GCFG, make_prompts(3), tmp_path, fake_generator([]))
    calls = []
    stats = run_generation("sd15", GCFG, make_prompts(6),
                           tmp_path, fake_generator(calls))
    # only the new ones
    assert calls == [3, 4, 5]
    assert stats == {"done_before": 3, "generated": 3, "total": 6}
    assert len(pd.read_csv(tmp_path / "log.csv")
               ) == 6         # one header, six rows


def test_orphan_png_without_log_row_is_regenerated(tmp_path):
    run_generation("sd15", GCFG, make_prompts(2), tmp_path, fake_generator([]))
    # Simulate a crash after saving image 2 but before logging it.
    Image.new("RGB", (8, 8)).save(tmp_path / image_name("sd15", 2))
    calls = []
    run_generation("sd15", GCFG, make_prompts(
        3), tmp_path, fake_generator(calls))
    assert calls == [2]
    with Image.open(tmp_path / image_name("sd15", 2)) as img:
        # the orphan was replaced
        assert img.size == (64, 48)
