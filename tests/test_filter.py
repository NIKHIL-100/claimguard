"""Checks for the Step 2.5 relevance filter (no GPU, no CLIP download needed). Run with:  pytest -q"""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.clip_scores import PROMPTS, group_probs, list_all_images
from src.select_fakes import select_fakes
from src.utils import load_config

CFG = load_config()


def test_group_probs_is_a_probability_and_follows_similarity():
    sims = np.array([[0.30, 0.20, 0.20],     # closest to the positive prompt
                     [0.20, 0.30, 0.20],     # closest to a negative prompt
                     [0.25, 0.25, 0.25]])    # no preference
    p = group_probs(sims, n_pos=1)
    assert ((p >= 0) & (p <= 1)).all()
    assert p[0] > 0.99 and p[1] < 0.01 and p[2] == pytest.approx(1 / 3)


def test_config_has_every_filter_setting():
    assert set(CFG["filter"]) >= {
        "clip_model", "clip_pretrained", "real_quantile", "scores_csv", "selection_csv"}


def test_every_question_has_positive_and_negative_prompts():
    for q, (pos, neg) in PROMPTS.items():
        assert pos and neg, q


def write_fakes(root, gen, n, missing=()):
    # like a Kaggle notebook output folder
    d = root / gen / "data" / "fake" / gen
    d.mkdir(parents=True)
    names = [f"{gen}_{i:05d}.png" for i in range(n)]
    for name in names:
        if name not in missing:
            Image.new("RGB", (8, 8)).save(d / name)
    pd.DataFrame({"file": names}).to_csv(d / "log.csv", index=False)


def tiny_cfg():
    return {**CFG, "generators": {"sd15": CFG["generators"]["sd15"], "sd15_inpaint": CFG["generators"]["sd15_inpaint"]}}


def test_lists_kept_reals_and_existing_fakes(tmp_path):
    write_fakes(tmp_path, "sd15", 3, missing={"sd15_00001.png"})
    write_fakes(tmp_path, "sd15_inpaint", 2)
    manifest = pd.DataFrame({"file_id": ["vehide_000001", "vehide_000002"], "original_path": ["a.jpg", "b.jpg"],
                             "keep": [True, False]})
    df = list_all_images(tiny_cfg(), tmp_path, manifest)

    assert list(df.loc[df["label"] == 0, "image_id"]) == [
        "vehide_000001"]       # dropped real excluded
    assert sorted(df.loc[df["generator"] == "sd15", "image_id"]) == [
        "sd15_00000", "sd15_00002"]
    assert (df.loc[df["generator"] == "sd15_inpaint", "label"] == 1).all()


def test_generator_attached_twice_is_an_error(tmp_path):
    write_fakes(tmp_path / "run_a", "sd15", 1)
    write_fakes(tmp_path / "run_b", "sd15", 1)
    write_fakes(tmp_path, "sd15_inpaint", 1)
    manifest = pd.DataFrame({"file_id": [], "original_path": [], "keep": []})
    with pytest.raises(SystemExit, match="several inputs"):
        list_all_images(tiny_cfg(), tmp_path, manifest)


def synthetic_scores(n_real=1000, n_t2i=200, n_inp=50, seed=0):
    rng = np.random.default_rng(seed)
    real = pd.DataFrame({"image_id": [f"r{i}" for i in range(n_real)], "label": 0, "generator": "none",
                         "p_car": rng.uniform(0.5, 1, n_real), "p_damage": rng.uniform(0.5, 1, n_real)})
    t2i = pd.DataFrame({"image_id": [f"sd15_{i:05d}" for i in range(n_t2i)], "label": 1, "generator": "sd15",
                        "p_car": rng.uniform(0, 1, n_t2i), "p_damage": rng.uniform(0, 1, n_t2i)})
    inp = pd.DataFrame({"image_id": [f"x{i}__sd15_inpaint" for i in range(n_inp)], "label": 1,
                        "generator": "sd15_inpaint", "p_car": 0.01, "p_damage": 0.01})   # low on purpose
    df = pd.concat([real, t2i, inp], ignore_index=True)
    df["p_closeup"] = 0.5
    return df


def cfg_with_keep(n_keep):
    cfg = tiny_cfg()
    cfg["generators"] = {"sd15": {**CFG["generators"]["sd15"], "n_keep": n_keep},
                         "sd15_inpaint": CFG["generators"]["sd15_inpaint"]}
    return cfg


def test_thresholds_come_from_reals_and_fakes_below_are_rejected():
    scores = synthetic_scores()
    sel, t = select_fakes(scores, cfg_with_keep(20))
    reals = scores[scores["label"] == 0]
    assert t["car"] == pytest.approx(reals["p_car"].quantile(0.05))

    sd15 = sel[sel["generator"] == "sd15"]
    kept = sd15[sd15["keep"]]
    assert len(kept) == 20
    assert (kept["p_car"] >= t["car"]).all() and (
        kept["p_damage"] >= t["damage"]).all()
    assert set(sd15.loc[sd15["p_car"] < t["car"], "reason"]) == {"no_car"}
    assert set(sd15["reason"]) <= {
        "no_car", "no_damage", "pass", "pass_not_sampled"}


def test_inpainted_fakes_are_all_kept_whatever_their_scores():
    sel, _ = select_fakes(synthetic_scores(), cfg_with_keep(20))
    inp = sel[sel["generator"] == "sd15_inpaint"]
    assert len(inp) == 50 and inp["keep"].all() and (
        inp["reason"] == "inpaint_kept").all()


def test_selection_is_reproducible_and_keeps_all_when_too_few_pass():
    a, _ = select_fakes(synthetic_scores(), cfg_with_keep(20))
    b, _ = select_fakes(synthetic_scores(), cfg_with_keep(20))
    # same seed -> same images
    assert a.equals(b)
    c, _ = select_fakes(synthetic_scores(), cfg_with_keep(10_000))
    sd15 = c[c["generator"] == "sd15"]
    assert sd15["keep"].sum() == (
        sd15["reason"] == "pass").sum()  # every passing image kept
