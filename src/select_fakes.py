"""Step 2.5b: choose which text-to-image fakes enter the dataset, using data/clip_scores.csv.

    python -m src.select_fakes

Rules (interview points):
1. DAMAGE THRESHOLD FROM THE REALS: t_damage = the 5% quantile of p_damage over real claim
   photos, so a fake must look at least as damaged as 95% of genuine photos.
   CAR THRESHOLD FIXED at 0.5 ("car" more likely than every non-car prompt together). The
   real-quantile rule gave 0.985 because CLIP is saturated on real photos (all ~1.0), and
   the Step 2.5 grids showed it rejecting ordinary whole cars.
2. RANDOM, NOT BEST: among the fakes that pass, we draw n_keep at random (seeded). Taking
   the highest-scoring ones would keep only the most "CLIP-typical" images and make the
   fakes less varied than the reals, a new bias.
3. INPAINTED FAKES ARE ALL KEPT: they are real damage photos with a small edited region,
   so they are on-topic by construction; their scores are kept for EDA only.
4. Every rejected image keeps its reason (no_car / no_damage), so the filter is auditable.

Output: data/fake_selection.csv  (one row per fake, with keep = True/False and a reason)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.inpaint import stable_seed
from src.utils import PROJECT_ROOT, load_config


def thresholds(scores: pd.DataFrame, fcfg: dict) -> dict[str, float]:
    reals = scores[scores["label"] == 0]
    if reals.empty:
        raise ValueError(
            "No real photos in the scores: the damage threshold comes from them.")
    return {"car": float(fcfg["car_threshold"]),
            "damage": float(reals["p_damage"].quantile(fcfg["real_quantile"]))}


def select_fakes(scores: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict[str, float]]:
    t = thresholds(scores, cfg["filter"])
    fakes = scores[scores["label"] == 1].copy()

    fakes["reason"] = np.select(
        [fakes["p_car"] < t["car"], fakes["p_damage"] < t["damage"]],
        ["no_car", "no_damage"], default="pass")
    fakes["keep"] = False

    for gen, gcfg in cfg["generators"].items():
        rows = fakes["generator"] == gen
        if gcfg["mode"] == "inpaint":
            fakes.loc[rows, "reason"] = "inpaint_kept"
            fakes.loc[rows, "keep"] = True
            continue
        passed = sorted(
            fakes.loc[rows & (fakes["reason"] == "pass"), "image_id"])
        n_keep = gcfg["n_keep"]
        if len(passed) < n_keep:
            print(
                f"warning: {gen}: only {len(passed)} images pass, fewer than n_keep={n_keep}; keeping all")
            chosen = passed
        else:
            rng = np.random.default_rng(
                stable_seed(cfg["seed"], "select", gen))
            chosen = list(rng.choice(passed, size=n_keep, replace=False))
        fakes.loc[fakes["image_id"].isin(chosen), "keep"] = True
        fakes.loc[rows & (fakes["reason"] == "pass") & ~
                  fakes["keep"], "reason"] = "pass_not_sampled"

    cols = ["image_id", "path", "generator", "p_car",
            "p_damage", "p_closeup", "reason", "keep"]
    return fakes[[c for c in cols if c in fakes.columns]].reset_index(drop=True), t


def main() -> None:
    cfg = load_config()
    fcfg = cfg["filter"]
    scores = pd.read_csv(PROJECT_ROOT / fcfg["scores_csv"])
    sel, t = select_fakes(scores, cfg)

    out = PROJECT_ROOT / fcfg["selection_csv"]
    sel.to_csv(out, index=False)
    print(
        f"Thresholds: p_car >= {t['car']:.3f} (fixed), p_damage >= {t['damage']:.3f} (5% quantile of real photos)\n")
    print("Reasons per generator:")
    print(pd.crosstab(sel["generator"], sel["reason"], margins=True), "\n")
    print("Kept per generator:")
    print(sel[sel["keep"]].groupby("generator").size().to_string())
    print(f"\nSaved {len(sel)} rows to {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
