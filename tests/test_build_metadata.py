"""Automated checks for the split logic. Run with:  pytest -q

These tests encode the two rules that protect our evaluation from being misleading:
1. no group (a real photo + its edited copies) appears in more than one split  -> no leakage
2. every image from the hold-out generator is in test_holdout                   -> fair generalisation test
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
IGNORE = shutil.ignore_patterns(
    "data", ".venv", ".git", "features", "checkpoints", "__pycache__", ".pytest_cache")


def build_copy(tmp_dir: Path) -> pd.DataFrame:
    """Copy the project, add dummy images, run build_metadata, return the CSV."""
    work = tmp_dir / "claimguard"
    shutil.copytree(PROJECT, work, ignore=IGNORE)
    subprocess.run([sys.executable, "scripts/make_dummy_data.py",
                   str(work)], cwd=work, check=True)
    subprocess.run([sys.executable, "-m", "src.build_metadata"],
                   cwd=work, check=True, capture_output=True)
    return pd.read_csv(work / "data" / "metadata.csv")


@pytest.fixture(scope="module")
def metadata(tmp_path_factory) -> pd.DataFrame:
    # Work on a copy of the project so real data is never touched.
    return build_copy(tmp_path_factory.mktemp("proj"))


def test_every_image_listed(metadata):
    assert len(metadata) == 40
    assert set(metadata["label"]) == {0, 1}


def test_no_group_crosses_splits(metadata):
    assert (metadata.groupby("group_id")["split"].nunique() == 1).all()


def test_inpaint_shares_split_with_its_source_photo(metadata):
    by_path = metadata.set_index("path")["split"]
    for i in range(6):
        assert by_path[f"data/fake/sd15_inpaint/car_{i}__dent.png"] == by_path[f"data/real/car_{i}.jpg"]


def test_holdout_generator_never_in_training(metadata):
    flux = metadata[metadata["generator"].str.startswith("flux")]
    assert (flux["split"] == "test_holdout").all()
    assert not metadata.loc[metadata["split"].isin(
        ["train", "val"]), "generator"].str.startswith("flux").any()


def test_split_is_reproducible(metadata, tmp_path):
    # Same seed -> identical split on a second run.
    again = build_copy(tmp_path)
    assert metadata["split"].tolist() == again["split"].tolist()
