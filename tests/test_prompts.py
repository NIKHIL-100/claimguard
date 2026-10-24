"""Checks for src/prompts.py. Run with:  pytest -q"""
from collections import Counter

from src.prompts import CAR_TYPES, make_prompts


def test_same_seed_gives_same_prompts():
    assert make_prompts(50, seed=42) == make_prompts(50, seed=42)


def test_different_seed_gives_different_prompts():
    a = [p["prompt"] for p in make_prompts(50, seed=1)]
    b = [p["prompt"] for p in make_prompts(50, seed=2)]
    assert a != b


def test_resume_safe_prefix():
    # Generating 100 then 300 must not change the first 100 (lets a run stop and resume).
    assert make_prompts(300)[:100] == make_prompts(100)


def test_car_is_always_named():
    for p in make_prompts(500):
        assert any(t in p["prompt"] for t in CAR_TYPES), p["prompt"]


def test_prompts_are_short():
    # SD 1.5's text encoder reads at most 77 tokens; ~40 words stays safely under it.
    assert max(len(p["prompt"].split()) for p in make_prompts(1000)) <= 40


def test_prompts_are_varied():
    prompts = [p["prompt"] for p in make_prompts(1000)]
    assert len(set(prompts)) >= 990


def test_framing_mix_matches_plan():
    counts = Counter(p["framing"] for p in make_prompts(2000))
    assert 0.45 < counts["close-up"] / 2000 < 0.55
