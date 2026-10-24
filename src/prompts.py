"""Step 2.3b: build varied, realistic prompts for the text-to-image generators.

Lessons from the SD 1.5 smoke test (Step 2.3a):
1. Name the car FIRST and always ("... of a white sedan with ..."): long detailed prompts
   made SD 1.5 drop the car entirely, and a fake with no car is a content shortcut
   (every real photo shows a car).
2. Keep prompts SHORT: one damage, one scene, one lighting phrase. SD 1.5's text encoder
   reads at most 77 tokens and loses track of the subject in long prompts.
3. Match the REAL photos: mostly close-ups (VehiDE is mostly close-up inspection shots),
   garages and Asian city streets, flat everyday lighting, sometimes assessor pen marks.

Every prompt is fully determined by (seed, index), so:
- the dataset is reproducible, and
- a generation run can stop and resume without changing earlier prompts.
The same prompt list is used for every generator, so all generators get the SAME content
mix; differences between them are then about the generator, not the scene.

Try it:  python -m src.prompts --n 10
"""
from __future__ import annotations

import argparse
import random

CAR_COLOURS = ["white", "silver", "grey",
               "black", "red", "blue", "dark blue", "beige"]
CAR_TYPES = ["sedan", "hatchback", "SUV",
             "pickup truck", "minivan", "compact car"]

# Each entry is a complete, physically sensible "damage on a part" phrase.
DAMAGES = [
    "a large dent in the rear bumper", "a dented front bumper", "deep scratches along the side door",
    "scraped paint on the front bumper", "a cracked headlight", "a smashed tail light",
    "a crushed front end", "a dented rear fender", "a cracked windshield", "a broken side mirror",
    "a dented door", "scratches and a dent on the rear door", "a dented hood", "a scratched rear bumper",
    "a cracked front bumper hanging loose", "a dented trunk lid", "deep scratches on the front fender",
    "a broken fog light", "a dented wheel arch", "a scratched wheel rim", "a cracked rear bumper",
    "paint chipped off the door edge",
]

# Weights follow the real data: VehiDE is mostly close-up inspection photos.
FRAMINGS = {"close-up": 0.5, "medium shot": 0.3, "wide shot": 0.2}
STYLES = ["amateur phone photo", "smartphone photo", "phone photo"]
SCENES = [
    "in a car repair garage", "in a garage with a tiled floor", "parked on a city street",
    "parked on a street in Vietnam", "in a parking lot", "in an outdoor car yard",
    "parked beside a building",
]
LIGHTING = ["daylight", "overcast daylight",
            "fluorescent light", "evening light"]
# assessor marks, as in VehiDE
EXTRAS = {"blue marker pen marks around the damage": 0.2}

NEGATIVE = ("cartoon, illustration, painting, 3d render, cgi, studio lighting, glossy, "
            "oversaturated, watermark, text, logo, blurry, deformed car")


def make_prompt(seed: int, index: int) -> dict:
    """The prompt for position `index`. Depends only on (seed, index)."""
    # A string seed is hashed with SHA-512 by Python's random module, so this is stable
    # across runs, machines and Python versions (unlike the built-in hash()).
    rng = random.Random(f"claimguard-{seed}-{index}")
    colour = rng.choice(CAR_COLOURS)
    car_type = rng.choice(CAR_TYPES)
    damage = rng.choice(DAMAGES)
    framing = rng.choices(list(FRAMINGS), weights=list(FRAMINGS.values()))[0]
    style = rng.choice(STYLES)
    scene = rng.choice(SCENES)
    lighting = rng.choice(LIGHTING)
    extras = [e for e, p in EXTRAS.items() if rng.random() < p]

    prompt = f"{framing} {style} of a {colour} {car_type} with {damage}, {scene}, {lighting}"
    if extras:
        prompt += ", " + ", ".join(extras)
    return {
        "index": index,
        "prompt": prompt,
        "negative_prompt": NEGATIVE,
        # seed for the generator's starting noise
        "image_seed": rng.randrange(2**32),
        "colour": colour, "car_type": car_type, "damage": damage, "framing": framing,
        "scene": scene, "lighting": lighting, "extras": ";".join(extras),
    }


def make_prompts(n: int, seed: int = 42) -> list[dict]:
    """The first n prompts. make_prompts(k) is always the start of make_prompts(n) for k <= n."""
    return [make_prompt(seed, i) for i in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for p in make_prompts(args.n, args.seed):
        print(f"[{p['index']:3d}] {p['prompt']}")


if __name__ == "__main__":
    main()
