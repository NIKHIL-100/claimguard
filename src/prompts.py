"""Step 2.3b: build varied, realistic prompts for the text-to-image generators.

Lessons from the smoke tests:
Step 2.3a (SD 1.5, hand-written prompts)
1. Name the car in every prompt: long detailed prompts made SD 1.5 drop the car entirely,
   and a fake with no car is a content shortcut (every real photo shows a car).
2. Keep prompts SHORT: SD 1.5's text encoder reads at most 77 tokens.
Step 2.3c (SD 1.5 vs SDXL, prompt builder v1)
3. "close-up ... of a <car> with <damage>" still produced WHOLE cars: the car was the
   subject, so the models centred it. Close-ups now make the DAMAGED PART the subject:
   "close-up ... of a large dent on the rear bumper of a white Toyota sedan".
4. Both models drifted to vintage American cars in American garages. Real VehiDE photos
   show modern Asian-market cars, so prompts name a modern make (Toyota, Hyundai, ...)
   and the negative prompt rules out vintage/classic cars and professional photography.

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
CAR_MAKES = ["Toyota", "Hyundai", "Kia", "Honda",
             "Mazda", "Ford", "Mitsubishi", "Chevrolet"]

# (damage, part) pairs that make physical sense together.
DAMAGES = [
    ("a large dent", "rear bumper"), ("a dent",
                                      "front bumper"), ("deep scratches", "side door"),
    ("scraped paint", "front bumper"), ("a cracked lens",
                                        "headlight"), ("smashed pieces", "tail light"),
    ("crumpled metal", "front end"), ("a deep dent",
                                      "rear fender"), ("a long crack", "windshield"),
    ("a broken housing", "side mirror"), ("a dent",
                                          "front door"), ("scratches and a dent", "rear door"),
    ("a dent", "hood"), ("long scratches",
                         "rear bumper"), ("a large crack", "front bumper"),
    ("a dent", "trunk lid"), ("deep scratches",
                              "front fender"), ("a broken cover", "fog light"),
    ("a dent", "wheel arch"), ("scrapes", "wheel rim"), ("a crack", "rear bumper"),
    ("chipped paint", "door edge"),
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

NEGATIVE = ("cartoon, illustration, painting, 3d render, cgi, studio lighting, professional photo, "
            "glossy, oversaturated, vintage car, classic car, watermark, text, logo, blurry, deformed car")


def make_prompt(seed: int, index: int) -> dict:
    """The prompt for position `index`. Depends only on (seed, index)."""
    # A string seed is hashed with SHA-512 by Python's random module, so this is stable
    # across runs, machines and Python versions (unlike the built-in hash()).
    rng = random.Random(f"claimguard-{seed}-{index}")
    colour = rng.choice(CAR_COLOURS)
    make = rng.choice(CAR_MAKES)
    car_type = rng.choice(CAR_TYPES)
    damage, part = rng.choice(DAMAGES)
    framing = rng.choices(list(FRAMINGS), weights=list(FRAMINGS.values()))[0]
    style = rng.choice(STYLES)
    scene = rng.choice(SCENES)
    lighting = rng.choice(LIGHTING)
    extras = [e for e, p in EXTRAS.items() if rng.random() < p]

    car = f"a {colour} {make} {car_type}"
    if framing == "close-up":
        # The damaged part is the subject, so the model frames the damage, not the whole car.
        prompt = f"close-up {style} of {damage} on the {part} of {car}, filling the frame, {lighting}"
    else:
        prompt = f"{framing} {style} of {car} with {damage} on the {part}, {scene}, {lighting}"
    if extras:
        prompt += ", " + ", ".join(extras)
    return {
        "index": index,
        "prompt": prompt,
        "negative_prompt": NEGATIVE,
        # seed for the generator's starting noise
        "image_seed": rng.randrange(2**32),
        "colour": colour, "make": make, "car_type": car_type, "damage": damage, "part": part,
        "framing": framing, "scene": "" if framing == "close-up" else scene,
        "lighting": lighting, "extras": ";".join(extras),
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
