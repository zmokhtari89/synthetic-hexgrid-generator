"""
tests for the hexgrid generator and the missing-circles image/ground-truth
contract
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import numpy as np
from generator.core import HexgridGenerator, HexgridConfig
from utils.io_utils import load_tif
from main import generate_dataset


def _make_generator(width=256, height=256, radius_ratio=0.049, margin=15.0, jitter_std=0.0):
    circle_radius = width * radius_ratio
    config = HexgridConfig(
        image_size=(width, height),
        circle_radius=circle_radius,
        margin=margin,
        jitter_std=jitter_std,
    )
    return HexgridGenerator(config), config


def test_no_overlap():
    generator, _ = _make_generator(jitter_std=2.0)
    circles = generator.generate_grid()

    assert len(circles) > 0
    for i, c1 in enumerate(circles):
        for c2 in circles[i + 1:]:
            dist = np.hypot(c1.x - c2.x, c1.y - c2.y)
            assert dist >= c1.r + c2.r - 1e-6, f"circles overlap: {c1} vs {c2}"


def test_circle_count_in_range():
    generator, config = _make_generator()
    circles = generator.generate_grid()

    width, height = config.image_size
    usable_area = (width - 2 * config.margin) * (height - 2 * config.margin)
    hex_cell_area = config.spacing ** 2 * np.sqrt(3) / 2
    expected = usable_area / hex_cell_area

    assert 0.5 * expected <= len(circles) <= 1.5 * expected


def test_same_seed_reproducibility():
    generator, _ = _make_generator(jitter_std=2.0)

    np.random.seed(123)
    circles_a = generator.generate_grid()

    np.random.seed(123)
    circles_b = generator.generate_grid()

    assert len(circles_a) == len(circles_b)
    for c1, c2 in zip(circles_a, circles_b):
        assert c1.x == c2.x and c1.y == c2.y and c1.r == c2.r


def test_missing_circles_are_absent_from_image(tmp_path):
    """
    the actual contract to protect: the 'missing' artifact must genuinely
    erase removed circles from the image, not just drop them from the json
    while still drawing them (the original bug in item 0/5 of
    docs/task_plan_reviewed.md).

    this replaces a blob-count check (cv2.connectedComponents count vs.
    len(gt['circles'])) that tested the wrong thing: tightly hex-packed
    circles can round together into one blob at render time regardless of
    the missing-circle logic (see item 4 in the plan), so a blob-count
    mismatch doesn't tell you whether the missing-circle contract itself is
    broken. checking presence/absence at each circle's own center pixel
    avoids that confound entirely: since ground-truth circles never overlap,
    a circle's center point can never fall inside a *different* circle's
    disk, so it's a safe point to probe regardless of any rounding/fusion
    elsewhere in the image.
    """
    width, height = 128, 128
    seed = 7

    for i in range(5):
        clean_dir = tmp_path / f"clean_{i}"
        missing_dir = tmp_path / f"missing_{i}"

        # same seed -> the full (pre-thinning) circle list is bit-for-bit
        # identical between the two runs, since thinning is the first thing
        # that consumes randomness *after* grid generation
        generate_dataset(
            output_dir=str(clean_dir), num_images=1,
            image_size=(width, height), force_artifact='clean', seed=seed + i,
        )
        generate_dataset(
            output_dir=str(missing_dir), num_images=1,
            image_size=(width, height), force_artifact='missing',
            force_strength=0.3, seed=seed + i,
        )

        full_circles = json.loads((clean_dir / "image_0000.json").read_text())['circles']
        missing_gt = json.loads((missing_dir / "image_0000.json").read_text())
        kept_circles = missing_gt['circles']
        img = load_tif(missing_dir / "image_0000.tif")

        assert len(kept_circles) < len(full_circles), \
            "missing artifact did not remove any circles for this seed"

        def px(c):
            return int(c['centre']['x'] * width), int(c['centre']['y'] * height)

        kept_positions = {px(c) for c in kept_circles}

        for c in kept_circles:
            x, y = px(c)
            assert img[y, x] > 0, \
                f"seed {seed + i}: kept circle at ({x},{y}) is not drawn in the image"

        for c in full_circles:
            pos = px(c)
            if pos in kept_positions:
                continue
            x, y = pos
            assert img[y, x] == 0, \
                f"seed {seed + i}: removed circle at ({x},{y}) is still drawn in the image"
