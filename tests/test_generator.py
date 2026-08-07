"""
tests for the hexgrid generator and the thinning image/ground-truth contract
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


def test_baseline_thinning_is_mandatory_and_erased(tmp_path):
    """
    checks presence/absence at each circle's own center pixel rather than a
    blob count: tightly hex-packed circles can round together into one blob
    at render time regardless of thinning, so a blob-count mismatch doesn't
    tell you whether thinning itself is broken. a circle's center point can
    never fall inside a *different* circle's disk (ground truth never
    overlaps), so it's a safe probe point.
    """
    width, height = 128, 128
    radius_ratio = 0.049
    out_dir = tmp_path / "clean"

    generate_dataset(
        output_dir=str(out_dir), num_images=3,
        image_size=(width, height), radius_ratio=radius_ratio,
        force_artifact='clean', seed=11,
    )

    generator, _ = _make_generator(width=width, height=height, radius_ratio=radius_ratio)
    full_circles = generator.generate_grid()
    full_count = len(full_circles)

    def px(c):
        return int(c['centre']['x'] * width), int(c['centre']['y'] * height)

    full_positions = {(int(c.x), int(c.y)) for c in full_circles}

    for i in range(3):
        gt = json.loads((out_dir / f"image_{i:04d}.json").read_text())
        circles = gt['circles']
        img = load_tif(out_dir / f"image_{i:04d}.tif")

        assert len(circles) < full_count, \
            f"image {i}: rendered the full, un-thinned grid ({len(circles)} circles)"

        kept_positions = {px(c) for c in circles}

        for c in circles:
            x, y = px(c)
            assert img[y, x] > 0, f"image {i}: kept circle at ({x},{y}) is not drawn"

        for x, y in full_positions:
            if (x, y) in kept_positions:
                continue
            assert img[y, x] == 0, f"image {i}: removed circle at ({x},{y}) is still drawn"
