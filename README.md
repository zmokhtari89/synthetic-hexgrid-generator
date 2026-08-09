# Synthetic Hexgrid Generator

Python implementation of a configurable synthetic dataset generator for circle detection tasks, developed for Position 26-93-8C.

## Overview

This tool generates grayscale images (TIF format) with circles arranged on a hexagonal grid, plus configurable image artifacts and ground-truth metadata (JSON). Designed for training and testing AI models in metrological applications.

**Features:**

- Configurable hexagonal grid generation (circle radius, spacing, margin, density), with radius either fixed per dataset or drawn per image from a range
- Optional random position jitter (off by default, since real Zenodo circles sit exactly on the grid)
- Mandatory grid thinning (a random 20-40% of grid positions removed every image, matching real Zenodo's subject model — not an optional artifact)
- 3 pixel-level artifact types: noise (Gaussian + impulse), blur (Gaussian + motion), uneven illumination
- Per-artifact uncertainty scaling for x, y, and radius
- TIF image output (anti-aliased, sub-pixel circle rendering) + JSON ground truth with uncertainties
- Supports random artifact mixing (training) or single-artifact datasets (testing)

---

## Installation & Setup

### Requirements
- Python 3.11+
- Git

### Installation

```bash
# Clone repository
git clone git@github.com:zmokhtari89/synthetic-hexgrid-generator.git
cd synthetic_hexgrid_generator

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Dependencies are listed in `requirements.txt`.

---

## Usage

### Generate Training Dataset (Random Artifact Mix)
```bash
python src/main.py --num 1000 --out data_train
```

### Generate Test Datasets (Single Artifact, like Zenodo)
```bash
# Clean (no artifacts)
python src/main.py --num 200 --out data_test_clean --artifact clean

# Blur at different levels
python src/main.py --num 200 --out data_test_blur_03 --artifact blur --strength 3
python src/main.py --num 200 --out data_test_blur_06 --artifact blur --strength 6
python src/main.py --num 200 --out data_test_blur_12 --artifact blur --strength 12

# Other artifacts
python src/main.py --num 200 --out data_test_noise --artifact noise --strength 5
```

### Command Line Options

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--num` | Number of images to generate | `1000` |
| `--size` | Image size (width=height in pixels) | `256` |
| `--out` | Output directory | `data` |
| `--jitter` | Position jitter standard deviation (pixels) | config value, else `0.0` |
| `--radius-ratio` | Circle radius as a fraction of image width, fixed for the whole dataset | config value, else `0.049` |
| `--radius-ratio-min` / `--radius-ratio-max` | Draw a different radius per image from this range instead (overrides `--radius-ratio` when both are set) | config value, else unset |
| `--spacing-ratio` | Center-to-center spacing as a fraction of the max spacing that still avoids overlap (must be `< 0.5`) | config value, else `0.48` |
| `--margin` | Clear border kept free of circles, in pixels | config value, else `15.0` |
| `--artifact` | Force specific artifact type (clean/blur/noise/illumination) | `None` (random mix) |
| `--strength` | Artifact strength (if forced) | Random within configured range |
| `--config` | Path to YAML config file | `configs/default_config.yaml` |
| `--seed` | Random seed for reproducibility | `42` |

A geometry flag (`--jitter`, `--radius-ratio*`, `--spacing-ratio`, `--margin`) is resolved in this order: CLI flag, then the `generator:` section of the config file, then the hardcoded default shown above.

---

## Output Format

### Image Files
- TIF format (grayscale, uint8)
- Filename: `image_XXXX.tif`

**Error tolerance:** at the default 256×256 resolution, a circle's position can't be pinned down more precisely than one pixel (1/256 ≈ 0.39% of image width), and radius inherits √2 times that, since it depends on two independent edge positions. This is the baseline value in every ground-truth file's `uncertainties` field (see Uncertainty Model below) — a detection shouldn't be judged wrong for missing by less than this amount. It's a simplified stand-in for the more detailed discretization error a real camera model would have; it is not meant to reproduce that in full.

### Ground Truth JSON
Example structure matching Zenodo convention:
```json
{
  "domain": {"width": 1.0, "height": 1.0},
  "camera": {"model": "synthetic generator", "resolution": [256, 256], "noise": 0.0},
  "circles": [
    {
      "centre": {"x": 0.5, "y": 0.5},
      "radius": 0.049,
      "uncertainties": {"x": 0.0039, "y": 0.0039, "r": 0.0055}
    }
  ],
  "summary": {"mean_radius": 0.049, "std_radius": 0.0005},
  "image": "image_0000.tif"
}
```

---

## Uncertainty Model

Each circle's ground truth includes an uncertainty for x, y, and r, computed in `compute_uncertainties()` (`src/utils/io_utils.py`). Independent sources of error are combined by taking the square root of the sum of their squares, the standard way to combine unrelated error sources: if two error sources are unrelated, their combined spread is smaller than just adding them, but larger than the biggest one alone.

**Baseline (always present):** every position estimate is limited by pixel discretization, so `σ_xy = 1 pixel`; a radius is derived from two independent edge positions, so `σ_r = √2 × σ_xy`.

**On top of the baseline, per artifact:**

- **Blur:** `σ_r = σ_xy × √(2 + S²)`, a form found by fitting to the real Zenodo blur measurements. It also matches how blur should behave physically: blurring smears the circle's edge outward without shifting its center on average, so position stays about as precise as the baseline while radius gets harder to pin down as blur increases.
- **Noise & illumination:** neither has real measurements to fit against, so both use the same guess: `a × S` is a bound on how far off a detector's estimate could plausibly be, converted to a standard uncertainty via the GUM rectangular-distribution rule (`u = bound/√3`, JCGM 100:2008 §4.3.7), then added in quadrature.

---

## Configuration

Edit `configs/default_config.yaml` to control:

- Grid geometry (radius, spacing, margin, jitter) — see the `generator:` section
- Artifact probabilities and strength ranges
- Uncertainty scaling parameters per artifact type

```yaml
generator:
  radius_ratio: 0.049
  radius_ratio_min: 0.036
  radius_ratio_max: 0.050
  spacing_ratio: 0.48
  margin: 15.0
  jitter_std: 0.0
  thinning_min: 0.2
  thinning_max: 0.4

noise_prob: 0.3
blur_prob: 0.3
illumination_prob: 0.3

uncertainty:
  noise:         {a: 0.5}   # first-principles stand-in
  illumination:  {a: 10.0}  # first-principles stand-in
```

---

## Part B: AI System Planning

See [`part_b/part_b_cheat_sheet.md`](part_b/part_b_cheat_sheet.md) for:

- A plan for an AI system that predicts circle parameters and their uncertainties
- A certification test plan for a black-box detection model, with metrics, acceptance criteria, and known limits
- The role this generated data plays in both of the above

---

## Testing

Run the automated test suite with pytest:
```bash
pytest tests/test_generator.py
```

It checks:

- Generated circles never overlap in the ground truth
- Circle count stays within the configured range
- The same seed reproduces the same output
- Grid thinning is applied to every image, and thinned-out circles are actually absent from the rendered image while every kept circle is present

Generate a small dataset by hand to inspect visually:

```bash
python src/main.py --num 10 --out data_test
```

---

## Known Limitations

- Non-overlap is guaranteed exactly in the ground-truth coordinates, but not pixel-for-pixel in the rendered image: very close circles can round to touching or overlapping pixels, and this can worsen once blur is applied.
- The noise and illumination uncertainty formulas are first-principles assumptions, not fits to measured data (unlike blur, where real Zenodo measurements were available).
- Synthetic artifacts approximate but do not fully replicate real camera noise, optics, and lighting.

---

## AI Tools Used

| Tool | Purpose | Verification |
|---|---|---|
| DeepSeek | Initial implementation: hexgrid generation, artifact pipeline, and ground-truth format | Manually reviewed and tested before further changes were built on top of it |
| Claude Code | Bug fixes, added test coverage and configuration options, rewrote the uncertainty formula, and wrote this documentation | All code manually reviewed and tested. The blur uncertainty formula was validated against Zenodo dataset measurements. Artifacts and rendering were checked visually and with the automated test suite. |

---

## Project Structure

```text
synthetic_hexgrid_generator/
├── src/
│   ├── generator/
│   │   └── core.py          # Hexgrid generation + overlap check
│   ├── artifacts/
│   │   └── disturbances.py  # Noise, blur, illumination
│   ├── utils/
│   │   └── io_utils.py      # TIF save, JSON ground truth, uncertainty formula
│   └── main.py               # CLI entry point
├── configs/
│   └── default_config.yaml  # Grid geometry, artifact, and uncertainty parameters
├── tests/
│   └── test_generator.py    # Automated test suite (pytest)
├── part_b/
│   ├── part_b_cheat_sheet.md  # AI system planning document
│   └── part_b_cheat_sheet.pdf
├── requirements.txt
├── README.md
└── .gitignore
```
