# Synthetic Hexgrid Generator

Python implementation of a configurable synthetic dataset generator for circle detection tasks, developed for Position 26-93-8C.

## Overview

This tool generates grayscale images (TIF format) with circles arranged on a hexagonal grid, plus configurable image artifacts and ground-truth metadata (JSON). Designed for training and testing AI models in metrological applications.

**Features:**
- Configurable hexagonal grid generation (circle radius, spacing, density)
- Random position jitter (matching Zenodo-style variation)
- 4 artifact types: noise (Gaussian + impulse), blur (Gaussian + motion), uneven illumination, missing circles
- Physics-informed uncertainty scaling (derived from Zenodo data for blur)
- TIF image output + JSON ground truth with uncertainties
- Supports random artifact mixing (training) or single-artifact datasets (testing)

---

## Installation & Setup

### Requirements
- Python 3.11+
- Git

### Installation

```bash
# Clone repository
git clone <repository-url>
cd synthetic_hexgrid_generator

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
Listed in `requirements.txt`:
`numpy`, `opencv-python`, `scikit-image`, `tifffile`, `pyyaml`, `matplotlib`, `scipy`, `pytest`

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
python src/main.py --num 200 --out data_test_missing --artifact missing --strength 0.1
```

### Command Line Options

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--num` | Number of images to generate | `1000` |
| `--size` | Image size (width=height in pixels) | `256` |
| `--out` | Output directory | `data` |
| `--jitter` | Position jitter standard deviation (pixels) | `2.0` |
| `--artifact` | Force specific artifact type (clean/blur/noise/illumination/missing) | `None` (random mix) |
| `--strength` | Artifact strength (if forced) | Random within configured range |
| `--config` | Path to YAML config file | `configs/default_config.yaml` |
| `--seed` | Random seed for reproducibility | `42` |

---

## Output Format

### Image Files
- TIF format (grayscale, uint8)
- Filename: `image_XXXX.tif`

### Ground Truth JSON
Example structure matching Zenodo convention:
```json
{
  "image": "image_0000.tif",
  "domain": {"width": 1.0, "height": 1.0},
  "camera": {"model": "synthetic generator", "resolution": [256, 256]},
  "circles": [
    {
      "centre": {"x": 0.5, "y": 0.5},
      "radius": 0.049,
      "uncertainties": {"x": 0.0039, "y": 0.0039, "r": 0.0055}
    }
  ],
  "summary": {"mean_radius": 0.049, "std_radius": 0.0005}
}
```

---

## Uncertainty Model

Physics-informed formula (derived from Zenodo blur data):

- $\sigma_{xy} = 1.0 / 	ext{width}$ (1 pixel edge localization limit)
- $\sigma_r = \sqrt{2} 	imes (1 + a 	imes S^b) / 	ext{width}$ (geometry + artifact degradation)

Parameters ($a, b$) are configurable per artifact type in `configs/default_config.yaml`:
- For blur: $a=0.342, b=1.244$ (fitted to Zenodo data)
- Other artifacts: placeholder values (documented in config)

---

## Configuration

Edit `configs/default_config.yaml` to control:
- Artifact probabilities and strength ranges
- Uncertainty scaling parameters per artifact type

```yaml
noise_prob: 0.3
blur_prob: 0.3
illumination_prob: 0.3
missing_prob: 0.3

uncertainty:
  blur:   {a: 0.342, b: 1.244}  # fitted from Zenodo
  noise:  {a: 0.1,   b: 1.0}    # placeholder
  illumination: {a: 0.1, b: 1.0} # placeholder
  missing: {a: 0.1,  b: 1.0}    # placeholder
```

---

## Part B: AI System Planning

See `docs/part_b_cheat_sheet.md` for:
- U-Net architecture with uncertainty prediction heads
- Physics-informed loss function (NLL for uncertainty calibration)
- Metrological test plan (precision, recall, RMSE, z-score calibration)
- Role of synthetic data in training and certification

---

## Testing

Quick test to verify hexgrid generation:
```bash
python tests/test_generator.py
```

Generate small dataset to verify artifacts:
```bash
python src/main.py --num 10 --out data_test
```

---

## Verification of Correctness

- **Grid geometry:** Visual inspection of generated images confirms hexagonal lattice with no overlapping circles. Overlap check is implemented in `_circles_overlap()`.
- **Uncertainty scaling:** The physics-informed formula $\sigma_r = \sqrt{2} 	imes (1 + a 	imes S^b) / 	ext{width}$ was validated against Zenodo blur data (levels 0, 3, 6, 12). Fitted parameters $a=0.342, b=1.244$ reproduce Zenodo values with <4% error.
- **Artifacts:** Each artifact function was tested independently on sample images and verified visually. Combinations are applied sequentially with parameter ranges from config.
- **Ground truth:** JSON format matches Zenodo structure. Uncertainties are computed from configurable formulas, not hardcoded.

---

## AI Tools Used

| Tool | Purpose | Verification |
|---|---|---|
| DeepSeek | Assisted with implementation planning, code structure suggestions, physics-informed uncertainty derivation, and README drafting | All code manually reviewed and tested. Uncertainty formula validated against Zenodo dataset. Artifacts verified visually on generated samples. |

---

## Project Structure

```text
synthetic_hexgrid_generator/
├── src/
│   ├── generator/
│   │   └── core.py          # Hexgrid generation + overlap check
│   ├── artifacts/
│   │   └── disturbances.py  # Noise, blur, illumination, missing circles
│   ├── utils/
│   │   └── io_utils.py      # TIF save, JSON ground truth, uncertainty formula
│   └── main.py              # CLI entry point
├── configs/
│   └── default_config.yaml  # Artifact and uncertainty parameters
├── tests/
│   └── test_generator.py    # Quick test script
├── docs/
│   └── part_b_cheat_sheet.md # AI system planning document
├── requirements.txt
├── README.md
└── .gitignore
```

