"""
file input/output utilities for tif and json
"""

import json
import numpy as np
import tifffile
from pathlib import Path
from typing import List, Dict, Any, Optional
from generator.core import Circle


def save_tif(image: np.ndarray, filepath: Path) -> None:
    """save grayscale image as tif"""
    if image.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    tifffile.imwrite(str(filepath), image, photometric='minisblack')


def load_tif(filepath: Path) -> np.ndarray:
    """load tif image as numpy array"""
    return tifffile.imread(str(filepath))


def compute_uncertainties(artifact_type: str, artifact_strength: float,
                          image_size: tuple, config: dict) -> tuple:
    """
    x,y,r uncertainties, combined in quadrature (GUM-style propagation)

    baseline: σ_xy = 1/width, σ_r = √2 × σ_xy (pixel discretization)

    blur: σ_r(S) = σ_xy × √(2 + S²), fit to real Zenodo ground truth; a
    symmetric kernel leaves σ_xy unchanged.

    noise/illumination: no detector or measurement exists to tell us how
    far off a circle estimate would actually be under these artifacts, so
    a×S is a guessed bound on that error, converted to a standard
    uncertainty via the GUM rectangular-distribution rule u = bound/√3
    (JCGM 100:2008 §4.3.7), then added in quadrature. unvalidated.
    """
    width, height = image_size

    base_xy = 1.0 / width
    base_r = np.sqrt(2) * base_xy

    S = artifact_strength
    params = config.get('uncertainty', {})

    if artifact_type == 'clean' or S == 0:
        return base_xy, base_r

    if artifact_type == 'blur':
        sigma_xy = base_xy
        sigma_r = base_xy * np.sqrt(2 + S ** 2)
        return sigma_xy, sigma_r

    if artifact_type in ('noise', 'illumination'):
        default_a = 0.5 if artifact_type == 'noise' else 10.0
        a = params.get(artifact_type, {}).get('a', default_a)
        bound = a * S / width
        u = bound / np.sqrt(3)
        sigma_xy = np.sqrt(base_xy ** 2 + u ** 2)
        sigma_r = np.sqrt(base_r ** 2 + u ** 2)
        return sigma_xy, sigma_r

    return base_xy, base_r


def circles_to_zenodo(circles: List[Circle], image_size: tuple,
                      artifact_type: str = 'clean',
                      artifact_strength: float = 0.0,
                      config: dict = None) -> dict:
    """
    convert circle objects to zenodo-compatible json format
    with physics-informed uncertainties
    """
    if config is None:
        config = {}
    
    width, height = image_size
    domain_aspect = height / width
    
    # compute uncertainties
    sigma_xy, sigma_r = compute_uncertainties(
        artifact_type, artifact_strength, image_size, config
    )
    
    circles_norm = []
    for c in circles:
        circles_norm.append({
            "centre": {
                "x": c.x / width,
                "y": c.y / height
            },
            "radius": c.r / width,
            "uncertainties": {
                "x": sigma_xy,
                "y": sigma_xy,
                "r": sigma_r
            }
        })
    
    # compute summary statistics
    radii = [c["radius"] for c in circles_norm]
    
    return {
        "domain": {
            "width": 1.0,
            "height": domain_aspect
        },
        "camera": {
            "model": "synthetic generator",
            "resolution": [width, height],
            "noise": 0.0
        },
        "circles": circles_norm,
        "summary": {
            "mean_radius": float(np.mean(radii)) if radii else 0.0,
            "std_radius": float(np.std(radii)) if radii else 0.0
        }
    }


def save_ground_truth(circles: List[Circle], image_name: str,
                      image_size: tuple, filepath: Path,
                      artifact_type: str = 'clean',
                      artifact_strength: float = 0.0,
                      config: dict = None) -> None:
    """
    save ground truth in zenodo-compatible json format
    """
    data = circles_to_zenodo(circles, image_size, artifact_type,
                             artifact_strength, config)
    data["image"] = image_name
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def generate_filename(base: str, index: int, ext: str) -> str:
    """generate zero-padded filename, e.g. image_001.tif"""
    return f"{base}_{index:04d}.{ext}"