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
    compute x,y and r uncertainties, combining independent contributions in
    quadrature (GUM-style uncertainty propagation)

    baseline (always present): σ_xy = 1/width (1 pixel edge localization
    limit), σ_r,base = √2 × σ_xy (diameter → radius propagation of two
    independent unit-pixel limits)

    per artifact type, on top of the baseline:
    - blur: exact closed form fit to real Zenodo ground truth,
      σ_r(S) = σ_xy × √(2 + S²) — a symmetric blur kernel keeps a
      centroid-style position estimate unbiased, so σ_xy is untouched and
      only σ_r grows.
    - noise: additive pixel noise perturbs every pixel a position/radius
      estimate is built from, unlike blur — no Zenodo ground truth exists
      to fit this exactly, so it's a first-principles stand-in
      (h(N) = a×N^b pixel-equivalents, i.e. /width like the baseline)
      added in quadrature to *both* σ_xy and σ_r.
    - illumination: a non-uniform gradient is a GUM Type B *systematic*
      bias (photogrammetry literature: "eccentricity error"), not random
      spread — bounded by a first-principles stand-in ε(S) = a×S^b
      pixel-equivalents and converted to a standard uncertainty via the
      GUM rectangular-distribution rule (u = ε/√3, since only the bound is
      known, not a distribution shape), then added in quadrature to both
      σ_xy and σ_r.

    both non-blur scalings are unvalidated first-principles assumptions (no
    Zenodo ground truth exists for noise/illumination variants), not fits.
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

    if artifact_type == 'noise':
        artifact_params = params.get('noise', {'a': 0.5, 'b': 1.0})
        a = artifact_params.get('a', 0.5)
        b = artifact_params.get('b', 1.0)
        noise_term = a * (S ** b) / width  # pixel-equivalents -> normalized
        sigma_xy = np.sqrt(base_xy ** 2 + noise_term ** 2)
        sigma_r = np.sqrt(base_r ** 2 + noise_term ** 2)
        return sigma_xy, sigma_r

    if artifact_type == 'illumination':
        artifact_params = params.get('illumination', {'a': 10.0, 'b': 1.0})
        a = artifact_params.get('a', 10.0)
        b = artifact_params.get('b', 1.0)
        bound = a * (S ** b) / width  # pixel-equivalents -> normalized
        u_illum = bound / np.sqrt(3)
        sigma_xy = np.sqrt(base_xy ** 2 + u_illum ** 2)
        sigma_r = np.sqrt(base_r ** 2 + u_illum ** 2)
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