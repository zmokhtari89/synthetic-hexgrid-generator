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
    compute x,y and r uncertainties based on physics-informed formula
    
    σ_xy = 1.0 / width  (1 pixel edge localization limit)
    σ_r = √2 × (1 + a × S^b) / width  (geometry + artifact degradation)
    
    where:
    - √2 comes from diameter → radius propagation
    - S is artifact strength
    - a, b are artifact-specific parameters
    """
    width, height = image_size
    
    # base uncertainty: 1 pixel in normalized coords
    base_xy = 1.0 / width
    
    # get artifact parameters from config
    params = config.get('uncertainty', {})
    artifact_params = params.get(artifact_type, {'a': 0.0, 'b': 1.0})
    a = artifact_params.get('a', 0.0)
    b = artifact_params.get('b', 1.0)
    
    # clean case: no artifacts
    if artifact_type == 'clean' or artifact_strength == 0:
        r_factor = 1.0
    else:
        r_factor = 1 + a * (artifact_strength ** b)
    
    # radius uncertainty: √2 × (1 + a×S^b) / width
    sigma_r = np.sqrt(2) * base_xy * r_factor
    
    return base_xy, sigma_r


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