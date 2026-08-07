"""
main entry point for dataset generation with artifacts
"""

import argparse
import numpy as np
import cv2
import yaml
from pathlib import Path
from typing import List, Optional
from generator.core import HexgridGenerator, HexgridConfig, Circle
from artifacts.disturbances import ArtifactPipeline
from utils.io_utils import save_tif, save_ground_truth, generate_filename


def create_image_from_circles(circles: List[Circle], image_size: tuple) -> np.ndarray:
    """render circles onto a blank image, anti-aliased and at sub-pixel precision"""
    width, height = image_size
    img = np.zeros((height, width), dtype=np.uint8)
    # cv2.circle's `shift` param takes fixed-point coordinates: multiply by
    # 2**shift and round, instead of the old int() truncation, so a circle
    # at x=10.7 is actually drawn at 10.7, not snapped to 10
    shift = 4
    scale = 1 << shift
    for c in circles:
        center = (int(round(c.x * scale)), int(round(c.y * scale)))
        radius = int(round(c.r * scale))
        cv2.circle(img, center, radius, 255, -1, lineType=cv2.LINE_AA, shift=shift)
    return img


def generate_dataset(output_dir: str, num_images: int = 1000,
                     image_size: tuple = (256, 256),
                     radius_ratio: float = 0.049,
                     radius_ratio_min: Optional[float] = None,
                     radius_ratio_max: Optional[float] = None,
                     spacing_ratio: float = 0.48,
                     jitter_std: float = 0.0,
                     margin: float = 15.0,
                     thinning_min: float = 0.2,
                     thinning_max: float = 0.4,
                     artifact_config: dict = None,
                     force_artifact: Optional[str] = None,
                     force_strength: Optional[float] = None,
                     seed: int = 42) -> None:

    np.random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    width, height = image_size

    # real Zenodo data draws one radius per image from a range, not one
    # fixed radius for the whole dataset (see docs/task_plan_reviewed.md
    # item 3) - only kicks in when both bounds are given, otherwise every
    # image uses the single fixed radius_ratio, as before
    draw_radius_per_image = radius_ratio_min is not None and radius_ratio_max is not None

    pipeline = ArtifactPipeline(artifact_config or {})

    print(f"generating {num_images} images")
    print(f"image size: {width}x{height}")
    if draw_radius_per_image:
        print(f"circle radius ratio: drawn per image from [{radius_ratio_min}, {radius_ratio_max}]")
    else:
        print(f"circle radius: {width * radius_ratio:.1f} px")
    print(f"jitter std: {jitter_std:.1f} px")
    print(f"baseline thinning: {thinning_min:.0%}-{thinning_max:.0%} of grid removed every image")

    if force_artifact:
        print(f"forcing artifact: {force_artifact} (strength: {force_strength or 'random'})")
    else:
        print("artifact mode: random mix")

    for i in range(num_images):
        # a new radius per image means a new spacing too (spacing derives
        # from radius via spacing_ratio), so the config/generator has to be
        # rebuilt every image rather than reused across the whole dataset
        image_radius_ratio = np.random.uniform(radius_ratio_min, radius_ratio_max) if draw_radius_per_image else radius_ratio
        circle_radius = width * image_radius_ratio

        config = HexgridConfig(
            image_size=image_size,
            circle_radius=circle_radius,
            margin=margin,
            spacing_ratio=spacing_ratio,
            jitter_std=jitter_std
        )
        generator = HexgridGenerator(config)

        # generate circles with random jitter (unique per image)
        full_circles = generator.generate_grid()

        # mandatory baseline thinning, every image (matches real Zenodo)
        baseline_keep_prob = np.random.uniform(1.0 - thinning_max, 1.0 - thinning_min)
        circles = generator.random_subset(full_circles, baseline_keep_prob)

        # render the image from the exact circles reported in the ground truth
        clean_image = create_image_from_circles(circles, image_size)

        # apply pixel-level artifacts (noise/blur/illumination) on top
        if force_artifact:
            degraded_image, _, artifact_type, artifact_strength = pipeline.apply_forced(
                clean_image, circles, force_artifact, force_strength
            )
        else:
            degraded_image, _, artifact_type, artifact_strength = pipeline.apply(clean_image, circles)

        # save image
        img_name = generate_filename('image', i, 'tif')
        img_path = output_dir / img_name
        save_tif(degraded_image, img_path)

        # save ground truth with physics-informed uncertainties
        gt_name = generate_filename('image', i, 'json')
        gt_path = output_dir / gt_name
        save_ground_truth(
            circles,
            img_name,
            image_size,
            gt_path,
            artifact_type=artifact_type,
            artifact_strength=artifact_strength,
            config=artifact_config
        )

        if (i + 1) % 100 == 0:
            artifacts_str = ', '.join(pipeline.applied_artifacts) if pipeline.applied_artifacts else 'clean'
            print(f"  generated {i+1}/{num_images} (artifacts: {artifacts_str})")
    
    print(f"done! saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate synthetic hexgrid dataset')
    parser.add_argument('--num', type=int, default=1000, help='number of images')
    parser.add_argument('--size', type=int, default=256, help='image size (width=height)')
    parser.add_argument('--out', type=str, default='data', help='output directory')
    parser.add_argument('--jitter', type=float, default=None,
                       help='position jitter std, pixels (default: config or 0.0)')
    parser.add_argument('--radius-ratio', type=float, default=None,
                       help='circle radius as a fraction of image width, fixed for the whole '
                            'dataset (default: config or 0.049; ignored if --radius-ratio-min/max '
                            'are set)')
    parser.add_argument('--radius-ratio-min', type=float, default=None,
                       help='lower bound of the per-image radius ratio range (default: config; '
                            'requires --radius-ratio-max too)')
    parser.add_argument('--radius-ratio-max', type=float, default=None,
                       help='upper bound of the per-image radius ratio range (default: config; '
                            'requires --radius-ratio-min too)')
    parser.add_argument('--spacing-ratio', type=float, default=None,
                       help='center-to-center spacing as a fraction of the max non-overlap '
                            'spacing, must be <0.5 (default: config or 0.48)')
    parser.add_argument('--margin', type=float, default=None,
                       help='clear border kept free of circles, pixels (default: config or 15.0)')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='artifact configuration file')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    # new arguments for forced artifacts
    parser.add_argument('--artifact', type=str, choices=['clean', 'blur', 'noise', 'illumination'],
                       help='force a specific artifact type (default: random mix)')
    parser.add_argument('--strength', type=float, default=None,
                       help='artifact strength (if not specified, random within configured range)')
    
    args = parser.parse_args()
    
    # load artifact config
    artifact_config = {}
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            artifact_config = yaml.safe_load(f)

    # CLI flags override the yaml config, which overrides hardcoded defaults
    gen_config = artifact_config.get('generator', {})
    radius_ratio = args.radius_ratio if args.radius_ratio is not None else gen_config.get('radius_ratio', 0.049)
    radius_ratio_min = args.radius_ratio_min if args.radius_ratio_min is not None else gen_config.get('radius_ratio_min')
    radius_ratio_max = args.radius_ratio_max if args.radius_ratio_max is not None else gen_config.get('radius_ratio_max')
    spacing_ratio = args.spacing_ratio if args.spacing_ratio is not None else gen_config.get('spacing_ratio', 0.48)
    margin = args.margin if args.margin is not None else gen_config.get('margin', 15.0)
    jitter_std = args.jitter if args.jitter is not None else gen_config.get('jitter_std', 0.0)
    thinning_min = gen_config.get('thinning_min', 0.2)
    thinning_max = gen_config.get('thinning_max', 0.4)

    generate_dataset(
        output_dir=args.out,
        num_images=args.num,
        image_size=(args.size, args.size),
        radius_ratio=radius_ratio,
        radius_ratio_min=radius_ratio_min,
        radius_ratio_max=radius_ratio_max,
        spacing_ratio=spacing_ratio,
        jitter_std=jitter_std,
        margin=margin,
        thinning_min=thinning_min,
        thinning_max=thinning_max,
        artifact_config=artifact_config,
        force_artifact=args.artifact,
        force_strength=args.strength,
        seed=args.seed
    )