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
    """render circles onto a blank image"""
    width, height = image_size
    img = np.zeros((height, width), dtype=np.uint8)
    for c in circles:
        cv2.circle(img, (int(c.x), int(c.y)), int(c.r), 255, -1)
    return img


def generate_dataset(output_dir: str, num_images: int = 1000,
                     image_size: tuple = (256, 256),
                     radius_ratio: float = 0.049,
                     spacing_ratio: float = 0.48,
                     jitter_std: float = 2.0,
                     margin: float = 15.0,
                     artifact_config: dict = None,
                     force_artifact: Optional[str] = None,
                     force_strength: Optional[float] = None,
                     seed: int = 42) -> None:
    
    np.random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    width, height = image_size
    circle_radius = width * radius_ratio
    
    config = HexgridConfig(
        image_size=image_size,
        circle_radius=circle_radius,
        margin=margin,
        spacing_ratio=spacing_ratio,
        jitter_std=jitter_std
    )
    
    generator = HexgridGenerator(config)
    pipeline = ArtifactPipeline(artifact_config or {})
    
    print(f"generating {num_images} images")
    print(f"image size: {width}x{height}")
    print(f"circle radius: {circle_radius:.1f} px")
    print(f"grid spacing: {config.spacing:.1f} px")
    print(f"jitter std: {jitter_std:.1f} px")
    
    if force_artifact:
        print(f"forcing artifact: {force_artifact} (strength: {force_strength or 'random'})")
    else:
        print("artifact mode: random mix")
    
    for i in range(num_images):
        # generate circles with random jitter (unique per image)
        circles = generator.generate_grid()
        
        # generate clean image
        clean_image = create_image_from_circles(circles, image_size)
        
        # apply artifacts
        if force_artifact:
            # force a specific artifact type
            degraded_image, modified_circles, artifact_type, artifact_strength = pipeline.apply_forced(
                clean_image, circles, force_artifact, force_strength
            )
        else:
            # random mix
            degraded_image, modified_circles, artifact_type, artifact_strength = pipeline.apply(clean_image, circles)
        
        # determine which circles to save
        if 'missing' in pipeline.applied_artifacts:
            circles_to_save = modified_circles
        else:
            circles_to_save = circles
        
        # save image
        img_name = generate_filename('image', i, 'tif')
        img_path = output_dir / img_name
        save_tif(degraded_image, img_path)
        
        # save ground truth with physics-informed uncertainties
        gt_name = generate_filename('image', i, 'json')
        gt_path = output_dir / gt_name
        save_ground_truth(
            circles_to_save,
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
    parser.add_argument('--jitter', type=float, default=2.0, help='position jitter std (pixels)')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='artifact configuration file')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    # new arguments for forced artifacts
    parser.add_argument('--artifact', type=str, choices=['clean', 'blur', 'noise', 'illumination', 'missing'],
                       help='force a specific artifact type (default: random mix)')
    parser.add_argument('--strength', type=float, default=None,
                       help='artifact strength (if not specified, random within configured range)')
    
    args = parser.parse_args()
    
    # load artifact config
    artifact_config = {}
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            artifact_config = yaml.safe_load(f)
    
    generate_dataset(
        output_dir=args.out,
        num_images=args.num,
        image_size=(args.size, args.size),
        jitter_std=args.jitter,
        artifact_config=artifact_config,
        force_artifact=args.artifact,
        force_strength=args.strength,
        seed=args.seed
    )