"""
quick test for hexgrid generator
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import cv2
import tifffile
import numpy as np
from generator.core import HexgridGenerator, HexgridConfig


def test_hexgrid():
    width, height = 256, 256
    circle_radius = width * 0.049  # match zenodo density
    
    config = HexgridConfig(
        image_size=(width, height),
        circle_radius=circle_radius,
        margin=15.0
    )
    
    generator = HexgridGenerator(config)
    circles = generator.generate_grid()
    
    print(f"generated {len(circles)} circles")
    print(f"radius: {circle_radius:.1f} px")
    print(f"spacing: {config.spacing:.1f} px")
    
    # render image
    img = np.zeros((height, width), dtype=np.uint8)
    for c in circles:
        cv2.circle(img, (int(c.x), int(c.y)), int(c.r), 255, -1)
    
    # save as tif
    tifffile.imwrite('test_hexgrid.tif', img, photometric='minisblack')
    print("saved test_hexgrid.tif")
    
    # print first 5 circles
    print("\nfirst 5 circles:")
    for i, c in enumerate(circles[:5]):
        print(f"  {i+1}: x={c.x:.1f}, y={c.y:.1f}, r={c.r:.1f}")


if __name__ == "__main__":
    test_hexgrid()