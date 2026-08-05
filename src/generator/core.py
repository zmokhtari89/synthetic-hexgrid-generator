"""
hexagonal grid circle generator
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Circle:
    x: float
    y: float
    r: float


@dataclass
class HexgridConfig:
    image_size: Tuple[int, int]  # (width, height)
    circle_radius: float  # radius in pixels
    margin: float = 10.0
    spacing_ratio: float = 0.48  # must be < 0.5 for no overlap
    jitter_std: float = 0.0  # standard deviation of random position jitter (pixels)
    
    @property
    def spacing(self) -> float:
        """center-to-center distance between adjacent circles"""
        return self.circle_radius / self.spacing_ratio
    
    @property
    def max_radius(self) -> float:
        """theoretical max radius without overlap"""
        return self.spacing / 2.0


class HexgridGenerator:
    """generates non-overlapping circles on a hexagonal lattice"""
    
    def __init__(self, config: HexgridConfig):
        self.config = config
        if config.spacing_ratio >= 0.5:
            raise ValueError("spacing_ratio must be < 0.5 to prevent overlap")
        self._basis_vectors = self._compute_basis()
    
    def _compute_basis(self) -> Tuple[np.ndarray, np.ndarray]:
        s = self.config.spacing
        u = np.array([s, 0.0])
        v = np.array([s * 0.5, s * np.sqrt(3.0) / 2.0])
        return u, v
    
    def _circles_overlap(self, c1: Circle, c2: Circle) -> bool:
        dx = c1.x - c2.x
        dy = c1.y - c2.y
        distance = np.sqrt(dx*dx + dy*dy)
        return distance < (c1.r + c2.r)
    
    def generate_grid(self, jitter: Optional[float] = None) -> List[Circle]:
        """
        generate circles with optional random position jitter
        
        jitter: if None, use self.config.jitter_std
                if provided, use this value instead
        """
        width, height = self.config.image_size
        margin = self.config.margin
        s = self.config.spacing
        radius = self.config.circle_radius
        jitter_std = jitter if jitter is not None else self.config.jitter_std
        
        u, v = self._basis_vectors
        
        nx = int((width - 2 * margin) / s) + 3
        ny = int((height - 2 * margin) / (s * np.sqrt(3) / 2)) + 3
        
        candidates = []
        
        for i in range(-1, nx + 1):
            for j in range(-1, ny + 1):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                
                if margin <= x <= width - margin and margin <= y <= height - margin:
                    candidates.append(Circle(x=x, y=y, r=radius))
        
        # apply jitter and check overlaps
        circles = []
        if jitter_std > 0:
            # shuffle to avoid bias in overlap removal
            np.random.shuffle(candidates)
        
        for c in candidates:
            if jitter_std > 0:
                # add random shift
                dx = np.random.normal(0, jitter_std)
                dy = np.random.normal(0, jitter_std)
                shifted = Circle(x=c.x + dx, y=c.y + dy, r=c.r)
            else:
                shifted = c
            
            # check if still inside image
            if not (margin <= shifted.x <= width - margin and 
                   margin <= shifted.y <= height - margin):
                continue
            
            # check overlap with existing circles
            overlap = False
            for existing in circles:
                if self._circles_overlap(shifted, existing):
                    overlap = True
                    break
            if not overlap:
                circles.append(shifted)
        
        return circles
    
    def random_subset(self, circles: List[Circle], keep_prob: float = 1.0) -> List[Circle]:
        if keep_prob >= 1.0:
            return circles
        
        mask = np.random.random(len(circles)) < keep_prob
        return [c for c, keep in zip(circles, mask) if keep]