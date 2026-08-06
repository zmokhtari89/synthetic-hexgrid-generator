"""
image artifacts for synthetic hexgrid generator
"""

import numpy as np
from scipy.ndimage import gaussian_filter, convolve
from typing import Tuple, Optional, List
from generator.core import Circle


def add_gaussian_noise(image: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """add gaussian noise to image"""
    noise = np.random.normal(0, sigma, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper(image: np.ndarray, prob: float = 0.01) -> np.ndarray:
    """add salt & pepper noise"""
    noisy = image.copy()
    salt = np.random.random(image.shape) < prob / 2
    pepper = np.random.random(image.shape) < prob / 2
    noisy[salt] = 255
    noisy[pepper] = 0
    return noisy


def add_gaussian_blur(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """apply gaussian blur"""
    return gaussian_filter(image.astype(np.float32), sigma=sigma).astype(np.uint8)


def add_motion_blur(image: np.ndarray, size: int = 5, angle: float = 0.0) -> np.ndarray:
    """apply motion blur along specified angle"""
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    
    if size <= 1:
        return image
    
    angle_rad = np.radians(angle)
    dx = int(size * np.cos(angle_rad))
    dy = int(size * np.sin(angle_rad))
    
    if dx == 0 and dy == 0:
        dx = 1
    
    max_step = max(abs(dx), abs(dy))
    for i in range(size):
        t = (i - center) / max_step if max_step > 0 else 0
        x = center + dx * t
        y = center + dy * t
        ix = int(round(x))
        iy = int(round(y))
        if 0 <= ix < size and 0 <= iy < size:
            kernel[iy, ix] = 1.0
    
    if kernel.sum() > 0:
        kernel = kernel / kernel.sum()
    
    return convolve(image.astype(np.float32), kernel).astype(np.uint8)


def add_uneven_illumination(image: np.ndarray, gradient_x: float = 0.0,
                            gradient_y: float = 0.0) -> np.ndarray:
    """apply non-uniform illumination gradient"""
    h, w = image.shape
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x, y)
    
    illum = 1.0 - gradient_x * X - gradient_y * Y
    illum = np.clip(illum, 0.3, 1.0)
    
    result = image.astype(np.float32) * illum
    return np.clip(result, 0, 255).astype(np.uint8)


def add_circle_jitter(circles: List[Circle], max_shift: float = 2.0,
                      keep_prob: float = 1.0) -> List[Circle]:
    """
    apply structural variations to circles (jitter and missing)
    this modifies the circle list directly, not the image
    """
    result = []
    for c in circles:
        if np.random.random() < keep_prob:
            dx = np.random.uniform(-max_shift, max_shift)
            dy = np.random.uniform(-max_shift, max_shift)
            result.append(Circle(x=c.x + dx, y=c.y + dy, r=c.r))
    return result


class ArtifactPipeline:
    """apply random artifacts with configurable probabilities and strengths"""
    
    def __init__(self, config: dict):
        self.config = config
        self.applied_artifacts = []
        self.artifact_strengths = {}
    
    def _get_artifact_strength(self, artifact_type: str) -> float:
        """return the strength of a specific artifact"""
        return self.artifact_strengths.get(artifact_type, 0.0)
    
    def apply(self, image: np.ndarray, circles: List[Circle]) -> Tuple[np.ndarray, List[Circle], str, float]:
        """
        apply configured pixel-level artifacts to image
        returns: (degraded_image, modified_circles, artifact_type, artifact_strength)

        artifact_type: 'clean', 'blur', 'noise', 'illumination'
        artifact_strength: the strength value used for uncertainty scaling

        does not handle 'missing' circles — that's a structural artifact
        decided before rendering, in generate_dataset() (main.py).
        """
        self.applied_artifacts = []
        self.artifact_strengths = {}
        
        modified_circles = circles
        final_artifact_type = 'clean'
        final_artifact_strength = 0.0
        
        # 1. noise
        if np.random.random() < self.config.get('noise_prob', 0.0):
            noise_type = np.random.choice(['gaussian', 'salt_pepper'])
            if noise_type == 'gaussian':
                sigma = np.random.uniform(1, self.config.get('noise_max_sigma', 10))
                image = add_gaussian_noise(image, sigma)
                self.artifact_strengths['noise'] = sigma
                final_artifact_type = 'noise'
                final_artifact_strength = sigma
            else:
                prob = np.random.uniform(0.005, self.config.get('salt_pepper_max_prob', 0.03))
                image = add_salt_pepper(image, prob)
                self.artifact_strengths['noise'] = prob * 100
                final_artifact_type = 'noise'
                final_artifact_strength = prob * 100
            self.applied_artifacts.append(noise_type)
        
        # 2. blur
        if np.random.random() < self.config.get('blur_prob', 0.0):
            blur_type = np.random.choice(['gaussian', 'motion'])
            if blur_type == 'gaussian':
                sigma = np.random.uniform(0.5, self.config.get('blur_max_sigma', 3))
                image = add_gaussian_blur(image, sigma)
                self.artifact_strengths['blur'] = sigma
                final_artifact_type = 'blur'
                final_artifact_strength = sigma
            else:
                max_size = self.config.get('motion_blur_max_size', 9)
                size = np.random.randint(3, max_size + 1, 1)[0]
                if size % 2 == 0:
                    size += 1
                angle = np.random.uniform(0, 180)
                image = add_motion_blur(image, size, angle)
                self.artifact_strengths['blur'] = size
                final_artifact_type = 'blur'
                final_artifact_strength = size
            self.applied_artifacts.append(blur_type)
        
        # 3. uneven illumination
        if np.random.random() < self.config.get('illumination_prob', 0.0):
            grad_x = np.random.uniform(-0.3, 0.3)
            grad_y = np.random.uniform(-0.3, 0.3)
            image = add_uneven_illumination(image, grad_x, grad_y)
            strength = max(abs(grad_x), abs(grad_y))
            self.artifact_strengths['illumination'] = strength
            final_artifact_type = 'illumination'
            final_artifact_strength = strength
            self.applied_artifacts.append('illumination')

        # note: "missing circles" is a structural (pre-render) artifact, not a
        # pixel-level one — it must be decided before the image is rendered so
        # the image and ground truth share one already-thinned circle list.
        # See generate_dataset() in main.py.

        # IMPORTANT: always return something, even if no artifacts applied
        return image, modified_circles, final_artifact_type, final_artifact_strength
            
    def apply_forced(self, image: np.ndarray, circles: List[Circle],
                     artifact_type: str, strength: Optional[float] = None) -> Tuple[np.ndarray, List[Circle], str, float]:
        """
        apply a specific artifact type with optional strength
        returns: (degraded_image, modified_circles, artifact_type, artifact_strength)
        """
        self.applied_artifacts = []
        self.artifact_strengths = {}
        
        modified_circles = circles
        final_artifact_type = 'clean'
        final_artifact_strength = 0.0
        
        if artifact_type == 'clean':
            return image, circles, 'clean', 0.0
        
        elif artifact_type == 'blur':
            if strength is None:
                # no controlled strength requested: sub-type and strength are
                # drawn together, so the number always matches its sub-type
                blur_type = np.random.choice(['gaussian', 'motion'])
                if blur_type == 'gaussian':
                    sigma = np.random.uniform(0.5, self.config.get('blur_max_sigma', 3))
                    image = add_gaussian_blur(image, sigma)
                    final_artifact_strength = sigma
                else:
                    max_size = self.config.get('motion_blur_max_size', 9)
                    size = np.random.randint(3, max_size + 1, 1)[0]
                    if size % 2 == 0:
                        size += 1
                    angle = np.random.uniform(0, 180)
                    image = add_motion_blur(image, size, angle)
                    final_artifact_strength = size
                final_artifact_type = 'blur'
                self.applied_artifacts.append(blur_type)
            else:
                # a specific strength was requested (e.g. building a
                # controlled-degradation dataset variant): pin the sub-type to
                # gaussian so `strength` always means the same thing (a sigma),
                # matching Zenodo's documented blur method and this repo's
                # sigma-based uncertainty model. Randomizing gaussian/motion
                # here would silently give the same number two different
                # physical meanings within one dataset variant.
                sigma = strength
                image = add_gaussian_blur(image, sigma)
                final_artifact_type = 'blur'
                final_artifact_strength = sigma
                self.applied_artifacts.append('gaussian')

        elif artifact_type == 'noise':
            if strength is None:
                noise_type = np.random.choice(['gaussian', 'salt_pepper'])
                if noise_type == 'gaussian':
                    sigma = np.random.uniform(1, self.config.get('noise_max_sigma', 10))
                    image = add_gaussian_noise(image, sigma)
                    final_artifact_strength = sigma
                else:
                    prob = np.random.uniform(0.005, self.config.get('salt_pepper_max_prob', 0.03))
                    image = add_salt_pepper(image, prob)
                    final_artifact_strength = prob * 100
                final_artifact_type = 'noise'
                self.applied_artifacts.append(noise_type)
            else:
                # same reasoning as blur above: pin to gaussian so `strength`
                # consistently means a sigma, not sometimes a sigma and
                # sometimes a salt-and-pepper probability*100.
                sigma = strength
                image = add_gaussian_noise(image, sigma)
                final_artifact_type = 'noise'
                final_artifact_strength = sigma
                self.applied_artifacts.append('gaussian')


        elif artifact_type == 'illumination':
            if strength is None:
                grad_x = np.random.uniform(-0.3, 0.3)
                grad_y = np.random.uniform(-0.3, 0.3)
            else:
                grad_x = strength
                grad_y = strength * np.random.uniform(-1, 1)
            image = add_uneven_illumination(image, grad_x, grad_y)
            final_strength = max(abs(grad_x), abs(grad_y))
            final_artifact_type = 'illumination'
            final_artifact_strength = final_strength
            self.applied_artifacts.append('illumination')

        # note: 'missing' is not handled here — it's a structural (pre-render)
        # artifact decided before the image exists, in generate_dataset()
        # (main.py). apply_forced() only ever receives pixel-level types.

        return image, modified_circles, final_artifact_type, final_artifact_strength
        
        # 2. blur
        if np.random.random() < self.config.get('blur_prob', 0.0):
            blur_type = np.random.choice(['gaussian', 'motion'])
            if blur_type == 'gaussian':
                sigma = np.random.uniform(0.5, self.config.get('blur_max_sigma', 3))
                image = add_gaussian_blur(image, sigma)
                self.artifact_strengths['blur'] = sigma
                final_artifact_type = 'blur'
                final_artifact_strength = sigma
            else:
                max_size = self.config.get('motion_blur_max_size', 9)
                size = np.random.randint(3, max_size + 1, 1)[0]
                if size % 2 == 0:
                    size += 1
                angle = np.random.uniform(0, 180)
                image = add_motion_blur(image, size, angle)
                self.artifact_strengths['blur'] = size
                final_artifact_type = 'blur'
                final_artifact_strength = size
            self.applied_artifacts.append(blur_type)
        
        # 3. uneven illumination
        if np.random.random() < self.config.get('illumination_prob', 0.0):
            grad_x = np.random.uniform(-0.3, 0.3)
            grad_y = np.random.uniform(-0.3, 0.3)
            image = add_uneven_illumination(image, grad_x, grad_y)
            strength = max(abs(grad_x), abs(grad_y))
            self.artifact_strengths['illumination'] = strength
            final_artifact_type = 'illumination'
            final_artifact_strength = strength
            self.applied_artifacts.append('illumination')
        
        # 4. structural variations (missing circles)
        if np.random.random() < self.config.get('missing_prob', 0.0):
            keep_prob = np.random.uniform(0.7, 0.95)
            modified_circles = add_circle_jitter(circles, 0, keep_prob)
            strength = 1 - keep_prob  # fraction missing
            self.artifact_strengths['missing'] = strength
            final_artifact_type = 'missing'
            final_artifact_strength = strength
            self.applied_artifacts.append('missing')
        
        return image, modified_circles, final_artifact_type, final_artifact_strength