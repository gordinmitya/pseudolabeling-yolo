"""Fake teacher model implementations for demonstration purposes."""

import numpy as np
from pathlib import Path
from typing import List, Optional
from PIL import Image

from src.models.base import TeacherModel
from src.data.structures import Annotation, BoundingBox


class RandomDetectorModel(TeacherModel):
    """Generates random detections with varying numbers of bounding boxes.
    
    This model simulates a general-purpose detector that generates 0-5 random
    bounding boxes per image with random class IDs, positions, sizes, and
    confidence scores.
    """
    
    def __init__(self, class_names: List[str], random_seed: Optional[int] = None):
        """Initialize the random detector model.
        
        Args:
            class_names: List of class names for detection
            random_seed: Optional seed for reproducibility
        """
        super().__init__(class_names)
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def predict(self, image_path: Path) -> Annotation:
        """Generate random predictions for an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Annotation with random bounding boxes
        """
        # Load image to get dimensions
        img = Image.open(image_path)
        img_width, img_height = img.size
        img.close()
        
        # Generate random number of boxes (0-5)
        num_boxes = np.random.randint(0, 6)
        
        bounding_boxes = []
        for _ in range(num_boxes):
            # Random class
            class_id = np.random.randint(0, self.num_classes)
            
            # Random position (center point)
            x_center = np.random.uniform(0.1, 0.9)
            y_center = np.random.uniform(0.1, 0.9)
            
            # Random size (ensure it stays within bounds)
            max_width = min(2 * x_center, 2 * (1 - x_center))
            max_height = min(2 * y_center, 2 * (1 - y_center))
            width = np.random.uniform(0.05, min(0.5, max_width))
            height = np.random.uniform(0.05, min(0.5, max_height))
            
            # Random confidence
            confidence = np.random.uniform(0.5, 0.99)
            
            bbox = BoundingBox(
                class_id=class_id,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
                confidence=confidence
            )
            bounding_boxes.append(bbox)
        
        return Annotation(
            image_name=image_path.stem,
            image_path=image_path,
            bounding_boxes=bounding_boxes,
            image_width=img_width,
            image_height=img_height
        )
    
    def get_model_name(self) -> str:
        """Return the model identifier.
        
        Returns:
            Model name string
        """
        return "RandomDetector"


class ConservativeDetectorModel(TeacherModel):
    """Generates conservative detections with fewer, larger boxes.
    
    This model simulates a high-precision detector that generates 0-2 bounding
    boxes per image with higher confidence scores and larger box sizes,
    preferring only 2-3 specific classes.
    """
    
    def __init__(self, class_names: List[str], random_seed: Optional[int] = None):
        """Initialize the conservative detector model.
        
        Args:
            class_names: List of class names for detection
            random_seed: Optional seed for reproducibility
        """
        super().__init__(class_names)
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Prefer only 2-3 specific classes (use first 3 classes)
        self.preferred_classes = list(range(min(3, self.num_classes)))
    
    def predict(self, image_path: Path) -> Annotation:
        """Generate conservative predictions for an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Annotation with conservative bounding boxes
        """
        # Load image to get dimensions
        img = Image.open(image_path)
        img_width, img_height = img.size
        img.close()
        
        # Generate fewer boxes (0-2)
        num_boxes = np.random.randint(0, 3)
        
        bounding_boxes = []
        for _ in range(num_boxes):
            # Prefer specific classes
            class_id = np.random.choice(self.preferred_classes)
            
            # Random position (center point)
            x_center = np.random.uniform(0.2, 0.8)
            y_center = np.random.uniform(0.2, 0.8)
            
            # Larger boxes (min size: 0.1)
            max_width = min(2 * x_center, 2 * (1 - x_center))
            max_height = min(2 * y_center, 2 * (1 - y_center))
            width = np.random.uniform(0.1, min(0.6, max_width))
            height = np.random.uniform(0.1, min(0.6, max_height))
            
            # Higher confidence
            confidence = np.random.uniform(0.7, 0.99)
            
            bbox = BoundingBox(
                class_id=class_id,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
                confidence=confidence
            )
            bounding_boxes.append(bbox)
        
        return Annotation(
            image_name=image_path.stem,
            image_path=image_path,
            bounding_boxes=bounding_boxes,
            image_width=img_width,
            image_height=img_height
        )
    
    def get_model_name(self) -> str:
        """Return the model identifier.
        
        Returns:
            Model name string
        """
        return "ConservativeDetector"


class AggressiveDetectorModel(TeacherModel):
    """Generates aggressive detections with many, smaller boxes.
    
    This model simulates a high-recall detector that generates 3-10 bounding
    boxes per image with lower confidence scores and smaller box sizes,
    detecting across all available classes.
    """
    
    def __init__(self, class_names: List[str], random_seed: Optional[int] = None):
        """Initialize the aggressive detector model.
        
        Args:
            class_names: List of class names for detection
            random_seed: Optional seed for reproducibility
        """
        super().__init__(class_names)
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def predict(self, image_path: Path) -> Annotation:
        """Generate aggressive predictions for an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Annotation with aggressive bounding boxes
        """
        # Load image to get dimensions
        img = Image.open(image_path)
        img_width, img_height = img.size
        img.close()
        
        # Generate more boxes (3-10)
        num_boxes = np.random.randint(3, 11)
        
        bounding_boxes = []
        for _ in range(num_boxes):
            # All classes available
            class_id = np.random.randint(0, self.num_classes)
            
            # Random position (center point)
            x_center = np.random.uniform(0.1, 0.9)
            y_center = np.random.uniform(0.1, 0.9)
            
            # Smaller boxes (min size: 0.02)
            max_width = min(2 * x_center, 2 * (1 - x_center))
            max_height = min(2 * y_center, 2 * (1 - y_center))
            width = np.random.uniform(0.02, min(0.4, max_width))
            height = np.random.uniform(0.02, min(0.4, max_height))
            
            # Lower confidence
            confidence = np.random.uniform(0.3, 0.9)
            
            bbox = BoundingBox(
                class_id=class_id,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
                confidence=confidence
            )
            bounding_boxes.append(bbox)
        
        return Annotation(
            image_name=image_path.stem,
            image_path=image_path,
            bounding_boxes=bounding_boxes,
            image_width=img_width,
            image_height=img_height
        )
    
    def get_model_name(self) -> str:
        """Return the model identifier.
        
        Returns:
            Model name string
        """
        return "AggressiveDetector"
