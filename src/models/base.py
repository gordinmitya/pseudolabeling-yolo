"""Abstract base class for teacher models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from src.data.structures import Annotation


class TeacherModel(ABC):
    """Abstract base class for teacher models.
    
    All teacher models must inherit from this class and implement
    the required abstract methods.
    
    Attributes:
        class_names: List of class names the model can detect
        num_classes: Number of classes the model can detect
    """
    
    def __init__(self, class_names: List[str]):
        """Initialize the teacher model.
        
        Args:
            class_names: List of class names for detection
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    @abstractmethod
    def predict(self, image_path: Path) -> Annotation:
        """Generate predictions for a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Annotation object containing predicted bounding boxes
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier.
        
        Returns:
            String identifier for this model
        """
        pass
