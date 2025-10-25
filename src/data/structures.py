"""Data structures for bounding boxes and annotations."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class BoundingBox:
    """Represents a bounding box in YOLO format.
    
    Attributes:
        class_id: Zero-indexed class identifier
        x_center: Normalized x-coordinate of box center (0-1)
        y_center: Normalized y-coordinate of box center (0-1)
        width: Normalized width of box (0-1)
        height: Normalized height of box (0-1)
        confidence: Optional confidence score (0-1)
    """
    
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: Optional[float] = None
    
    def __post_init__(self) -> None:
        """Validate bounding box values after initialization."""
        if not 0 <= self.x_center <= 1:
            raise ValueError(f"x_center must be between 0 and 1, got {self.x_center}")
        if not 0 <= self.y_center <= 1:
            raise ValueError(f"y_center must be between 0 and 1, got {self.y_center}")
        if not 0 <= self.width <= 1:
            raise ValueError(f"width must be between 0 and 1, got {self.width}")
        if not 0 <= self.height <= 1:
            raise ValueError(f"height must be between 0 and 1, got {self.height}")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be between 0 and 1, got {self.confidence}")
        if self.class_id < 0:
            raise ValueError(f"class_id must be non-negative, got {self.class_id}")
    
    def to_yolo_format(self) -> str:
        """Convert bounding box to YOLO format string.
        
        Returns:
            String in format "class_id x_center y_center width height"
        """
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"
    
    @classmethod
    def from_pixels(
        cls,
        class_id: int,
        x_center_px: float,
        y_center_px: float,
        width_px: float,
        height_px: float,
        img_width: int,
        img_height: int,
        confidence: Optional[float] = None
    ) -> "BoundingBox":
        """Create BoundingBox from pixel coordinates.
        
        Args:
            class_id: Zero-indexed class identifier
            x_center_px: X-coordinate of box center in pixels
            y_center_px: Y-coordinate of box center in pixels
            width_px: Width of box in pixels
            height_px: Height of box in pixels
            img_width: Image width in pixels
            img_height: Image height in pixels
            confidence: Optional confidence score
            
        Returns:
            BoundingBox instance with normalized coordinates
        """
        return cls(
            class_id=class_id,
            x_center=x_center_px / img_width,
            y_center=y_center_px / img_height,
            width=width_px / img_width,
            height=height_px / img_height,
            confidence=confidence
        )


@dataclass
class Annotation:
    """Represents annotations for a single image.
    
    Attributes:
        image_name: Filename without extension
        image_path: Path to the image file
        bounding_boxes: List of bounding boxes in the image
        image_width: Image width in pixels
        image_height: Image height in pixels
    """
    
    image_name: str
    image_path: Path
    bounding_boxes: List[BoundingBox]
    image_width: int
    image_height: int
    
    def to_yolo_lines(self) -> List[str]:
        """Convert all bounding boxes to YOLO format lines.
        
        Returns:
            List of YOLO format strings, one per bounding box
        """
        return [bbox.to_yolo_format() for bbox in self.bounding_boxes]
    
    def has_objects(self) -> bool:
        """Check if annotation contains any bounding boxes.
        
        Returns:
            True if there are any bounding boxes, False otherwise
        """
        return len(self.bounding_boxes) > 0
