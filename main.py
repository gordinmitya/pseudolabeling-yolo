"""Main entry point for pseudo-labeling system."""

import argparse
import logging
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import numpy as np
from PIL import Image

from src.models.base import TeacherModel
from src.models.fake_models import (
    RandomDetectorModel,
    ConservativeDetectorModel,
    AggressiveDetectorModel
)
from src.data.structures import Annotation, BoundingBox
from src.utils.writer import (
    write_yolo_annotation,
    create_dataset_yaml,
    copy_images_to_split
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


# Default class names for fake models
DEFAULT_CLASS_NAMES = ["person", "car", "dog", "cat", "bicycle", "truck"]


def create_dummy_images(output_dir: Path, num_images: int) -> List[Path]:
    """Create dummy images for testing.
    
    Args:
        output_dir: Directory to save dummy images
        num_images: Number of dummy images to create
        
    Returns:
        List of paths to created dummy images
    """
    dummy_dir = output_dir / "dummy_images"
    dummy_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = []
    for i in range(num_images):
        # Create a simple colored image
        img = Image.new('RGB', (640, 480), color=(
            np.random.randint(100, 200),
            np.random.randint(100, 200),
            np.random.randint(100, 200)
        ))
        
        img_path = dummy_dir / f"img_{i+1:03d}.jpg"
        img.save(img_path)
        image_paths.append(img_path)
        img.close()
    
    logger.info(f"Created {num_images} dummy images in {dummy_dir}")
    return image_paths


def get_image_files(input_dir: Path) -> List[Path]:
    """Get all image files from input directory.
    
    Args:
        input_dir: Input directory to scan
        
    Returns:
        List of image file paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_paths = []
    
    if not input_dir.exists():
        logger.warning(f"Input directory {input_dir} does not exist")
        return image_paths
    
    for ext in image_extensions:
        image_paths.extend(input_dir.glob(f'*{ext}'))
    
    return sorted(image_paths)


def initialize_model(
    model_type: str,
    class_names: List[str],
    random_seed: int = None
) -> TeacherModel:
    """Initialize a teacher model based on type.
    
    Args:
        model_type: Type of model ('random', 'conservative', 'aggressive')
        class_names: List of class names
        random_seed: Optional random seed for reproducibility
        
    Returns:
        Initialized teacher model
    """
    if model_type == 'random':
        return RandomDetectorModel(class_names, random_seed)
    elif model_type == 'conservative':
        return ConservativeDetectorModel(class_names, random_seed)
    elif model_type == 'aggressive':
        return AggressiveDetectorModel(class_names, random_seed)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def ensemble_predictions(
    annotations: List[Annotation],
    class_names: List[str]
) -> Annotation:
    """Combine predictions from multiple models using ensemble.
    
    For simplicity, this combines all bounding boxes from all models.
    In production, you might use Non-Maximum Suppression (NMS) or voting.
    
    Args:
        annotations: List of annotations from different models
        class_names: List of class names
        
    Returns:
        Combined annotation
    """
    if not annotations:
        raise ValueError("No annotations provided for ensemble")
    
    # Use the first annotation as base
    base = annotations[0]
    
    # Combine all bounding boxes
    all_boxes = []
    for ann in annotations:
        all_boxes.extend(ann.bounding_boxes)
    
    return Annotation(
        image_name=base.image_name,
        image_path=base.image_path,
        bounding_boxes=all_boxes,
        image_width=base.image_width,
        image_height=base.image_height
    )


def process_images(
    image_paths: List[Path],
    models: List[TeacherModel],
    use_ensemble: bool = False
) -> List[Annotation]:
    """Process images with teacher models.
    
    Args:
        image_paths: List of image paths to process
        models: List of teacher models to use
        use_ensemble: Whether to combine predictions from all models
        
    Returns:
        List of annotations
    """
    annotations = []
    
    for idx, img_path in enumerate(image_paths, 1):
        logger.info(f"Processing image {idx}/{len(image_paths)}: {img_path.name}")
        
        if use_ensemble:
            # Get predictions from all models
            model_annotations = []
            for model in models:
                ann = model.predict(img_path)
                model_annotations.append(ann)
            
            # Combine predictions
            annotation = ensemble_predictions(model_annotations, models[0].class_names)
        else:
            # Use only the first model
            annotation = models[0].predict(img_path)
        
        annotations.append(annotation)
    
    return annotations


def save_annotations(
    annotations: List[Annotation],
    split_images: Dict[str, List[Path]],
    output_dir: Path
) -> None:
    """Save annotations to YOLO format files.
    
    Args:
        annotations: List of annotations
        split_images: Dictionary with 'train' and 'val' image paths
        output_dir: Output directory
    """
    # Create a mapping from image name to annotation
    ann_map = {ann.image_name: ann for ann in annotations}
    
    # Create label directories
    train_label_dir = output_dir / 'labels' / 'train'
    val_label_dir = output_dir / 'labels' / 'val'
    train_label_dir.mkdir(parents=True, exist_ok=True)
    val_label_dir.mkdir(parents=True, exist_ok=True)
    
    # Write train annotations
    for img_path in split_images['train']:
        img_name = img_path.stem
        if img_name in ann_map:
            label_path = train_label_dir / f"{img_name}.txt"
            write_yolo_annotation(ann_map[img_name], label_path)
    
    # Write val annotations
    for img_path in split_images['val']:
        img_name = img_path.stem
        if img_name in ann_map:
            label_path = val_label_dir / f"{img_name}.txt"
            write_yolo_annotation(ann_map[img_name], label_path)


def calculate_statistics(
    annotations: List[Annotation],
    class_names: List[str]
) -> Dict:
    """Calculate statistics about the annotations.
    
    Args:
        annotations: List of annotations
        class_names: List of class names
        
    Returns:
        Dictionary with statistics
    """
    total_images = len(annotations)
    images_with_detections = sum(1 for ann in annotations if ann.has_objects())
    images_without_detections = total_images - images_with_detections
    
    total_boxes = sum(len(ann.bounding_boxes) for ann in annotations)
    avg_boxes = total_boxes / total_images if total_images > 0 else 0
    
    # Count boxes per class
    class_counts = defaultdict(int)
    for ann in annotations:
        for bbox in ann.bounding_boxes:
            class_counts[bbox.class_id] += 1
    
    return {
        'total_images': total_images,
        'images_with_detections': images_with_detections,
        'images_without_detections': images_without_detections,
        'total_boxes': total_boxes,
        'avg_boxes': avg_boxes,
        'class_counts': class_counts
    }


def print_summary(
    stats: Dict,
    split_images: Dict[str, List[Path]],
    class_names: List[str],
    output_dir: Path
) -> None:
    """Print summary statistics.
    
    Args:
        stats: Statistics dictionary
        split_images: Dictionary with train/val splits
        class_names: List of class names
        output_dir: Output directory
    """
    logger.info("\n===== Summary =====")
    logger.info(f"Total images: {stats['total_images']}")
    logger.info(f"Train images: {len(split_images['train'])}")
    logger.info(f"Val images: {len(split_images['val'])}")
    logger.info(f"Images with detections: {stats['images_with_detections']}")
    logger.info(f"Images without detections: {stats['images_without_detections']}")
    logger.info(f"Average boxes per image: {stats['avg_boxes']:.1f}")
    
    if stats['class_counts']:
        logger.info("Class distribution:")
        for class_id, count in sorted(stats['class_counts'].items()):
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            logger.info(f"  - {class_name}: {count}")
    
    logger.info(f"\nOutput saved to: {output_dir}")
    logger.info("Ready for YOLOv11 training!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Pseudo-labeling system for YOLOv11'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('input/images'),
        help='Input images directory (default: input/images)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('output'),
        help='Output directory (default: output)'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['random', 'conservative', 'aggressive', 'ensemble'],
        default='random',
        help='Model type to use (default: random)'
    )
    parser.add_argument(
        '--split-ratio',
        type=float,
        default=0.8,
        help='Train/val split ratio (default: 0.8)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--num-dummy-images',
        type=int,
        default=10,
        help='Number of dummy images to create if no real images found (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # Create output directory structure
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get image files
    image_paths = get_image_files(args.input_dir)
    
    # Create dummy images if no real images found
    if not image_paths:
        logger.info(f"No images found in {args.input_dir}")
        logger.info(f"Creating {args.num_dummy_images} dummy images...")
        image_paths = create_dummy_images(args.output_dir, args.num_dummy_images)
    
    logger.info(f"Found {len(image_paths)} images")
    
    # Initialize models
    use_ensemble = args.model == 'ensemble'
    models = []
    
    if use_ensemble:
        logger.info("Initializing 3 teacher models for ensemble")
        models = [
            RandomDetectorModel(DEFAULT_CLASS_NAMES, args.seed),
            ConservativeDetectorModel(DEFAULT_CLASS_NAMES, args.seed),
            AggressiveDetectorModel(DEFAULT_CLASS_NAMES, args.seed)
        ]
    else:
        logger.info(f"Initializing {args.model} model")
        models = [initialize_model(args.model, DEFAULT_CLASS_NAMES, args.seed)]
    
    # Process images
    logger.info("Processing images...")
    annotations = process_images(image_paths, models, use_ensemble)
    
    # Split dataset
    logger.info(f"Splitting dataset ({args.split_ratio:.0%}/{1-args.split_ratio:.0%} train/val split)")
    split_images = copy_images_to_split(
        image_paths,
        args.output_dir,
        args.split_ratio,
        args.seed
    )
    
    # Save annotations
    logger.info("Writing YOLO annotations...")
    save_annotations(annotations, split_images, args.output_dir)
    
    # Create dataset.yaml
    logger.info("Creating dataset.yaml...")
    create_dataset_yaml(args.output_dir, DEFAULT_CLASS_NAMES, args.split_ratio)
    
    # Calculate and print statistics
    stats = calculate_statistics(annotations, DEFAULT_CLASS_NAMES)
    print_summary(stats, split_images, DEFAULT_CLASS_NAMES, args.output_dir)


if __name__ == '__main__':
    main()
