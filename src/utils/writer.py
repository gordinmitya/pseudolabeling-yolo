"""Utilities for writing YOLO format annotations."""

import shutil
import random
from pathlib import Path
from typing import Dict, List

import yaml

from src.data.structures import Annotation


def write_yolo_annotation(annotation: Annotation, output_label_path: Path) -> None:
    """Write annotation to a YOLO format .txt file.
    
    If the annotation has no objects, no file is created (YOLO convention).
    
    Args:
        annotation: Annotation object to write
        output_label_path: Path where the .txt file should be written
    """
    # YOLO convention: no file if no objects
    if not annotation.has_objects():
        return
    
    # Ensure parent directory exists
    output_label_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write YOLO format lines
    lines = annotation.to_yolo_lines()
    with open(output_label_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def create_dataset_yaml(
    output_dir: Path,
    class_names: List[str],
    split_ratio: float = 0.8
) -> None:
    """Create dataset.yaml file for YOLO training.
    
    Args:
        output_dir: Output directory where dataset.yaml will be created
        class_names: List of class names in order (index = class_id)
        split_ratio: Train/val split ratio (not used in yaml but for documentation)
    """
    # Create the yaml content
    yaml_content = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    # Write yaml file
    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)


def copy_images_to_split(
    image_paths: List[Path],
    output_dir: Path,
    split_ratio: float = 0.8,
    random_seed: int = None
) -> Dict[str, List[Path]]:
    """Copy images to train/val directories based on split ratio.
    
    Args:
        image_paths: List of source image paths
        output_dir: Output directory containing images/train and images/val
        split_ratio: Ratio of images for training (0-1)
        random_seed: Optional seed for reproducible splitting
        
    Returns:
        Dictionary with 'train' and 'val' keys containing lists of copied paths
    """
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
    
    # Shuffle images
    shuffled_paths = image_paths.copy()
    random.shuffle(shuffled_paths)
    
    # Calculate split point
    split_idx = int(len(shuffled_paths) * split_ratio)
    train_paths = shuffled_paths[:split_idx]
    val_paths = shuffled_paths[split_idx:]
    
    # Create directories
    train_img_dir = output_dir / 'images' / 'train'
    val_img_dir = output_dir / 'images' / 'val'
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy train images
    train_copied = []
    for img_path in train_paths:
        dest_path = train_img_dir / img_path.name
        shutil.copy2(img_path, dest_path)
        train_copied.append(dest_path)
    
    # Copy val images
    val_copied = []
    for img_path in val_paths:
        dest_path = val_img_dir / img_path.name
        shutil.copy2(img_path, dest_path)
        val_copied.append(dest_path)
    
    return {
        'train': train_copied,
        'val': val_copied
    }
