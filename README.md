# Pseudo-Labeling System for YOLOv11

A production-ready Python project for pseudo-labeling images using multiple teacher models (detection models), with outputs formatted for YOLOv11 fine-tuning.

## Overview

This project provides a framework for generating pseudo-labels (synthetic annotations) for unlabeled images using teacher models. The generated annotations are in YOLO format, ready for training YOLOv11 or other YOLO-based object detection models.

**Key Features:**
- Abstract base class for easy integration of custom teacher models
- Three built-in fake detector models for demonstration
- Ensemble prediction support
- Automatic train/val dataset splitting
- YOLO format annotation export
- Comprehensive statistics and logging

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd pseudolabeling-yolo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Directory Structure

```
pseudo_labeling/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract TeacherModel base class
│   │   └── fake_models.py   # Fake teacher model implementations
│   ├── data/
│   │   ├── __init__.py
│   │   └── structures.py    # BoundingBox, Annotation classes
│   └── utils/
│       ├── __init__.py
│       └── writer.py        # YOLO format writer
├── input/
│   └── images/              # Place input images here
├── output/
│   ├── images/
│   │   ├── train/           # Training images (auto-generated)
│   │   └── val/             # Validation images (auto-generated)
│   ├── labels/
│   │   ├── train/           # Training labels (auto-generated)
│   │   └── val/             # Validation labels (auto-generated)
│   └── dataset.yaml         # YOLO dataset configuration (auto-generated)
├── main.py                  # Entry point script
├── requirements.txt
└── README.md
```

## Usage

### Basic Usage with Single Model

Process images with a single detector model:

```bash
# Using random detector
python main.py --model random

# Using conservative detector (fewer, larger boxes)
python main.py --model conservative

# Using aggressive detector (more, smaller boxes)
python main.py --model aggressive
```

### Ensemble Usage

Combine predictions from multiple models:

```bash
python main.py --model ensemble
```

### Custom Options

```bash
python main.py \
    --input-dir path/to/images \
    --output-dir path/to/output \
    --model random \
    --split-ratio 0.8 \
    --seed 42 \
    --num-dummy-images 20
```

**Arguments:**
- `--input-dir`: Directory containing input images (default: `input/images`)
- `--output-dir`: Output directory for processed data (default: `output`)
- `--model`: Model type - `random`, `conservative`, `aggressive`, or `ensemble` (default: `random`)
- `--split-ratio`: Train/validation split ratio (default: 0.8)
- `--seed`: Random seed for reproducibility (optional)
- `--num-dummy-images`: Number of dummy images to create if no real images found (default: 10)

### Working with Real Images

Place your images in the `input/images/` directory:

```bash
mkdir -p input/images
cp /path/to/your/images/*.jpg input/images/
python main.py --model random
```

## Built-in Models

### 1. RandomDetectorModel
- Generates 0-5 random bounding boxes per image
- Random class IDs across all available classes
- Random positions and sizes
- Confidence scores: 0.5-0.99
- **Use case:** General-purpose testing

### 2. ConservativeDetectorModel
- Generates 0-2 bounding boxes per image
- Prefers 2-3 specific classes
- Larger bounding boxes (min size: 0.1 normalized)
- Higher confidence: 0.7-0.99
- **Use case:** Simulating high-precision detector

### 3. AggressiveDetectorModel
- Generates 3-10 bounding boxes per image
- Detects across all available classes
- Smaller bounding boxes (min size: 0.02 normalized)
- Lower confidence: 0.3-0.9
- **Use case:** Simulating high-recall detector

## YOLO Format Explanation

The system generates annotations in YOLO format:

### Annotation File Format (`.txt`)
Each image has a corresponding `.txt` file with the same name. Each line represents one bounding box:

```
class_id x_center y_center width height
```

**Example (`image_001.txt`):**
```
0 0.512000 0.345000 0.234000 0.456000
1 0.723000 0.612000 0.156000 0.289000
```

**Field Descriptions:**
- `class_id`: Zero-indexed class ID (integer)
- `x_center`: Normalized x-coordinate of box center (0-1)
- `y_center`: Normalized y-coordinate of box center (0-1)
- `width`: Normalized width of box (0-1)
- `height`: Normalized height of box (0-1)

**Important:** Images with no objects have NO corresponding `.txt` file (YOLO convention).

### Dataset Configuration (`dataset.yaml`)

```yaml
path: /absolute/path/to/output
train: images/train
val: images/val

names:
  0: person
  1: car
  2: dog
  3: cat
  4: bicycle
  5: truck
```

## Adding Real Teacher Models

Replace fake models with actual detection models by creating a new class that inherits from `TeacherModel`:

### Example: Integrating YOLOv8

```python
from ultralytics import YOLO
from pathlib import Path
from typing import List
from PIL import Image

from src.models.base import TeacherModel
from src.data.structures import Annotation, BoundingBox


class YOLOv8TeacherModel(TeacherModel):
    """Real YOLOv8 teacher model."""
    
    def __init__(self, model_path: str, class_names: List[str]):
        """Initialize YOLOv8 model.
        
        Args:
            model_path: Path to YOLOv8 weights file
            class_names: List of class names
        """
        super().__init__(class_names)
        self.model = YOLO(model_path)
    
    def predict(self, image_path: Path) -> Annotation:
        """Run inference on an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Annotation with predicted bounding boxes
        """
        # Run inference
        results = self.model(image_path, verbose=False)[0]
        
        # Get image dimensions
        img = Image.open(image_path)
        img_width, img_height = img.size
        img.close()
        
        # Convert results to BoundingBox objects
        bounding_boxes = []
        for box in results.boxes:
            # Get box coordinates (xyxy format)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Convert to center format
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            # Create BoundingBox
            bbox = BoundingBox.from_pixels(
                class_id=int(box.cls[0]),
                x_center_px=x_center,
                y_center_px=y_center,
                width_px=width,
                height_px=height,
                img_width=img_width,
                img_height=img_height,
                confidence=float(box.conf[0])
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
        """Return model identifier."""
        return "YOLOv8"
```

### Using Your Custom Model

Modify `main.py` to use your custom model:

```python
from your_models import YOLOv8TeacherModel

# In the main() function, replace model initialization:
if args.model == 'yolov8':
    models = [YOLOv8TeacherModel('yolov8n.pt', DEFAULT_CLASS_NAMES)]
```

### Example: Integrating Other Models

The same pattern applies to any detection framework:

```python
class CustomTeacherModel(TeacherModel):
    def __init__(self, class_names: List[str]):
        super().__init__(class_names)
        # Initialize your model here
        
    def predict(self, image_path: Path) -> Annotation:
        # Run your model inference
        # Convert outputs to Annotation object
        pass
    
    def get_model_name(self) -> str:
        return "CustomModel"
```

## Next Steps for Training YOLOv11

After generating pseudo-labels:

1. **Install Ultralytics:**
```bash
pip install ultralytics
```

2. **Train YOLOv11:**
```python
from ultralytics import YOLO

# Load a pretrained model
model = YOLO('yolo11n.pt')

# Train the model
results = model.train(
    data='output/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='pseudo_labeled_model'
)
```

3. **Validate the model:**
```python
metrics = model.val()
```

4. **Run inference:**
```python
results = model('path/to/test/image.jpg')
```

## Expected Output Example

After running `python main.py --model ensemble --num-dummy-images 20`:

```
INFO: Found 0 images in input/images/
INFO: Creating 20 dummy images...
INFO: Created 20 dummy images in output/dummy_images
INFO: Found 20 images
INFO: Initializing 3 teacher models for ensemble
INFO: Processing images...
INFO: Processing image 1/20: img_001.jpg
...
INFO: Processing image 20/20: img_020.jpg
INFO: Splitting dataset (80%/20% train/val split)
INFO: Writing YOLO annotations...
INFO: Creating dataset.yaml...

===== Summary =====
INFO: Total images: 20
INFO: Train images: 16
INFO: Val images: 4
INFO: Images with detections: 18
INFO: Images without detections: 2
INFO: Average boxes per image: 4.2
INFO: Class distribution:
INFO:   - person: 23
INFO:   - car: 18
INFO:   - dog: 15
INFO:   - cat: 12
INFO:   - bicycle: 16
INFO:   - truck: 10
INFO: 
INFO: Output saved to: output
INFO: Ready for YOLOv11 training!
```

## Output Structure

```
output/
├── images/
│   ├── train/               # 16 training images
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── val/                 # 4 validation images
│       ├── img_017.jpg
│       └── ...
├── labels/
│   ├── train/               # Training annotations
│   │   ├── img_001.txt
│   │   ├── img_002.txt
│   │   └── ...
│   └── val/                 # Validation annotations
│       ├── img_017.txt
│       └── ...
└── dataset.yaml             # YOLO dataset configuration
```

## Code Quality Features

- **Type hints** on all functions and methods
- **Docstrings** following Google style
- **PEP 8** compliant code formatting
- **Error handling** for missing directories and invalid inputs
- **Logging** using Python's logging module
- **Modular design** following SOLID principles
- **No external ML dependencies** (except for real model integration)

## Troubleshooting

### No images found
If you see "No images found", the system will automatically create dummy images for testing:
```bash
python main.py --num-dummy-images 50
```

### Permission errors
Ensure you have write permissions for the output directory:
```bash
chmod -R u+w output/
```

### Import errors
Make sure you're running from the project root directory and dependencies are installed:
```bash
pip install -r requirements.txt
python main.py
```

## Contributing

To extend this project:

1. Add new teacher models in `src/models/`
2. Inherit from `TeacherModel` base class
3. Implement `predict()` and `get_model_name()` methods
4. Update `main.py` to include your model

## License

See LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{pseudolabeling_yolo,
  title = {Pseudo-Labeling System for YOLOv11},
  year = {2025},
  url = {https://github.com/gordinmitya/pseudolabeling-yolo}
}
```

## Contact

For questions or issues, please open an issue on GitHub.
