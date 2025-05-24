# Document Classifier

A machine learning system for classifying different types of documents using computer vision and pattern recognition techniques. The system can distinguish between comics, books, manuscripts, typewritten documents, and tickets.

## Features

- **Multiple Classifier Configurations**: Four different classifier setups with varying preprocessing techniques
- **Document Scanning**: Automatic document rectification and perspective correction
- **Flexible Preprocessing**: Support for grayscale conversion and image transformations
- **Dimensionality Reduction**: Optional Linear Discriminant Analysis (LDA) for feature reduction
- **Model Persistence**: Save and load trained models for reuse
- **Comprehensive Evaluation**: Detailed performance metrics and classification reports

## Document Classes

The system classifies documents into five categories:

- **Comics**: Comic books and graphic novels
- **Libros** (Books): Regular printed books
- **Manuscrito** (Manuscript): Handwritten documents
- **Mecanografiado** (Typewritten): Typewritten documents
- **Tickets**: Receipts, tickets, and similar small documents

## Classifier Configurations

| Classifier | LDA | Document Scanner | Grayscale |
|------------|-----|------------------|-----------|
| C1         | ❌   | ❌                | ❌         |
| C2         | ✅   | ❌                | ❌         |
| C3         | ❌   | ✅                | ✅         |
| C4         | ✅   | ✅                | ✅         |

## Requirements

```bash
pip install opencv-python numpy scikit-learn joblib argparse
```

### Dependencies

- `opencv-python`: Image processing and computer vision
- `numpy`: Numerical computations
- `scikit-learn`: Machine learning algorithms
- `joblib`: Model serialization
- `argparse`: Command-line argument parsing

## Installation

1. Clone or download the repository
2. Install the required dependencies
3. Ensure you have the `scanner.py` module (DocumentScanner class)
4. Create the following directory structure:

```
project/
├── data/
│   ├── Learning/
│   │   ├── comics/
│   │   ├── libros/
│   │   ├── manuscrito/
│   │   ├── mecanografiado/
│   │   └── tickets/
│   └── Test/
│       ├── comics/
│       ├── libros/
│       ├── manuscrito/
│       ├── mecanografiado/
│       └── tickets/
├── models/ (created automatically)
└── main.py
```

## Usage

### Training Mode

Train all classifier configurations:

```bash
python main.py --train
```

This will:
- Load training images from `./data/Learning/`
- Train four different classifier configurations
- Save models to `./models/` directory

### Evaluation Mode

Evaluate all trained classifiers:

```bash
python main.py --test
```

This will:
- Load test images from `./data/Test/`
- Evaluate each classifier configuration
- Display accuracy scores and classification reports

### Single Image Prediction

Classify a single image:

```bash
python main.py path/to/image.jpg
```

Uses the best performing classifier (C2 by default) to predict the document class.

## Configuration

### Key Parameters

- `IMG_SIZE = (400, 300)`: Standard image dimensions for processing
- `BEST_CLASSIFIER = 1`: Index of the best performing classifier for predictions
- `TRAIN_DIR = "./data/Learning"`: Training data directory
- `TEST_DIR = "./data/Test"`: Test data directory

### Classifier Configurations

Modify `CLASSIFIER_CONFIGS` to experiment with different preprocessing combinations:

```python
CLASSIFIER_CONFIGS = {
    'C1': {'use_lda': False, 'use_scanner': False, 'to_gray': False},
    'C2': {'use_lda': True, 'use_scanner': False, 'to_gray': False},
    'C3': {'use_lda': False, 'use_scanner': True, 'to_gray': True},
    'C4': {'use_lda': True, 'use_scanner': True, 'to_gray': True}
}
```

## Data Organization

Organize your training and test data in the following structure:

```
data/
├── Learning/
│   ├── comics/
│   │   ├── comic1.jpg
│   │   ├── comic2.png
│   │   └── ...
│   ├── libros/
│   │   ├── book1.jpg
│   │   └── ...
│   └── ... (other classes)
└── Test/
    ├── comics/
    │   ├── test_comic1.jpg
    │   └── ...
    └── ... (other classes)
```

## Model Architecture

### Base Classifier
- **Algorithm**: Support Vector Machine (SVM) with linear kernel
- **Features**: Raw pixel values or LDA-transformed features
- **Preprocessing**: StandardScaler for feature normalization

### Optional Components
- **Document Scanner**: Automatic document rectification using perspective transformation
- **Linear Discriminant Analysis**: Dimensionality reduction to 4 components
- **Grayscale Conversion**: Color to grayscale transformation for certain configurations

## Output Examples

### Training Output
```
[TRAINING COMPLETED] Classifier C1
[TRAINING COMPLETED] Classifier C2
[TRAINING COMPLETED] Classifier C3
[TRAINING COMPLETED] Classifier C4
```

### Evaluation Output
```
[EVALUATING] Classifier C1
Accuracy: 0.8542
              precision    recall  f1-score   support
...

=== RESULTS SUMMARY ===
C1: 0.8542
C2: 0.8790
C3: 0.8333
C4: 0.8611
```

### Prediction Output
```
Prediction: libros
```

## File Structure

- `main.py`: Main script with all functionality
- `scanner.py`: Document scanner module (required dependency)
- `models/`: Directory containing trained models
  - `model_C*.joblib`: Trained SVM models
  - `scaler_C*.joblib`: Feature scalers
  - `model_C*_lda.joblib`: LDA transformers (for applicable classifiers)

## Performance Tips

1. **Image Quality**: Use high-quality, well-lit images for better results
2. **Consistent Sizing**: The system resizes images to 400x300, maintain aspect ratios when possible
3. **Data Balance**: Ensure balanced representation of all document classes in training data
4. **Scanner Usage**: Use document scanner for images with perspective distortion or poor alignment

## Troubleshooting

### Common Issues

1. **"Could not load image" error**: Check file path and image format compatibility
2. **"Could not rectify image" error**: Document scanner failed - image may be too distorted
3. **Low accuracy**: Ensure sufficient training data and proper class balance
4. **Missing models**: Run training mode before evaluation or prediction

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

## License

This project is provided as-is for educational and research purposes.

## Contributing

When contributing to this project:
1. Maintain code style and documentation standards
2. Test new features with multiple classifier configurations
3. Update this README for any new functionality or configuration options