# Improved Scene Localization System

A powerful computer vision tool that uses CLIP (Contrastive Language-Image Pre-Training) to locate and identify objects or scenes in images based on natural language queries.

## Features

- **Smart Query Expansion**: Automatically generates variations of your search query for better detection
- **Adaptive Window Sizing**: Dynamically adjusts detection windows based on image dimensions
- **Confidence-Based Detection**: Provides confidence scores and quality assessments for each detection
- **Professional Visualization**: Creates high-quality result images with bounding boxes and crops
- **Multiple Detection Support**: Finds up to 3 best matches per query
- **Metadata Export**: Saves detailed information about each detection

## Requirements

### System Requirements
- Python 3.7 or higher
- CUDA-compatible GPU (optional but recommended for faster processing)
- At least 4GB RAM
- 2GB free disk space for model downloads

## Installation

### Step 1: Get the Project

Choose one of the following methods:

#### Option A: Git Clone (Recommended)
```bash
# Navigate to the project directory
cd scene_localization_system

# Clone the repository
git clone https://github.com/Rishabh1925/scene_localization_system.git
```

#### Option B: Manual Download
- Create a new folder on your computer and name it `SceneLocalization`
- Download the `scene_localizer.py` file and place it inside the `SceneLocalization` folder
- Navigate to the directory:
```bash
cd /path/to/SceneLocalization
```

### Step 2: Install Dependencies

#### Option A: Create `requirements.txt` (Recommended)
Inside the project folder, create a file named `requirements.txt` and add the following content to it:
```
torch>=1.9.0
torchvision>=0.10.0
transformers>=4.21.0
opencv-python>=4.5.0
pillow>=8.0.0
matplotlib>=3.3.0
numpy>=1.19.0
scipy>=1.7.0
```

Then install with:
```bash
pip install -r requirements.txt
```

#### Option B: Using pip
```bash
pip install torch torchvision transformers opencv-python pillow matplotlib numpy scipy
```

#### Option C: Using conda
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers opencv-python pillow matplotlib scipy
```

### Step 3: Verify Installation
Test your installation by running:
```bash
python -c "import torch, transformers, cv2, PIL; print('All dependencies installed successfully!')"
```

### Step 4: Create Virtual Environment (Optional but Recommended)

**Windows:**
```cmd
# Create virtual environment
python -m venv scene_env

# Activate virtual environment
scene_env\Scripts\activate
```

**macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv scene_env

# Activate virtual environment
source scene_env/bin/activate
```

## Project Structure

```
scene-localization-system/
│
├── scene_localizer.py          # Main application script
├── README.md                   # This documentation file
├── requirements.txt            # Python dependencies
│            
├── street_scene.jpg
├── market_photo.png
├── office_meeting.jpeg
├── park_dogs.bmp
│
├── scene_env/                  # Virtual Environment Folder Created
│
└── improved_detections/        
    ├── detection_1_score_0.647_confidence_high.jpg
    ├── detection_1_score_0.647_confidence_high_metadata.txt
    ├── detection_2_score_0.423_confidence_medium.jpg
    └── detection_2_score_0.423_confidence_medium_metadata.txt
```

## Usage

### Basic Usage

1. **Place your images** in the same directory as `scene_localizer.py`
   - Supported formats: JPG, JPEG, PNG, BMP, TIFF, WebP, GIF

2. **Command Line Usage**

```bash
# Run the script
python scene_localizer.py
```

3. **Follow the interactive prompts**:
   - Select an image (if multiple are available)
   - Enter search queries when prompted
   - View results in the generated visualization

## Output Files

### Main Visualization (`improved_result.jpg`)
- Shows the original image with bounding boxes around detected objects
- Displays confidence scores and quality ratings
- Includes cropped regions for detailed inspection

### Individual Detections (`improved_detections/` folder)
Each detection is saved as:
- `detection_1_score_0.647_confidence_high.jpg` - Cropped image
- `detection_1_score_0.647_confidence_high_metadata.txt` - Detailed information

### Metadata File Contents
```
Query: person talking
Matched Query: two people conversing
Confidence Score: 0.6470
Bounding Box: (234, 156, 456, 389)
Window Size: (200, 200)
Crop Size: 222x233 pixels
```

## Query Examples

### Good Query Examples
- **Specific objects**: "red car", "brown dog", "person wearing hat"
- **Actions**: "person walking", "dog running", "people talking"
- **Scenes**: "street vendor", "market stall", "outdoor café"
- **Relationships**: "two people conversing", "person with bicycle"

### Tips for Better Results
1. **Be specific**: "red sports car" vs "car"
2. **Use descriptive adjectives**: "elderly person", "large building"
3. **Include context**: "person selling goods", "dog playing in park"
4. **Try variations**: If one query doesn't work, try synonyms

### Performance Optimization

#### For Faster Processing
1. **Use GPU**: Install CUDA-compatible PyTorch
2. **Reduce image size**: Resize large images (>2000px) before processing
3. **Use specific queries**: Avoid overly broad terms

#### For Better Accuracy
1. **High-quality images**: Use clear, well-lit images
2. **Appropriate size**: Images should be at least 400x400 pixels
3. **Visible objects**: Ensure target objects are clearly visible and not occluded

## Technical Details

### Model Information
- **Base Model**: OpenAI CLIP-ViT-B/32
- **Input Resolution**: 224x224 pixels (for CLIP processing)
- **Feature Dimensions**: 512-dimensional embeddings
- **Languages Supported**: English (primary), limited multilingual support

### Detection Algorithm
1. **Image Preprocessing**: Resize and normalize input images
2. **Query Expansion**: Generate semantic variations of input queries
3. **Sliding Window**: Apply adaptive window sizes across the image
4. **Feature Extraction**: Extract CLIP embeddings for image regions and text queries
5. **Similarity Computation**: Calculate cosine similarity between image and text features
6. **Non-Maximum Suppression**: Remove overlapping detections
7. **Confidence Filtering**: Keep only high-confidence results

## License

This project uses the following open-source components:
- **CLIP Model**: MIT License (OpenAI)
- **PyTorch**: BSD License
- **Transformers**: Apache 2.0 License (Hugging Face)

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Ensure all dependencies are properly installed
3. Verify image files are in supported formats
4. Check that images are clear and objects are visible

## Version History

- **v1.0**: Initial release with basic CLIP-based detection
- **v2.0**: Added smart query expansion and adaptive windowing
- **v2.1**: Improved visualization and metadata export
- **v2.2**: Enhanced error handling and fallback detection
