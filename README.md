# Scene Localization System

A powerful computer vision tool that uses CLIP (Contrastive Language-Image Pre-Training) to locate and identify objects or scenes in images based on natural language queries.

## Features

- **Smart Query Expansion**: Automatically generates variations of your search query for better detection
- **Adaptive Window Sizing**: Dynamically adjusts detection windows based on image dimensions
- **Confidence-Based Detection**: Provides confidence scores and quality assessments for each detection
- **Professional Visualization**: Creates high-quality result images with bounding boxes and crops
- **Multiple Detection Support**: Finds up to 3 best matches per query
- **Metadata Export**: Saves detailed information about each detection

## System Requirements

- Python 3.7 or higher
- CUDA-compatible GPU (optional but recommended for faster processing)
- At least 4GB RAM
- 2GB free disk space for model downloads

## Quick Start

### 1. Create Project Directory

First, create a new folder on your computer for this project:

**Windows:**
```cmd
mkdir scene-localization
cd scene-localization
```

**macOS/Linux:**
```bash
mkdir scene-localization
cd scene-localization
```

### 2. Clone the Repository

```bash
git clone https://github.com/Rishabh1925/scene_localization_system.git
cd scene_localization_system
```

### 3. Set Up Virtual Environment (Recommended)

**macOS/Linux:**
```bash
python3 -m venv scene_env
source scene_env/bin/activate
```

**Windows:**
```cmd
python -m venv scene_env
scene_env\Scripts\activate
```

### 4. Install Dependencies

Create a `requirements.txt` file with the following content:

```txt
torch
torchvision
transformers
opencv-python
numpy
Pillow
matplotlib
scipy
flask
flask-cors
```

Then install:

```bash
pip install -r requirements.txt
```

### 5. Verify Installation

```bash
python -c "import torch, transformers, cv2, PIL, flask; print('All dependencies installed successfully!')"
```

## Usage

### Starting the Application

1. Start the web application:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to `http://127.0.0.1:5000` (or the address displayed in your terminal)
3. Upload your image files (JPG, JPEG, PNG, BMP, TIFF, WebP, GIF) through the web interface
4. Enter your search queries and click "Analyze Image"
5. **Please be patient!** The analysis takes time to process (usually 1 - 5 minutes depending on your hardware and image complexity). The system is running complex AI computations in the background.

## While You Wait...

**Think About This Project:**
- How do you think this system combines computer vision and natural language processing?
- What makes CLIP special compared to traditional image recognition systems?
- Why might the sliding window approach be effective for object localization?

**Fun CLIP & Vision Transformer Trivia:**
- **CLIP stands for**: Contrastive Language-Image Pre-Training - it learned from 400 million image-text pairs!
- **ViT Revolution**: Vision Transformers (ViT) proved that the transformer architecture (originally for text) could beat CNNs at image tasks
- **Zero-shot Magic**: CLIP can recognize objects it was never explicitly trained to identify - it just needs a text description
- **Multimodal Learning**: CLIP understands both images AND text in the same mathematical space - that's why you can search images with natural language

## Output Files

The system generates:

- **`improved_result.jpg`**: Main visualization with bounding boxes, confidence scores, and quality ratings
- **`improved_detections/` folder**: Individual cropped images and metadata files for each detection

Example metadata file:
```txt
Query: person talking
Matched Query: two people conversing
Confidence Score: 0.6470
Bounding Box: (234, 156, 456, 389)
Window Size: (200, 200)
Crop Size: 222x233 pixels
```

## Query Tips

### Good Query Examples

- **Specific objects**: "red car", "brown dog", "person wearing hat"
- **Actions**: "person walking", "dog running"
- **Scenes**: "street vendor", "outdoor café"
- **Relationships**: "two people conversing", "person with bicycle"

### Best Practices

- **Be specific**: Use descriptive terms like "red sports car" instead of just "car"
- **Try alternatives**: Use synonyms if initial queries don't work
- **Use high-quality images**: Clear, well-lit images with visible objects work best
- **Optimize image size**: Resize large images (>2000px) for better performance
- **Use GPU acceleration**: CUDA-compatible GPU recommended for faster processing

## Technical Details

- **Base Model**: OpenAI CLIP-ViT-B/32
- **Input Resolution**: Images processed at 224x224 pixels
- **Algorithm**: Sliding window with cosine similarity between CLIP embeddings, non-maximum suppression for overlapping detections

## Project Structure

```
scene-localization-system/
│
├── app.py                         # Main Flask web application
├── index.html                     # HTML templates
├── README.md                      # Documentation
├── requirements.txt               # Python dependencies
├── scene_localizer.py
│ 
│
├── scene_env/                     # Virtual environment (After you create it)
│
├── static/       
│   └── images/
│     ├── test1.jpg
│     ├── test2.png
│     └── test3.png       
│
└── improved_detections/           # Output folder
    ├── detection_1_score_0.647_confidence_high.jpg
    ├── detection_1_score_0.304_confidence_low.jpg
    └── ...
```

## 📄 License

This project uses components under the following licenses:

- **CLIP Model**: MIT License
- **PyTorch**: BSD License
- **Transformers**: Apache 2.0 License

## 🛠️ Troubleshooting

If you encounter issues, check:

1. All dependencies are properly installed
2. Image files are in supported formats
3. Objects you're searching for are clearly visible in the image

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For questions or issues, please open an issue on GitHub or contact the maintainer.

---

**Made with ❤️ using CLIP and Python**
