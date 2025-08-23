from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import tempfile
import uuid
from scene_localizer import ImprovedSceneLocalizer
import base64
from PIL import Image
import io
import json

app = Flask(__name__, static_folder='.')
CORS(app)  # Allow cross-origin requests

# Initialize the localizer once when the app starts
print("Initializing Scene Localizer...")
localizer = ImprovedSceneLocalizer()
print("Scene Localizer ready!")

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_scene():
    try:
        # Check if image and query are provided
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        if 'query' not in request.form:
            return jsonify({"error": "No query provided"}), 400
        
        image_file = request.files['image']
        query = request.form['query'].strip()
        
        if image_file.filename == '':
            return jsonify({"error": "No image selected"}), 400
        
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        # Create a temporary file for the uploaded image
        # Using tempfile ensures proper cleanup even if something goes wrong
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_path = temp_file.name
            
            # Save the uploaded image to temporary file
            image_file.save(temp_path)
        
        print(f"Processing image: {temp_path}")
        print(f"Query: '{query}'")
        
        try:
            # Run the scene localization
            detections = localizer.localize_scene(temp_path, query)
            
            # Process results
            if detections and len(detections) > 0:
                # Get the best detection
                best_detection = detections[0]
                bbox = best_detection['bbox']
                score = best_detection['score']
                
                # Load image to get dimensions for normalization
                with Image.open(temp_path) as img:
                    img_width, img_height = img.size
                
                # Normalize bounding box coordinates (0-1 range)
                normalized_bbox = {
                    "x": bbox[0] / img_width,
                    "y": bbox[1] / img_height, 
                    "width": (bbox[2] - bbox[0]) / img_width,
                    "height": (bbox[3] - bbox[1]) / img_height
                }
                
                # Determine confidence level
                if score > 0.5:
                    confidence_level = "High"
                elif score > 0.35:
                    confidence_level = "Medium"  
                else:
                    confidence_level = "Low"
                
                result = {
                    "success": True,
                    "query": query,
                    "confidence": float(score),
                    "confidenceLevel": confidence_level,
                    "boundingBox": normalized_bbox,
                    "description": f"Object detected with {confidence_level.lower()} confidence ({score:.1%})",
                    "queryUsed": best_detection.get('query_used', query),
                    "detectionCount": len(detections)
                }
                
                # Add additional detections if available
                if len(detections) > 1:
                    additional_detections = []
                    for i, det in enumerate(detections[1:3], 2):  # Max 2 more
                        det_bbox = det['bbox']
                        additional_detections.append({
                            "confidence": float(det['score']),
                            "boundingBox": {
                                "x": det_bbox[0] / img_width,
                                "y": det_bbox[1] / img_height,
                                "width": (det_bbox[2] - det_bbox[0]) / img_width,
                                "height": (det_bbox[3] - det_bbox[1]) / img_height
                            }
                        })
                    result["additionalDetections"] = additional_detections
                
            else:
                result = {
                    "success": False,
                    "query": query,
                    "confidence": 0.0,
                    "confidenceLevel": "None",
                    "boundingBox": {"x": 0, "y": 0, "width": 0, "height": 0},
                    "description": "No confident detection found. Try rephrasing your query or using more specific terms.",
                    "detectionCount": 0,
                    "suggestions": [
                        "Be more specific (e.g., 'red car' instead of 'car')",
                        "Try alternative descriptions (e.g., 'person walking' vs 'pedestrian')",
                        "Use broader categories if being too specific",
                        "Ensure the object is clearly visible in the image"
                    ]
                }
            
            print(f"Analysis complete. Success: {result['success']}")
            return jsonify(result)
            
        except Exception as processing_error:
            print(f"Processing error: {processing_error}")
            return jsonify({
                "error": "Failed to process image", 
                "details": str(processing_error),
                "success": False
            }), 500
        
        finally:
            # Clean up: remove temporary file
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    print(f"Cleaned up temp file: {temp_path}")
            except Exception as cleanup_error:
                print(f"Warning: Could not clean up temp file: {cleanup_error}")
    
    except Exception as e:
        print(f"API error: {e}")
        return jsonify({
            "error": "Internal server error", 
            "details": str(e),
            "success": False
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        "status": "healthy",
        "model_loaded": localizer is not None,
        "device": str(localizer.device) if localizer else "unknown"
    })

@app.route('/api/info', methods=['GET']) 
def get_info():
    """Get system information"""
    return jsonify({
        "model": "CLIP-ViT-B/32",
        "features": [
            "Smart Query Expansion",
            "Adaptive Window Sizing", 
            "Confidence-Based Detection",
            "Multiple Detection Support"
        ],
        "supported_formats": ["JPG", "JPEG", "PNG", "BMP", "TIFF", "WebP", "GIF"],
        "max_detections": 3
    })

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Please use images under 16MB."}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error occurred"}), 500

if __name__ == '__main__':
    # Render sets PORT environment variable
    port = int(os.environ.get('PORT', 10000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"Starting Flask app on host 0.0.0.0 port {port}")
    print(f"Debug mode: {debug_mode}")
    
    # Ensure we bind to all interfaces and the correct port
    app.run(
        host='0.0.0.0', 
        port=port, 
        debug=False,  # Always False in production
        threaded=True
    )
