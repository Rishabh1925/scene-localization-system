from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import base64
import io
from PIL import Image
from scene_localizer import ImprovedSceneLocalizer
import traceback
import json

app = Flask(__name__)
CORS(app)
localizer = None

def initialize_localizer():
    """Initialize the scene localizer"""
    global localizer
    if localizer is None:
        try:
            localizer = ImprovedSceneLocalizer()
            print("Scene localizer initialized successfully")
        except Exception as e:
            print(f"Error initializing localizer: {e}")
            traceback.print_exc()

@app.route('/')
def serve_index():
    """Serve the HTML file"""
    return send_from_directory('.', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Analyze image with query"""
    try:
        if localizer is None:
            initialize_localizer()
            if localizer is None:
                return jsonify({'error': 'Failed to initialize AI model'}), 500
        
        data = request.json
        if not data:
            return jsonify({'error': 'No data received'}), 400
        
        image_data = data.get('image')
        query = data.get('query')
        
        if not image_data or not query:
            return jsonify({'error': 'Image and query are required'}), 400
        
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            
            temp_path = 'temp_image.jpg'
            with open(temp_path, 'wb') as f:
                f.write(image_bytes)
            
            print(f"Processing query: '{query}'")
            
        except Exception as e:
            return jsonify({'error': f'Failed to process image: {str(e)}'}), 400
        
        try:
            detections = localizer.localize_scene(temp_path, query)
            
            results = []
            for i, detection in enumerate(detections):
                bbox = detection['bbox']
                score = detection['score']
                
                if score > 0.5:
                    confidence = 'High'
                    confidence_color = '#10b981'
                elif score > 0.35:
                    confidence = 'Medium'
                    confidence_color = '#f59e0b'
                else:
                    confidence = 'Low'
                    confidence_color = '#ef4444'
                
                result = {
                    'id': i + 1,
                    'bbox': {
                        'x': bbox[0],
                        'y': bbox[1],
                        'width': bbox[2] - bbox[0],
                        'height': bbox[3] - bbox[1]
                    },
                    'score': round(score, 4),
                    'confidence': confidence,
                    'confidence_color': confidence_color,
                    'query_used': detection.get('query_used', query)
                }
                results.append(result)
            
            try:
                os.remove(temp_path)
            except:
                pass
            
            result_image_path = None
            if os.path.exists('improved_result.jpg'):
                try:
                    with open('improved_result.jpg', 'rb') as f:
                        result_image_data = base64.b64encode(f.read()).decode()
                        result_image_path = f"data:image/jpeg;base64,{result_image_data}"
                except:
                    pass
            
            response = {
                'success': True,
                'detections': results,
                'total_detections': len(results),
                'result_image': result_image_path,
                'message': f'Found {len(results)} detection(s)' if results else 'No confident detections found'
            }
            return jsonify(response)
            
        except Exception as e:
            print(f"Detection error: {e}")
            traceback.print_exc()
            return jsonify({'error': f'Detection failed: {str(e)}'}), 500
            
    except Exception as e:
        print(f"Server error: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': localizer is not None
    })

if __name__ == '__main__':

    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        print("Starting Scene Localization Server...")
        print("Initializing AI model (this may take a moment)...")

    initialize_localizer()

    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        print("Server is ready. Press Ctrl+C to stop.")
        print("\n--> Open your browser and go to: http://127.0.0.1:5000\n")

    app.run(debug=True, port=5000)
