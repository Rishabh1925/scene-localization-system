import matplotlib
matplotlib.use('Agg')
import torch
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
import warnings
import glob
from scipy.ndimage import maximum_filter
warnings.filterwarnings('ignore')

class ImprovedSceneLocalizer:
    def __init__(self, device=None):
        """Initializing scene localization system"""
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nInitializing Scene Localization System...")
        print(f"Using device: {self.device}")
        
        print("Loading CLIP-ViT-B/32 Model...\n")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(self.device)
        self.model.eval()
        
        print("\nModel loaded successfully!\n")
    
    def preprocess_image(self, image_path):
        """Load and validate image"""
        try:
            image = Image.open(image_path).convert('RGB')
            if image.size[0] < 200 or image.size[1] < 200:
                print("Image too small, resizing...")
                image = image.resize((max(400, image.size[0]), max(400, image.size[1])), Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            return None
    
    def generate_smart_queries(self, original_query):
        """Generating contextually relevant query variations"""
        query_lower = original_query.lower().strip()
        queries = [original_query]
        query_expansions = {
            'person talking': [
                'two people conversing', 'people having conversation', 
                'man and woman talking', 'person speaking', 'dialogue between people'
            ],
            'people talking': [
                'conversation between people', 'group discussion', 
                'people conversing', 'social interaction', 'people communicating'
            ],
            'vendor': [
                'street vendor', 'market seller', 'person selling goods', 
                'shopkeeper', 'merchant at stall'
            ],
            'street vendor': [
                'person selling items', 'market vendor', 'street seller',
                'vendor with goods', 'person at market stall'
            ],
            'car': ['automobile', 'vehicle', 'motor car', 'parked car'],
            'person walking': ['pedestrian', 'walking person', 'person moving', 'walking human'],
            'dog': ['canine', 'pet dog', 'domestic dog'],
            'cat': ['feline', 'domestic cat', 'pet cat']
        }
        
        for key, expansions in query_expansions.items():
            if key in query_lower:
                queries.extend(expansions[:3])  
                break
        else:
            if 'person' in query_lower:
                queries.append(query_lower.replace('person', 'human'))
                queries.append(query_lower.replace('person', 'individual'))
            elif any(word in query_lower for word in ['man', 'woman', 'people']):
                queries.append(f"human {query_lower}")
        
        return list(dict.fromkeys(queries))[:5]
    
    def adaptive_sliding_window(self, image, query_variations):
        """Sliding window"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        print(f"Analyzing image: {w}x{h} pixels")
        
        base_size = min(w, h) // 4
        window_sizes = [
            (base_size, base_size),
            (int(base_size * 1.2), int(base_size * 1.2)),
            (int(base_size * 1.5), int(base_size * 1.5)),
            (int(base_size * 0.8), int(base_size * 1.2)),  
            (int(base_size * 1.2), int(base_size * 0.8)),  
        ]
        
        window_sizes = [(ws_w, ws_h) for ws_w, ws_h in window_sizes 
                       if ws_w <= w * 0.8 and ws_h <= h * 0.8]
        
        print(f"Using {len(window_sizes)} window sizes")
    
        text_features_list = []
        for query in query_variations:
            text_inputs = self.processor(text=[query], return_tensors="pt", padding=True)
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_features_list.append(text_features)
        
        all_scores = []
        positions = []
        
        for window_w, window_h in window_sizes:
            stride = max(20, min(window_w, window_h) // 6)
            
            for y in range(0, h - window_h + 1, stride):
                for x in range(0, w - window_w + 1, stride):
                    x2, y2 = x + window_w, y + window_h
                    
                    window_img = image.crop((x, y, x2, y2))
                    
                    image_inputs = self.processor(images=window_img, return_tensors="pt", padding=True)
                    image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
                    
                    with torch.no_grad():
                        image_features = self.model.get_image_features(**image_inputs)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        
                        max_similarity = 0
                        best_query_idx = 0
                        
                        for i, text_features in enumerate(text_features_list):
                            similarity = torch.cosine_similarity(text_features, image_features, dim=-1)
                            score = similarity.item()
                            
                            if score > max_similarity:
                                max_similarity = score
                                best_query_idx = i
                    
                    all_scores.append(max_similarity)
                    positions.append({
                        'bbox': (x, y, x2, y2),
                        'score': max_similarity,
                        'window_size': (window_w, window_h),
                        'query_used': query_variations[best_query_idx]
                    })
        
        if not positions:
            return []
        
        scores_array = np.array(all_scores)
        
        threshold = max(0.15, np.percentile(scores_array, 85))  
        good_detections = [pos for pos, score in zip(positions, all_scores) if score >= threshold]
        
        if not good_detections:
            best_idx = np.argmax(scores_array)
            good_detections = [positions[best_idx]]
        
        good_detections.sort(key=lambda x: x['score'], reverse=True)
        
        final_detections = []
        for detection in good_detections[:10]: 
            bbox = detection['bbox']
            is_suppressed = False
            
            for final_det in final_detections:
                iou = self.calculate_iou(bbox, final_det['bbox'])
                if iou > 0.3: 
                    is_suppressed = True
                    break
            
            if not is_suppressed:
                final_detections.append(detection)
                if len(final_detections) >= 3: 
                    break
        
        return final_detections
    
    def calculate_iou(self, box1, box2):
        """Calculating Intersection over Union"""
        x1, y1, x2, y2 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1, yi1 = max(x1, x1_2), max(y1, y1_2)
        xi2, yi2 = min(x2, x2_2), min(y2, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def create_professional_visualization(self, image_path, detections, query_text, save_path="improved_result.jpg"):
        """Creating high-quality visualization"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print("Could not load image for visualization")
                return
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]
            
            if not detections:
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                ax.imshow(image_rgb)
                ax.set_title(f'Query: "{query_text}"\nNo confident detection found', 
                            fontsize=16, fontweight='bold', color='red')
                ax.axis('off')
                
                plt.tight_layout()
                try:
                    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
                    print(f"No detection result saved to {save_path}")
                except Exception as save_error:
                    print(f"Could not save image: {save_error}")
                
                plt.show()
                plt.close()
                return
            
            num_detections = min(len(detections), 3)
            
            if num_detections == 0:
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                axes = [ax]
            else:
                total_plots = 1 + num_detections
                fig_width = 6 * total_plots if total_plots <= 2 else 6 + 4 * (total_plots - 1)
                fig, axes = plt.subplots(1, total_plots, figsize=(fig_width, 8))

                if total_plots == 1:
                    axes = [axes]
                elif not isinstance(axes, (list, np.ndarray)):
                    axes = [axes]
                else:
                    axes = list(axes) if isinstance(axes, np.ndarray) else axes
            
            main_ax = axes[0]
            main_ax.imshow(image_rgb)
            main_ax.set_title(f'Query: "{query_text}"', fontsize=16, fontweight='bold', pad=20)
            main_ax.axis('off')
            
            colors = ['red', 'blue', 'green']

            for i, detection in enumerate(detections[:3]):
                bbox = detection['bbox']
                score = detection['score']
                color = colors[i % len(colors)]

                rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                   fill=False, edgecolor=color, linewidth=4, alpha=0.8)
                main_ax.add_patch(rect)

                confidence_level = "High" if score > 0.5 else "Medium" if score > 0.35 else "Low"
                text_color = 'white' if score > 0.35 else 'black'
                
                main_ax.text(bbox[0], max(10, bbox[1] - 10), 
                            f'#{i+1}: {score:.3f} ({confidence_level})', 
                            bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.8),
                            fontsize=11, color=text_color, fontweight='bold')

            for i, detection in enumerate(detections[:num_detections]):
                if i + 1 >= len(axes):
                    break
                    
                bbox = detection['bbox']
                score = detection['score']
                query_used = detection.get('query_used', query_text)

                y1, y2 = max(0, bbox[1]), min(h, bbox[3])
                x1, x2 = max(0, bbox[0]), min(w, bbox[2])
                
                if y2 > y1 and x2 > x1:
                    cropped = image_rgb[y1:y2, x1:x2]
                    
                    ax = axes[i + 1]
                    ax.imshow(cropped)

                    if score > 0.5:
                        quality = "Excellent Match"
                        title_color = 'green'
                    elif score > 0.35:
                        quality = "Good Match"
                        title_color = 'orange'
                    else:
                        quality = "Possible Match"
                        title_color = 'red'
                    
                    title = f'Detection #{i+1}\nScore: {score:.3f}\n{quality}'
                    if query_used != query_text:
                        title += f'\nMatched: "{query_used[:20]}..."' if len(query_used) > 20 else f'\nMatched: "{query_used}"'
                    
                    ax.set_title(title, fontsize=10, fontweight='bold', color=title_color, pad=10)
                    ax.axis('off')
            
            plt.tight_layout()

            try:
                plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', 
                           edgecolor='none', pad_inches=0.2)
                print(f"High-quality result saved to {save_path}")
            except Exception as save_error:
                print(f"Could not save visualization: {save_error}")
            
            plt.show()
            
            try:
                os.makedirs("improved_detections", exist_ok=True)
                
                for i, detection in enumerate(detections):
                    bbox = detection['bbox']
                    score = detection['score']

                    y1, y2 = max(0, bbox[1]), min(h, bbox[3])
                    x1, x2 = max(0, bbox[0]), min(w, bbox[2])
                    
                    if y2 > y1 and x2 > x1:
                        cropped = image_rgb[y1:y2, x1:x2]

                        confidence_label = 'high' if score > 0.5 else 'medium' if score > 0.35 else 'low'
                        crop_filename = f"improved_detections/detection_{i+1}_score_{score:.3f}_confidence_{confidence_label}.jpg"
                        
                        crop_pil = Image.fromarray(cropped)
                        crop_pil.save(crop_filename, quality=95, optimize=True)

                        metadata_file = crop_filename.replace('.jpg', '_metadata.txt')
                        with open(metadata_file, 'w') as f:
                            f.write(f"Query: {query_text}\n")
                            f.write(f"Matched Query: {detection.get('query_used', query_text)}\n")
                            f.write(f"Confidence Score: {score:.4f}\n")
                            f.write(f"Bounding Box: {bbox}\n")
                            f.write(f"Window Size: {detection.get('window_size', 'Unknown')}\n")
                            f.write(f"Crop Size: {cropped.shape[1]}x{cropped.shape[0]} pixels\n")
                
                print(f"{len(detections)} detection(s) saved to improved_detections/ folder")
                
            except Exception as save_error:
                print(f"Could not save individual crops: {save_error}")
                
        except Exception as viz_error:
            print(f"Visualization error: {viz_error}")
            print("Detection results (text only):")
            for i, detection in enumerate(detections, 1):
                bbox = detection['bbox']
                score = detection['score']
                print(f"   Detection #{i}: Score={score:.3f}, BBox=({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})")
    
    def localize_scene(self, image_path, query_text):
        """Main localization function with robust error handling"""
        try:
            if not image_path:
                raise ValueError("image_path cannot be empty or None")
            if not query_text:
                raise ValueError("query_text cannot be empty or None")
            
            print(f"\nStarting Scene Localization...")
            print(f"Query: '{query_text}'")
            print(f"Image: {image_path}")
            
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                return []
            
            image = self.preprocess_image(image_path)
            if image is None:
                print("Failed to load image")
                return []
            
            print(f"Image size: {image.size[0]}x{image.size[1]} pixels")

            query_variations = self.generate_smart_queries(query_text)
            print(f"Using {len(query_variations)} query variations:")
            for i, q in enumerate(query_variations, 1):
                print(f"   {i}. '{q}'")

            try:
                detections = self.adaptive_sliding_window(image, query_variations)
            except Exception as detection_error:
                print(f"Detection error: {detection_error}")
                print("Trying with simplified parameters...")

                try:
                    detections = self.basic_sliding_window_fallback(image, [query_text])
                except Exception as fallback_error:
                    print(f"Fallback detection also failed: {fallback_error}")
                    detections = []
            
            if not detections:
                print("No confident detections found")
                print("Suggestions:")
                print("   - Try more specific terms (e.g., 'red car' vs 'car')")
                print("   - Use alternative descriptions (e.g., 'person walking' vs 'pedestrian')")
                print("   - Try broader categories (e.g., 'vehicle' vs 'specific car model')")
                print("   - Check if the object is clearly visible in the image")

                try:
                    self.create_professional_visualization(image_path, [], query_text)
                except Exception as viz_error:
                    print(f"Could not create visualization: {viz_error}")
                
                return []
            
            print(f"Found {len(detections)} confident detection(s)!")

            for i, det in enumerate(detections, 1):
                bbox = det['bbox']
                score = det['score']
                query_used = det.get('query_used', query_text)
                
                confidence = "HIGH" if score > 0.5 else "MEDIUM" if score > 0.35 else "LOW"
                
                print(f"  Detection #{i}:")
                print(f"     Score: {score:.3f} ({confidence} confidence)")
                print(f"     Location: ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})")
                print(f"     Size: {bbox[2]-bbox[0]}x{bbox[3]-bbox[1]} pixels")
                if query_used != query_text:
                    print(f"     Matched query: '{query_used}'")

            try:
                self.create_professional_visualization(image_path, detections, query_text)
            except Exception as viz_error:
                print(f"Visualization error: {viz_error}")
                print("Results saved as text summary instead")
            
            return detections
            
        except Exception as main_error:
            print(f"Main localization error: {main_error}")
            print("Please check:")
            print("   - Image file is valid and readable")
            print("   - Query text is not empty")
            print("   - System has enough memory")
            import traceback
            traceback.print_exc()
            return []
    
    def basic_sliding_window_fallback(self, image, query_list):
        """Simple fallback detection method"""
        print("Using basic fallback detection...")
        
        img_array = np.array(image)
        h, w = img_array.shape[:2]

        window_size = min(w, h) // 3
        stride = window_size // 2
        
        best_detection = None
        best_score = 0

        text_inputs = self.processor(text=query_list, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        for y in range(0, h - window_size + 1, stride):
            for x in range(0, w - window_size + 1, stride):
                x2, y2 = x + window_size, y + window_size
                
                window_img = image.crop((x, y, x2, y2))
                
                image_inputs = self.processor(images=window_img, return_tensors="pt", padding=True)
                image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
                
                with torch.no_grad():
                    image_features = self.model.get_image_features(**image_inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    similarity = torch.cosine_similarity(text_features, image_features, dim=-1)
                    score = similarity.max().item()
                    
                    if score > best_score:
                        best_score = score
                        best_detection = {
                            'bbox': (x, y, x2, y2),
                            'score': score,
                            'window_size': (window_size, window_size),
                            'query_used': query_list[0]
                        }
        
        return [best_detection] if best_detection and best_score > 0.1 else []

def main():
    """Main function"""
    print("\nScene Localization System\n")
    print("Featuring:")
    print("   - Smart query expansion")
    print("   - Adaptive window sizing")
    print("   - Confidence-based detection")
    print("   - Professional visualization\n")
    
    localizer = ImprovedSceneLocalizer()

    uploaded_files = []
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp', '*.gif']
    
    for ext in image_extensions:
        uploaded_files.extend(glob.glob(ext))
        uploaded_files.extend(glob.glob(ext.upper()))
    
    if not uploaded_files:
        print("No image files found in current directory!")
        print("Please upload or place image files in the same directory as this script.")
        print("Supported formats: JPG, PNG, BMP, TIFF, WebP, GIF")
        return
    
    print(f"Found {len(uploaded_files)} image(s):")
    for i, file in enumerate(uploaded_files, 1):
        size_mb = os.path.getsize(file) / (1024*1024)
        print(f"   {i}. {file} ({size_mb:.1f} MB)")

    if len(uploaded_files) == 1:
        image_path = uploaded_files[0]
        print(f"Using: {image_path}")
    else:
        try:
            choice = input(f"\nEnter image number (1-{len(uploaded_files)}) or press Enter for #1: ").strip()
            if choice:
                idx = int(choice) - 1
                if 0 <= idx < len(uploaded_files):
                    image_path = uploaded_files[idx]
                else:
                    image_path = uploaded_files[0]
            else:
                image_path = uploaded_files[0]
        except:
            image_path = uploaded_files[0]
        print(f"Selected: {image_path}")

    print(f"\nInteractive Query Testing")
    print(f"Enter your search queries to test the system")
    print(f"Examples:")
    print(f"   • 'person talking' - Find people in conversation")
    print(f"   • 'vendor selling goods' - Locate market vendors")
    print(f"   • 'red car' - Find red vehicles")
    print(f"   • 'dog playing' - Locate active dogs")
    
    while True:
        query = input("\nEnter query (or 'quit' to exit): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q', 'done']:
            break
        
        if not query:
            print("Please enter a search query!")
            continue
        
        try:
            detections = localizer.localize_scene(image_path, query)
            
            if detections:
                best_score = detections[0]['score']
                if best_score > 0.5:
                    print("Excellent detection quality!")
                elif best_score > 0.35:
                    print("Good detection found!")
                else:
                    print("Low confidence detection - consider rephrasing query")
            else:
                print("No detection found - try a different description")
                
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            print(f"Debug info:")
            print(f"  - Image path: {image_path}")
            print(f"  - Query: {query}")
            print(f"  - Query type: {type(query)}")
            import traceback
            traceback.print_exc()

        if input("\nContinue with another query? (y/n): ").strip().lower() in ['n', 'no']:
            break
    
    print(f"\nSession complete!")
    print(f"Check these files:")
    print(f"   - improved_result.jpg - Main visualization")
    print(f"   - improved_detections/ - Individual crops with metadata")
    print(f"\nThanks for using the Scene Localization System!")

if __name__ == "__main__":
    main()
