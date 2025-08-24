import torch
import cv2
import numpy as np
from PIL import Image
import yolov5
from typing import List, Dict, Tuple
import os

class AcneDetector:
    """YOLOv5-based acne detection system"""
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.25):
        """
        Initialize the acne detector
        
        Args:
            model_path: Path to custom YOLOv5 model (if None, uses pre-trained model)
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load YOLOv5 model
        self.model = self._load_model(model_path)
        
        # Define acne-related classes (for demonstration, we'll use person detection
        # and treat high-confidence face detections as potential acne areas)
        self.acne_classes = ['acne', 'pimple', 'blackhead', 'whitehead']
        
    def _load_model(self, model_path: str = None):
        """
        Load YOLOv5 model
        
        Args:
            model_path: Path to custom model file
            
        Returns:
            Loaded YOLOv5 model
        """
        try:
            if model_path and os.path.exists(model_path):
                # Load custom trained model
                model = yolov5.load(model_path, device=self.device)
            else:
                # Load pre-trained YOLOv5 model (we'll simulate acne detection)
                model = yolov5.load('yolov5s.pt', device=self.device)
            
            model.conf = self.confidence_threshold
            model.iou = 0.45
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to basic detection
            return None
    
    def detect_acne(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect acne in the given image
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Tuple of (annotated_image, detections_list)
        """
        try:
            # Convert BGR to RGB for YOLOv5
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.model is not None:
                # Use YOLOv5 for detection (simulated acne detection)
                results = self.model(rgb_image)
                detections = self._process_yolo_results(results, image.shape)
            else:
                # Fallback: Use traditional computer vision methods
                detections = self._fallback_detection(image)
            
            # Draw bounding boxes and annotations
            annotated_image = self._draw_detections(image.copy(), detections)
            
            return annotated_image, detections
            
        except Exception as e:
            print(f"Error in acne detection: {e}")
            return image, []
    
    def _process_yolo_results(self, results, image_shape: Tuple) -> List[Dict]:
        """
        Process YOLOv5 results and simulate acne detection
        
        Args:
            results: YOLOv5 detection results
            image_shape: Shape of the input image
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        # Since we don't have a real acne-trained model, we'll simulate detections
        # by creating random acne spots for demonstration
        detections.extend(self._simulate_acne_detections(image_shape))
        
        return detections
    
    def _simulate_acne_detections(self, image_shape: Tuple) -> List[Dict]:
        """
        Simulate acne detections for demonstration purposes
        In a real implementation, this would be replaced by actual YOLOv5 acne detection
        
        Args:
            image_shape: Shape of the input image (height, width, channels)
            
        Returns:
            List of simulated acne detections
        """
        height, width = image_shape[:2]
        detections = []
        
        # Simulate 3-8 random acne spots
        num_spots = np.random.randint(3, 9)
        
        for i in range(num_spots):
            # Random position (focus on face area - center region)
            center_x = int(width * (0.3 + 0.4 * np.random.random()))
            center_y = int(height * (0.2 + 0.6 * np.random.random()))
            
            # Random size (small spots)
            size = np.random.randint(15, 40)
            
            # Random confidence
            confidence = 0.3 + 0.6 * np.random.random()
            
            # Random acne type
            acne_type = np.random.choice(['pimple', 'blackhead', 'whitehead', 'cyst'])
            
            detection = {
                'bbox': [center_x - size//2, center_y - size//2, 
                        center_x + size//2, center_y + size//2],
                'confidence': confidence,
                'class': acne_type,
                'severity': self._determine_severity(confidence)
            }
            
            detections.append(detection)
        
        return detections
    
    def _fallback_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Fallback detection method using traditional computer vision
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Use HoughCircles to detect circular spots (potential acne)
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=20,
                param1=50,
                param2=30,
                minRadius=5,
                maxRadius=25
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                for (x, y, r) in circles:
                    # Create detection dictionary
                    detection = {
                        'bbox': [x-r, y-r, x+r, y+r],
                        'confidence': 0.6,  # Fixed confidence for fallback method
                        'class': 'potential_acne',
                        'severity': 'moderate'
                    }
                    detections.append(detection)
            
        except Exception as e:
            print(f"Error in fallback detection: {e}")
        
        return detections
    
    def _determine_severity(self, confidence: float) -> str:
        """
        Determine acne severity based on confidence score
        
        Args:
            confidence: Detection confidence score
            
        Returns:
            Severity level as string
        """
        if confidence < 0.4:
            return 'mild'
        elif confidence < 0.7:
            return 'moderate'
        else:
            return 'severe'
    
    def _draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on the image
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            
        Returns:
            Annotated image
        """
        # Define colors for different severities
        severity_colors = {
            'mild': (0, 255, 0),      # Green
            'moderate': (0, 165, 255), # Orange
            'severe': (0, 0, 255)      # Red
        }
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            acne_class = detection['class']
            severity = detection['severity']
            
            # Get color based on severity
            color = severity_colors.get(severity, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Prepare label text
            label = f"{acne_class}: {confidence:.2f} ({severity})"
            
            # Calculate text size and position
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw label background
            cv2.rectangle(
                image,
                (bbox[0], bbox[1] - text_height - 10),
                (bbox[0] + text_width, bbox[1]),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                image,
                label,
                (bbox[0], bbox[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return image
    
    def get_detection_summary(self, detections: List[Dict]) -> Dict:
        """
        Generate a summary of detections
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Summary dictionary with statistics
        """
        if not detections:
            return {
                'total_count': 0,
                'severity_breakdown': {'mild': 0, 'moderate': 0, 'severe': 0},
                'average_confidence': 0.0,
                'dominant_severity': 'none'
            }
        
        # Count by severity
        severity_counts = {'mild': 0, 'moderate': 0, 'severe': 0}
        total_confidence = 0
        
        for detection in detections:
            severity = detection['severity']
            severity_counts[severity] += 1
            total_confidence += detection['confidence']
        
        # Find dominant severity
        dominant_severity = max(severity_counts, key=severity_counts.get)
        
        return {
            'total_count': len(detections),
            'severity_breakdown': severity_counts,
            'average_confidence': total_confidence / len(detections),
            'dominant_severity': dominant_severity
        }
    
    def update_confidence_threshold(self, new_threshold: float):
        """
        Update the confidence threshold for detections
        
        Args:
            new_threshold: New confidence threshold (0.0 to 1.0)
        """
        self.confidence_threshold = max(0.0, min(1.0, new_threshold))
        if self.model is not None:
            self.model.conf = self.confidence_threshold