import cv2
import streamlit as st
import numpy as np
from PIL import Image
import time
from typing import Optional, Tuple

class WebcamCapture:
    """Simple webcam capture with live preview for acne detection"""
    
    def __init__(self):
        """
        Initialize webcam capture
        """
        self.camera = None
        self.is_active = False
        
    def initialize_camera(self, camera_index: int = 0) -> bool:
        """
        Initialize camera connection
        
        Args:
            camera_index: Camera device index (usually 0 for default camera)
            
        Returns:
            Success status
        """
        try:
            self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                return False
            
            # Set camera properties for better quality
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Test if camera can capture frames
            ret, frame = self.camera.read()
            if not ret or frame is None:
                self.camera.release()
                self.camera = None
                return False
            
            self.is_active = True
            return True
            
        except Exception as e:
            return False
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get current frame from camera
        
        Returns:
            Current frame or None if failed
        """
        if self.camera is None or not self.camera.isOpened():
            return None
        
        try:
            ret, frame = self.camera.read()
            if ret and frame is not None:
                return frame
            return None
        except Exception as e:
            return None
    
    def release_camera(self):
        """
        Release camera resources
        """
        self.is_active = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        cv2.destroyAllWindows()
    
    def is_camera_available(self) -> bool:
        """
        Check if camera is available and working
        
        Returns:
            Camera availability status
        """
        return self.camera is not None and self.camera.isOpened() and self.is_active

def streamlit_webcam_interface():
    """
    Streamlit interface for webcam functionality with live preview
    """
    st.subheader("📷 Live Webcam Feed")
    
    # Initialize webcam capture in session state
    if 'webcam_capture' not in st.session_state:
        st.session_state.webcam_capture = WebcamCapture()
    
    webcam = st.session_state.webcam_capture
    
    # Camera controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_button = st.button("🎥 Start Camera")
    
    with col2:
        capture_button = st.button("📸 Capture Photo")
    
    with col3:
        stop_button = st.button("⏹️ Stop Camera")
    
    # Handle button clicks
    if start_button:
        if webcam.initialize_camera():
            st.success("Camera started successfully!")
            st.rerun()
        else:
            st.error("Failed to start camera. Please check if camera is connected and not in use.")
    
    if stop_button:
        webcam.release_camera()
        st.success("Camera stopped.")
        st.rerun()
    
    # Live preview area
    if webcam.is_camera_available():
        # Create placeholder for live feed
        frame_placeholder = st.empty()
        
        # Continuous frame capture and display
        while webcam.is_camera_available():
            frame = webcam.get_frame()
            if frame is not None:
                # Convert BGR to RGB for Streamlit display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame, caption="Live Camera Feed", use_container_width=True, channels="RGB")
                
                # Handle capture button
                if capture_button:
                    # Store captured image in session state
                    st.session_state.captured_image = rgb_frame
                    st.session_state.image_for_analysis = rgb_frame
                    st.success("Photo captured! You can now analyze it for acne detection.")
                    
                    # Display captured image
                    st.subheader("📸 Captured Image")
                    st.image(rgb_frame, caption="Captured for Analysis", use_container_width=True)
                    break
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
            else:
                st.error("Failed to capture frame from camera.")
                break
    else:
        st.info("Click 'Start Camera' to begin live preview.")
        
        # Show placeholder image
        placeholder_image = np.zeros((480, 640, 3), dtype=np.uint8)
        placeholder_image.fill(128)  # Gray background
        st.image(placeholder_image, caption="Camera Preview (Not Active)", use_container_width=True)
    
    # Camera status
    st.subheader("📊 Camera Status")
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        camera_status = "🟢 Active" if webcam.is_camera_available() else "🔴 Inactive"
        st.metric("Camera", camera_status)
    
    with status_col2:
        if webcam.is_camera_available():
            st.metric("Resolution", "640x480")
        else:
            st.metric("Resolution", "N/A")
    
    # Instructions
    st.markdown("**📋 How to Use:**")
    st.markdown("""
    **Steps to capture and analyze:**
    
    1. **Start Camera**: Click to activate your webcam
    2. **Position yourself**: Make sure your face is clearly visible
    3. **Capture Photo**: Click when you're ready to take a photo
    4. **Analyze**: The captured image will be ready for acne detection
    
    **Tips for better results:**
    - Ensure good lighting
    - Keep your face centered in the frame
    - Avoid shadows on your face
    - Make sure the camera is stable
    """)

def camera_settings_interface():
    """
    Simple camera settings interface
    """
    # Resolution options
    resolution_options = {
        "640x480 (Standard)": (640, 480),
        "800x600 (SVGA)": (800, 600),
        "1280x720 (HD)": (1280, 720)
    }
    
    selected_resolution = st.selectbox(
        "Camera Resolution:",
        list(resolution_options.keys())
    )
    
    # Camera index selection
    camera_index = st.number_input(
        "Camera Index (0 for default):",
        min_value=0,
        max_value=3,
        value=0,
        help="Try different values if your camera doesn't work with 0"
    )
    
    if st.button("Apply Settings"):
        if 'webcam_capture' in st.session_state:
            webcam = st.session_state.webcam_capture
            webcam.release_camera()
            
            # Apply new settings
            if webcam.initialize_camera(camera_index):
                width, height = resolution_options[selected_resolution]
                webcam.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                webcam.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                st.success(f"Settings applied: {selected_resolution}")
            else:
                st.error("Failed to apply settings. Check camera connection.")
    
    # Troubleshooting
    st.markdown("**🔧 Troubleshooting:**")
    st.markdown("""
    - **Camera not detected**: Try different camera index values (0, 1, 2)
    - **Permission denied**: Allow camera access in your browser
    - **Camera in use**: Close other applications using the camera
    - **Poor quality**: Try different resolution settings
    - **Frozen feed**: Stop and restart the camera
    """)

# Utility functions
def process_webcam_image_for_detection(image: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Process webcam image for acne detection
    
    Args:
        image: Raw webcam image (RGB format)
        
    Returns:
        Tuple of (processed_image, success_status)
    """
    try:
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Apply slight enhancement for better detection
            processed_image = cv2.convertScaleAbs(image, alpha=1.1, beta=10)
            return processed_image, True
        return image, False
    except Exception as e:
        return image, False

def save_captured_image(image: np.ndarray, filename: str = None) -> str:
    """
    Save captured image to file
    
    Args:
        image: Image to save (RGB format)
        filename: Optional filename
        
    Returns:
        Saved filename
    """
    if filename is None:
        timestamp = int(time.time())
        filename = f"captured_image_{timestamp}.jpg"
    
    try:
        # Convert RGB to BGR for OpenCV
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, bgr_image)
        return filename
    except Exception as e:
        return None