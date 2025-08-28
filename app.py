import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import json
import pandas as pd
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Import custom modules
from acne_detector import AcneDetector
from chatbot import AcneChatbot
from knowledge_base import KnowledgeBase
from webcam_capture import streamlit_webcam_interface, camera_settings_interface

# Page configuration
st.set_page_config(
    page_title="Acne Detection & Treatment Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #2E86AB;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #A23B72;
    margin-bottom: 1rem;
}
.detection-box {
    border: 2px solid #2E86AB;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
}
.chat-container {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'acne_detector' not in st.session_state:
        st.session_state.acne_detector = AcneDetector(model_path='best.pt')
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = AcneChatbot()
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = KnowledgeBase()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'detected_acne' not in st.session_state:
        st.session_state.detected_acne = None

def main():
    """Main application function"""
    initialize_session_state()
    
    # Main header
    st.markdown('<h1 class="main-header">🔬 Real-Time Acne Detection & Treatment Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a feature:",
        ["Acne Detection", "Treatment Chatbot", "About"]
    )
    
    if page == "Acne Detection":
        acne_detection_page()
    elif page == "Treatment Chatbot":
        chatbot_page()
    elif page == "About":
        about_page()

def acne_detection_page():
    """Acne detection page"""
    st.markdown('<h2 class="sub-header">📸 Acne Detection</h2>', unsafe_allow_html=True)
    
    # Image input options
    input_option = st.radio(
        "Choose input method:",
        ["📁 Upload Image", "📷 Use Camera"]
    )
    
    if input_option == "📁 Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear facial image for acne detection"
        )
        if uploaded_file is not None:
            process_uploaded_image(uploaded_file)
    else:
        process_camera_image()

def process_uploaded_image(uploaded_file):
    """Process uploaded image for acne detection"""
    # Load and display image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.subheader("Detection Results")
        
        with st.spinner("Detecting acne..."):
            # Detect acne
            result_image, detections = st.session_state.acne_detector.detect_acne(img_array)
            
            # Display results
            st.image(result_image, caption="Acne Detection Results", use_container_width=True)
            
            # Store detection results
            st.session_state.detected_acne = detections
            
            # Display detection statistics
            display_detection_stats(detections)

def process_camera_image():
    """Process camera captured image"""
    st.subheader("📷 Camera Capture")
    
    # Camera settings
    with st.expander("⚙️ Camera Settings"):
        camera_settings_interface()
    
    # Webcam interface
    streamlit_webcam_interface()
    
    # Check if image was captured and stored in session state
    if 'image_for_analysis' in st.session_state:
        captured_image = st.session_state.image_for_analysis
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Captured Image")
            st.image(captured_image, caption="Camera Capture", use_container_width=True)
        
        with col2:
            st.subheader("Detection Results")
            
            with st.spinner("Detecting acne..."):
                # Detect acne
                result_image, detections = st.session_state.acne_detector.detect_acne(captured_image)
                
                # Display results
                st.image(result_image, caption="Acne Detection Results", use_container_width=True)
                
                # Store detection results
                st.session_state.detected_acne = detections
                
                # Display detection statistics
                display_detection_stats(detections)
                
                # Clear the analysis image after processing
                del st.session_state.image_for_analysis

def display_detection_stats(detections):
    """Display acne detection statistics"""
    if detections:
        st.markdown('<div class="detection-box">', unsafe_allow_html=True)
        st.success(f"✅ Detected {len(detections)} acne spot(s)")
        
        # Severity analysis
        severity_counts = {"mild": 0, "moderate": 0, "severe": 0}
        for detection in detections:
            confidence = detection.get('confidence', 0)
            if confidence < 0.5:
                severity_counts["mild"] += 1
            elif confidence < 0.8:
                severity_counts["moderate"] += 1
            else:
                severity_counts["severe"] += 1
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mild", severity_counts["mild"])
        with col2:
            st.metric("Moderate", severity_counts["moderate"])
        with col3:
            st.metric("Severe", severity_counts["severe"])
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Suggest chatbot consultation
        if st.button("💬 Get Treatment Advice"):
            st.info("💡 Navigate to the 'Treatment Chatbot' page in the sidebar to get personalized treatment advice based on your detection results!")
    else:
        st.info("No acne detected in the image.")

def chatbot_page():
    """Treatment chatbot page"""
    st.markdown('<h2 class="sub-header">💬 Treatment Chatbot</h2>', unsafe_allow_html=True)
    
    # Display chat history
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    if st.session_state.chat_history:
        for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
            st.markdown(f"**You:** {user_msg}")
            st.markdown(f"**Assistant:** {bot_msg}")
            st.markdown("---")
    else:
        st.info("👋 Hello! I'm your acne treatment assistant. Ask me anything about acne care and treatment!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    user_input = st.text_input(
        "Ask about acne treatment:",
        placeholder="e.g., What's the best treatment for moderate acne?",
        key="chat_input"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        send_button = st.button("Send 📤")
    
    if send_button and user_input:
        process_chat_message(user_input)
    
    # Quick questions
    st.subheader("Quick Questions")
    quick_questions = [
        "What causes acne?",
        "How to prevent acne?",
        "Best skincare routine for acne?",
        "When to see a dermatologist?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(quick_questions):
        with cols[i % 2]:
            if st.button(question, key=f"quick_{i}"):
                process_chat_message(question)

def process_chat_message(user_input):
    """Process user chat message and generate response"""
    try:
        with st.spinner("Generating response..."):
            # Get response from chatbot
            response = st.session_state.chatbot.get_response(
                user_input, 
                st.session_state.detected_acne
            )
            
            # Add to chat history
            st.session_state.chat_history.append((user_input, response))
            
            # Rerun to update chat display
            st.rerun()
            
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")

def about_page():
    """About page with project information"""
    st.markdown('<h2 class="sub-header">ℹ️ About This Application</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Project Overview
    This Real-Time Acne Detection & Treatment Assistant combines computer vision and natural language processing 
    to provide comprehensive acne care solutions.
    
    ### 🔧 Key Features
    - **Real-time Acne Detection**: Upload images or use webcam for instant acne detection
    - **AI-Powered Chatbot**: Get personalized treatment advice using advanced NLP models
    - **Knowledge Base**: Access curated acne treatment information with RAG technology
    - **User-Friendly Interface**: Simple and intuitive Streamlit-based web application
    
    ### 🛠️ Technologies Used
    - **Frontend & Backend**: Streamlit
    - **Computer Vision**: YOLOv5, OpenCV
    - **NLP**: Hugging Face Transformers, GPT-2
    - **RAG System**: Sentence Transformers, FAISS
    - **Knowledge Base**: Local JSON/CSV storage
    
    ### 📋 How to Use
    1. **Acne Detection**: Upload a facial image or use the camera to detect acne spots
    2. **Treatment Advice**: Chat with the AI assistant for personalized treatment recommendations
    3. **Follow Recommendations**: Implement suggested treatments and track your progress
    
    ### ⚠️ Disclaimer
    This application is for educational and informational purposes only. Always consult with a 
    qualified dermatologist for professional medical advice.
    """)
    
    # System status
    st.subheader("🔍 System Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        detector_status = "✅ Ready" if hasattr(st.session_state, 'acne_detector') else "❌ Not Loaded"
        st.metric("Acne Detector", detector_status)
    
    with col2:
        chatbot_status = "✅ Ready" if hasattr(st.session_state, 'chatbot') else "❌ Not Loaded"
        st.metric("Chatbot", chatbot_status)
    
    with col3:
        kb_status = "✅ Ready" if hasattr(st.session_state, 'knowledge_base') else "❌ Not Loaded"
        st.metric("Knowledge Base", kb_status)

if __name__ == "__main__":
    main()