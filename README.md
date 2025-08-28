# Real-Time Acne Detection & Treatment Assistant

🔬 An interactive web application that combines computer vision and natural language processing to provide comprehensive acne detection and personalized treatment recommendations.

## 🎯 Project Overview

This application uses YOLOv5 for real-time acne detection in facial images and provides personalized treatment advice through an AI-powered chatbot. The system integrates Retrieval-Augmented Generation (RAG) with a comprehensive knowledge base to deliver accurate, evidence-based treatment recommendations.

## ✨ Key Features

### 🔍 Real-Time Acne Detection

- Upload facial images or use webcam for live capture
- YOLOv5-based acne detection with bounding box visualization
- Severity classification (mild, moderate, severe)
- Detection statistics and analysis

### 💬 AI-Powered Treatment Chatbot

- Personalized treatment recommendations
- RAG-enhanced responses using local knowledge base
- Context-aware conversations based on detection results
- Support for various acne-related queries

### 📚 Comprehensive Knowledge Base

- Curated acne treatment information
- Skincare routine recommendations
- Lifestyle tips and prevention strategies
- Myths vs. facts about acne

### 🖥️ User-Friendly Interface

- Clean, modern Streamlit-based web interface
- Responsive design with intuitive navigation
- Real-time webcam integration
- Interactive chat interface

## 🛠️ Technologies Used

### Frontend & Backend

- **Streamlit**: Web application framework
- **OpenCV**: Image processing and webcam integration
- **Pillow**: Image manipulation

### Computer Vision

- **YOLOv5**: Object detection for acne identification
- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computations

### Natural Language Processing

- **Hugging Face Transformers**: Pre-trained language models
- **Sentence Transformers**: Semantic embeddings
- **scikit-learn**: TF-IDF vectorization and similarity

### Knowledge Management

- **JSON**: Local knowledge base storage
- **Pandas**: Data manipulation
- **FAISS**: Efficient similarity search (optional)

## 📋 Prerequisites

- Python 3.8 or higher
- Webcam (optional, for real-time capture)
- Internet connection (for downloading models)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd acne-detection-assistant
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Required Models

The application will automatically download required models on first run:

- YOLOv5 model weights
- Sentence transformer models
- Hugging Face language models

## 🎮 Usage

### Starting the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using Acne Detection

1. Navigate to the "Acne Detection" page
2. Upload an image or use the webcam
3. View detection results with severity analysis
4. Get treatment recommendations based on results

### Chatbot Interaction

1. Go to the "Treatment Chatbot" page
2. Ask questions about acne treatment
3. Get personalized recommendations
4. Use quick question buttons for common queries

### Webcam Features

1. Initialize your camera
2. Start live preview
3. Capture photos when ready
4. Analyze captured images for acne

## 📁 Project Structure

```
acne-detection-assistant/
├── app.py                      # Main Streamlit application
├── acne_detector.py           # YOLOv5 acne detection module
├── chatbot.py                 # AI chatbot with RAG
├── knowledge_base.py          # Knowledge base management
├── webcam_capture.py          # Webcam integration
├── acne_knowledge_base.json   # Treatment knowledge base
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 🔧 Configuration

### Model Settings

- **Confidence Threshold**: Adjust detection sensitivity in `acne_detector.py`
- **Search Method**: Configure RAG search method in `knowledge_base.py`
- **Response Style**: Modify chatbot responses in `chatbot.py`

### Camera Settings

- **Resolution**: Default 640x480, configurable in webcam interface
- **FPS**: Default 30 FPS, adjustable in settings
- **Device Index**: Auto-detection with manual override option

## 📊 Knowledge Base

The application includes a comprehensive knowledge base with:

### Treatment Options

- Topical treatments (Benzoyl Peroxide, Salicylic Acid, Retinoids)
- Oral medications (Antibiotics, Isotretinoin)
- Professional treatments (Chemical Peels, Light Therapy)
- Hormonal therapies

### Skincare Routines

- Morning and evening routines
- Product recommendations
- Application tips and precautions

### Lifestyle Factors

- Diet and acne relationship
- Stress management
- Exercise and hygiene tips
- Common myths debunked

## 🤖 AI Models

### Acne Detection

- **Base Model**: YOLOv5s (small variant for speed)
- **Custom Training**: Simulated acne detection for demonstration
- **Fallback**: Traditional computer vision methods

### Chatbot

- **Language Model**: DialoGPT-medium for conversations
- **Embeddings**: all-MiniLM-L6-v2 for semantic search
- **RAG System**: Hybrid TF-IDF and semantic similarity

## 🔒 Privacy & Security

- **Local Processing**: All image analysis happens locally
- **No Data Storage**: Images are not permanently stored
- **Privacy First**: No personal data is transmitted externally
- **Secure Models**: Only trusted, open-source models used

## ⚠️ Important Disclaimers

- **Educational Purpose**: This application is for educational and informational purposes only
- **Not Medical Advice**: Always consult qualified dermatologists for professional medical advice
- **Accuracy Limitations**: AI detection may not be 100% accurate
- **Individual Variation**: Treatment effectiveness varies between individuals

## 🐛 Troubleshooting

### Common Issues

**Camera Not Working**

- Check camera permissions in browser
- Ensure no other applications are using the camera
- Try different camera indices in settings

**Model Loading Errors**

- Check internet connection for initial model download
- Verify sufficient disk space (models ~500MB total)
- Try restarting the application

**Performance Issues**

- Reduce image resolution for faster processing
- Close other resource-intensive applications
- Consider using CPU-only mode if GPU causes issues

**Installation Problems**

- Update pip: `pip install --upgrade pip`
- Install Visual C++ Build Tools (Windows)
- Use conda instead of pip for complex dependencies

### Error Messages

**"No module named 'torch'"**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**"Camera initialization failed"**

- Check camera drivers and connections
- Try running as administrator (Windows)
- Test camera with other applications first

## 🔄 Updates & Maintenance

### Updating Dependencies

```bash
pip install --upgrade -r requirements.txt
```

### Model Updates

- Models are cached locally after first download
- Clear cache to force re-download: delete `~/.cache/torch/hub/`
- Update model versions in code as needed

### Knowledge Base Updates

- Edit `acne_knowledge_base.json` to add new treatments
- Restart application to reload knowledge base
- Validate JSON format after manual edits

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- **Model Training**: Train custom acne detection models
- **UI/UX**: Enhance user interface design
- **Knowledge Base**: Add more treatment information
- **Performance**: Optimize processing speed
- **Features**: Add new functionality

### Development Setup

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Submit pull request with detailed description

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **YOLOv5**: Ultralytics for the object detection framework
- **Hugging Face**: For pre-trained language models
- **Streamlit**: For the excellent web framework
- **OpenCV**: For computer vision capabilities
- **Medical Community**: For acne treatment research and guidelines

## 📞 Support

For questions, issues, or suggestions:

1. Check the troubleshooting section above
2. Search existing issues in the repository
3. Create a new issue with detailed description
4. Include system information and error messages

## 🔮 Future Enhancements

- **Mobile App**: React Native or Flutter version
- **Cloud Deployment**: Heroku/AWS deployment options
- **Advanced Models**: Custom-trained acne detection models
- **Progress Tracking**: Before/after comparison features
- **Dermatologist Integration**: Professional consultation booking
- **Multi-language Support**: Internationalization
- **API Development**: REST API for third-party integration

---

**Remember**: This tool is designed to supplement, not replace, professional medical advice. Always consult with qualified healthcare providers for serious skin conditions.
