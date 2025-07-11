# ReFaceX: AI-Powered Face Swapping & Enhancement Tool

![ReFaceX Banner](https://via.placeholder.com/1200x400) <!-- Replace with your project banner -->

ReFaceX is a powerful AI-driven tool that allows you to seamlessly swap faces and enhance image quality with just one click! It leverages the latest advancements in face enhancement and swapping technologies, providing professional-level results in seconds. Whether you're working in film, advertising, or content creation, ReFaceX delivers both precision and speed.

## 📌 Examples
| Input | Enhanced | Swapped |
|-------|----------|---------|
| ![Input](images/Image5.avif) | ![Enhanced](https://via.placeholder.com/300) | ![Swapped](https://via.placeholder.com/300) |

## ✨ Features

- **🎭 AI-Powered Face Enhancement**  
  Using state-of-the-art models like GFPGAN and Real-ESRGAN, ReFaceX significantly improves the quality of images, restoring fine facial details.

- **🔄 Realistic Face Swapping**  
  Powered by InsightFace, our face-swapping technology is precise and natural, giving lifelike results that are indistinguishable from reality.

- **⚡ Fast & Efficient**  
  Optimized for speed without sacrificing accuracy, making it easy for users to generate high-quality results in minimal time.

- **💼 Professional-Grade Output**  
  Tailored for professionals in film, TV, advertising, and design, but also accessible for casual users.

## 🛠️ Tech Stack

- **GFPGAN & Real-ESRGAN** – For facial enhancement and image super-resolution.
- **InsightFace** – For accurate face detection, alignment, and swapping.
- **Python** – Core programming language for AI model integration.
- **Flask** – Backend framework for handling model operations.
- **Streamlit** – For the interactive web interface (optional).

## 📂 Directory Structure

```
└── ReFaceX/
    ├── face_enhancer.py        # Face enhancement module
    ├── faceswap.py             # Face swapping module
    ├── main.py                 # Main application script
    ├── requirements.txt        # Python dependencies
    ├── images/                 # Sample/test images
    │   └── Image5.avif
    ├── model/                  # Pretrained models
    │   ├── ReFaceX-SoC.onnx
    │   ├── PreTrainedRealESRGAN_x2.pth
    │   └── RealESRGAN_x4.pth
    ├── streamlit_app/          # Streamlit web app (optional)
    │   ├── __init__.py
    │   ├── app.py
    │   ├── utils.py
    │   └── styles/
    │       └── custom.css
    └── upscaler/               # RealESRGAN upscaling module
        ├── __init__.py
        └── RealESRGAN/
            ├── __init__.py
            ├── arch_utils.py
            ├── model.py
            ├── rrdbnet_arch.py
            └── utils.py
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA (for GPU acceleration, optional)
- pip

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/vineetg2003/ReFaceX.git
   cd ReFaceX
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download required models and place them in the `model/` directory.

### Usage
#### Command Line Interface (CLI)
Run the main script:
```bash
python main.py --input input.jpg --output output.jpg --mode [enhance|swap]
```

#### Web Interface (Streamlit)
If using the Streamlit app:
```bash
streamlit run streamlit_app/app.py
```

## 📄 License
This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## 📧 Contact
For inquiries, contact:  
📩 vineetg2003@example.com  
🌐 [https://github.com/vineetg2003](https://github.com/vineetg2003)


