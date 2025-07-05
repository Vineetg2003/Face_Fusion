# streamlit_app/utils.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from faceswap import fine_face_swap
from insightface.app import FaceAnalysis
import insightface

def run_face_swap(img1_path, img2_path, enhance=True, device='cpu'):
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    model_output_path = 'model/FaceFusion-SoC.onnx'
    if not os.path.exists(model_output_path):
        raise FileNotFoundError("FaceFusion ONNX model not found in 'model/' folder.")

    swapper = insightface.model_zoo.get_model(model_output_path, download=False, download_zip=False)
    
    result, _ = fine_face_swap(
        img1_fn=img1_path,
        img2_fn=img2_path,
        app=app,
        swapper=swapper,
        enhance=enhance,
        enhancer='REAL-ESRGAN 2x',
        device=device
    )
    return result
