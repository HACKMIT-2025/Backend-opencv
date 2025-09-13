"""
Modal deployment for Image Recognition API
Deploy with: modal deploy modal_app.py
"""

import modal
from modal import Image, Stub, web_endpoint, Mount
import json

# Create Modal stub
stub = Stub("image-recognition-api")

# Define the Docker image with dependencies
image = (
    Image.debian_slim()
    .apt_install("libglib2.0-0", "libsm6", "libxext6", "libxrender-dev", "libgomp1", "libgthread-2.0-0")
    .pip_install(
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "opencv-python==4.8.1.78",
        "numpy==1.24.3",
        "python-multipart==0.0.6",
        "pydantic==2.5.0"
    )
)

# Mount the local code
stub = Stub(
    "image-recognition-api",
    image=image,
    mounts=[Mount.from_local_file("main.py", remote_path="/app/main.py")]
)

@stub.function(gpu=None, container_idle_timeout=300)
@web_endpoint(method="GET")
def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "image-recognition-api"}

@stub.function(gpu=None, container_idle_timeout=300)
@web_endpoint(method="POST")
def process_base64(data: dict):
    """Process a base64 encoded image"""
    import cv2
    import numpy as np
    import base64
    import sys
    sys.path.append("/app")
    from main import ShapeDetector

    try:
        # Extract parameters
        image_base64 = data.get("image_base64", "")
        debug = data.get("debug", False)
        simplify_contours = data.get("simplify_contours", True)
        max_vertices = data.get("max_vertices", 200)
        simplification_factor = data.get("simplification_factor", 0.1)
        use_convex_decomposition = data.get("use_convex_decomposition", False)

        if not image_base64:
            return {"error": "No image data provided"}, 400

        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid base64 image data"}, 400

        # Process image
        detector = ShapeDetector(
            debug=debug,
            simplify_contours=simplify_contours,
            max_vertices=max_vertices,
            simplification_factor=simplification_factor,
            use_convex_decomposition=use_convex_decomposition
        )

        results = detector.process_image(img)

        return results

    except Exception as e:
        return {"error": str(e)}, 500

@stub.function(gpu=None, container_idle_timeout=300)
@web_endpoint(method="POST")
def process_image(file_data: bytes, params: dict = None):
    """Process an uploaded image file"""
    import cv2
    import numpy as np
    import sys
    sys.path.append("/app")
    from main import ShapeDetector

    try:
        # Default parameters
        if params is None:
            params = {}

        debug = params.get("debug", False)
        simplify_contours = params.get("simplify_contours", True)
        max_vertices = params.get("max_vertices", 200)
        simplification_factor = params.get("simplification_factor", 0.1)

        # Decode image
        nparr = np.frombuffer(file_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image file"}, 400

        # Process image
        detector = ShapeDetector(
            debug=debug,
            simplify_contours=simplify_contours,
            max_vertices=max_vertices,
            simplification_factor=simplification_factor,
            use_convex_decomposition=False
        )

        results = detector.process_image(img)

        return results

    except Exception as e:
        return {"error": str(e)}, 500

@stub.function(gpu=None, container_idle_timeout=300)
@web_endpoint(method="GET")
def info():
    """API information endpoint"""
    return {
        "name": "Image Recognition API",
        "version": "1.0.0",
        "description": "Hand-drawn shape recognition for game level creation",
        "endpoints": {
            "/health": "GET - Health check",
            "/info": "GET - API information",
            "/process_base64": "POST - Process base64 encoded image",
            "/process_image": "POST - Process uploaded image file"
        },
        "supported_shapes": [
            "Triangles (starting points)",
            "Circles (end points)",
            "Other shapes (rigid bodies/walls)"
        ]
    }

# ASGI app for Modal serving
@stub.function(gpu=None)
@modal.asgi_app()
def fastapi_app():
    """Main FastAPI application for Modal"""
    import sys
    sys.path.append("/app")
    from main import app
    return app