"""
FastAPI application for hand-drawn image recognition
Processes images to identify triangles, circles, and other shapes for game level creation
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import json
import io
from typing import List, Tuple, Dict, Any
import base64
from pydantic import BaseModel

app = FastAPI(
    title="Image Recognition API",
    description="API for recognizing hand-drawn shapes in images",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProcessImageRequest(BaseModel):
    """Request model for processing base64 encoded images"""
    image_base64: str
    debug: bool = False
    simplify_contours: bool = True
    max_vertices: int = 200
    simplification_factor: float = 0.1
    use_convex_decomposition: bool = False

class ShapeDetector:
    """Detects and classifies shapes in hand-drawn images"""

    def __init__(self, debug=False, simplify_contours=True, max_vertices=200, simplification_factor=0.1, use_convex_decomposition=False):
        self.debug = debug
        self.simplify_contours = simplify_contours
        self.max_vertices = max_vertices
        self.simplification_factor = simplification_factor
        self.use_convex_decomposition = use_convex_decomposition

    def preprocess_image(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the image for shape detection"""
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Morphological operations to clean up the image
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        return img, thresh

    def get_contours(self, thresh: np.ndarray) -> List[np.ndarray]:
        """Find and filter contours in the processed image"""
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area to remove noise
        min_area = 100
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        return filtered_contours

    def is_triangle(self, contour: np.ndarray) -> bool:
        """Check if contour is approximately a triangle"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area < 100 or perimeter < 30:
            return False

        # Try different epsilon values for approximation
        for epsilon_factor in [0.015, 0.02, 0.025, 0.03]:
            epsilon = epsilon_factor * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 3:
                points = approx.reshape(-1, 2)

                # Calculate side lengths
                side_lengths = []
                for i in range(3):
                    p1 = points[i]
                    p2 = points[(i + 1) % 3]
                    length = np.linalg.norm(p1 - p2)
                    side_lengths.append(length)

                # Check basic triangle properties
                min_side = min(side_lengths)
                max_side = max(side_lengths)

                if min_side < 10 or max_side > 500:
                    continue

                # Check triangle inequality
                side_lengths.sort()
                if side_lengths[0] + side_lengths[1] <= side_lengths[2] * 1.1:
                    continue

                # Calculate angles
                angles = []
                for i in range(3):
                    p1 = points[i]
                    p2 = points[(i + 1) % 3]
                    p3 = points[(i + 2) % 3]

                    v1 = p1 - p2
                    v2 = p3 - p2

                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle) * 180 / np.pi
                    angles.append(angle)

                # Check if angles are reasonable
                if all(20 <= angle <= 140 for angle in angles):
                    return True

        return False

    def is_circle_or_dot(self, contour: np.ndarray) -> bool:
        """Check if contour is approximately a circle"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area < 50 or perimeter < 20:
            return False

        # Calculate circularity
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.6:
                return True

        return False

    def get_centroid(self, contour: np.ndarray) -> Tuple[int, int]:
        """Calculate the centroid of a contour"""
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        return (cx, cy)

    def simplify_contour(self, contour: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Simplify contour for faster physics simulation and rendering"""
        original_vertices = len(contour)
        simplified_contour = contour
        simplification_info = {
            "original_vertices": original_vertices,
            "simplified_vertices": original_vertices,
            "method": "none",
            "is_convex": False,
            "convex_pieces": []
        }

        if not self.simplify_contours:
            return simplified_contour, simplification_info

        # Use approxPolyDP to reduce vertices
        perimeter = cv2.arcLength(contour, True)
        best_epsilon = self.simplification_factor * perimeter
        best_approx = contour

        for epsilon_multiplier in [0.01, 0.05, 0.1, 1.0, 2.0, 4.0]:
            epsilon = best_epsilon * epsilon_multiplier
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) <= self.max_vertices:
                best_approx = approx
                simplification_info["method"] = "approxPolyDP"
                break

        simplified_contour = best_approx
        simplification_info["simplified_vertices"] = len(simplified_contour)

        # Check if simplified contour is convex
        if len(simplified_contour) >= 3:
            hull = cv2.convexHull(simplified_contour)
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(simplified_contour)

            if hull_area > 0 and contour_area / hull_area > 0.95:
                simplification_info["is_convex"] = True
                simplified_contour = hull
                simplification_info["simplified_vertices"] = len(simplified_contour)
                simplification_info["method"] = "convex_hull"

        return simplified_contour, simplification_info

    def process_image(self, img: np.ndarray) -> Dict[str, Any]:
        """Main processing function to detect all shapes and classify them"""
        # Preprocess image
        original_img, processed_img = self.preprocess_image(img)

        # Get contours
        contours = self.get_contours(processed_img)

        # Scale factor
        scale_factor = 0.3

        # Initialize results
        results = {
            "starting_points": [],
            "end_points": [],
            "rigid_bodies": [],
            "image_size": original_img.shape[:2],
            "scale_factor": scale_factor
        }

        # Classify each contour
        for i, contour in enumerate(contours):
            centroid = self.get_centroid(contour)
            contour_area = cv2.contourArea(contour)

            # Apply scaling factor
            scaled_centroid = (int(centroid[0] * scale_factor), int(centroid[1] * scale_factor))

            if self.is_triangle(contour):
                results["starting_points"].append({
                    "coordinates": scaled_centroid,
                    "area": contour_area,
                    "contour_id": i,
                    "shape_type": "triangle"
                })
            elif self.is_circle_or_dot(contour):
                results["end_points"].append({
                    "coordinates": scaled_centroid,
                    "area": contour_area,
                    "contour_id": i,
                    "shape_type": "circle"
                })
            else:
                # Everything else is a rigid body
                simplified_contour, simplification_info = self.simplify_contour(contour)

                x, y, w, h = cv2.boundingRect(simplified_contour)
                scaled_bounding_box = (
                    int(x * scale_factor),
                    int(y * scale_factor),
                    int(w * scale_factor),
                    int(h * scale_factor)
                )

                # Scale contour points
                scaled_contour_points = []
                for point in simplified_contour:
                    scaled_point = point[0] * scale_factor
                    scaled_contour_points.append([int(scaled_point[0]), int(scaled_point[1])])

                results["rigid_bodies"].append({
                    "bounding_box": scaled_bounding_box,
                    "centroid": scaled_centroid,
                    "area": contour_area,
                    "contour_id": i,
                    "contour_points": scaled_contour_points,
                    "simplification_info": simplification_info
                })

        return results

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Image Recognition API",
        "endpoints": {
            "/process_image": "POST - Process an uploaded image file",
            "/process_base64": "POST - Process a base64 encoded image",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/process_image")
async def process_image_file(
    file: UploadFile = File(...),
    debug: bool = False,
    simplify_contours: bool = True,
    max_vertices: int = 200,
    simplification_factor: float = 0.1
):
    """Process an uploaded image file"""
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Process image
        detector = ShapeDetector(
            debug=debug,
            simplify_contours=simplify_contours,
            max_vertices=max_vertices,
            simplification_factor=simplification_factor,
            use_convex_decomposition=False
        )

        results = detector.process_image(img)

        return JSONResponse(content=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_base64")
async def process_base64_image(request: ProcessImageRequest):
    """Process a base64 encoded image"""
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")

        # Process image
        detector = ShapeDetector(
            debug=request.debug,
            simplify_contours=request.simplify_contours,
            max_vertices=request.max_vertices,
            simplification_factor=request.simplification_factor,
            use_convex_decomposition=request.use_convex_decomposition
        )

        results = detector.process_image(img)

        return JSONResponse(content=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)