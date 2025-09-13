#!/usr/bin/env python3
"""
Hand-drawn Image Recognition for Game Level Creation

This program processes hand-drawn images and identifies:
1. Triangles (almost equilateral) -> Starting points
2. Circles -> End points  
3. Lines/other shapes -> Walls/obstacles

Usage: python image_recognition.py <image_path> --output level_data.json
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from typing import List, Tuple, Dict, Any
try:
    import pymunk
    import pymunk.autogeometry
    PYMUNK_AVAILABLE = True
except ImportError:
    PYMUNK_AVAILABLE = False
    print("Warning: pymunk not available. Convex decomposition will be disabled.")


class ShapeDetector:
    """Detects and classifies shapes in hand-drawn images"""
    
    def __init__(self, debug=False, simplify_contours=True, max_vertices=200, simplification_factor=0.1, use_convex_decomposition=True):
        self.debug = debug
        self.simplify_contours = simplify_contours
        self.max_vertices = max_vertices
        self.simplification_factor = simplification_factor
        self.use_convex_decomposition = use_convex_decomposition and PYMUNK_AVAILABLE
        
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess the image for shape detection"""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive threshold to handle varying lighting
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations to clean up the image
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        if self.debug:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title('Original')
            plt.subplot(1, 3, 2)
            plt.imshow(gray, cmap='gray')
            plt.title('Grayscale')
            plt.subplot(1, 3, 3)
            plt.imshow(thresh, cmap='gray')
            plt.title('Processed')
            plt.show()
        
        return img, thresh
    
    def get_contours(self, thresh: np.ndarray) -> List[np.ndarray]:
        """Find and filter contours in the processed image"""
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area to remove noise
        min_area = 100  # Adjust based on image size
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        return filtered_contours
    
    def is_triangle(self, contour: np.ndarray) -> bool:
        """Check if contour is approximately a triangle - IMPROVED VERSION"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Skip very small shapes
        if area < 100 or perimeter < 30:
            return False
        
        # Try different epsilon values for approximation
        for epsilon_factor in [0.015, 0.02, 0.025, 0.03]:
            epsilon = epsilon_factor * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it has 3 vertices
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
                
                # Must have reasonable side lengths
                if min_side < 10 or max_side > 500:
                    continue
                
                # Check if it's not too elongated (triangle inequality roughly)
                side_lengths.sort()
                if side_lengths[0] + side_lengths[1] <= side_lengths[2] * 1.1:
                    continue
                
                # Calculate angles to ensure it's a reasonable triangle
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
                
                # Check if angles are reasonable (between 20 and 140 degrees)
                if all(20 <= angle <= 140 for angle in angles):
                    return True
        
        return False
    
    def is_circle_or_dot(self, contour: np.ndarray) -> bool:
        """Check if contour is approximately a circle (easier to draw than star)"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if area < 50 or perimeter < 20:
            return False
        
        # Calculate circularity (4π*area/perimeter²)
        # Perfect circle = 1, square ≈ 0.785
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.6:  # Reasonably circular
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
        
        # Step 1: Use approxPolyDP to reduce vertices
        perimeter = cv2.arcLength(contour, True)
        
        # Try different epsilon values to get desired vertex count
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
        
        # Step 2: Check if simplified contour is convex
        if len(simplified_contour) >= 3:
            hull = cv2.convexHull(simplified_contour)
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(simplified_contour)
            
            # Consider convex if the difference is small
            if hull_area > 0 and contour_area / hull_area > 0.95:
                simplification_info["is_convex"] = True
                simplified_contour = hull  # Use convex hull for better physics
                simplification_info["simplified_vertices"] = len(simplified_contour)
                simplification_info["method"] = "convex_hull"
        
        # Step 3: If still too complex or non-convex, try convex decomposition
        if (len(simplified_contour) > self.max_vertices or not simplification_info["is_convex"]) and self.use_convex_decomposition:
            try:
                # Convert contour to list of points for pymunk
                points = simplified_contour.reshape(-1, 2).astype(float)
                
                # Use pymunk's convex decomposition
                convex_pieces = pymunk.autogeometry.convex_decomposition(points.tolist(), 0.0)
                
                if convex_pieces:
                    # Store convex pieces
                    simplification_info["convex_pieces"] = []
                    for piece in convex_pieces:
                        piece_array = np.array(piece, dtype=np.int32).reshape(-1, 1, 2)
                        simplification_info["convex_pieces"].append(piece_array)
                    
                    simplification_info["method"] = "convex_decomposition"
                    simplification_info["num_pieces"] = len(convex_pieces)
                    
                    # Use the largest piece as the main contour for visualization
                    if convex_pieces:
                        largest_piece = max(convex_pieces, key=lambda p: cv2.contourArea(np.array(p, dtype=np.int32).reshape(-1, 1, 2)))
                        simplified_contour = np.array(largest_piece, dtype=np.int32).reshape(-1, 1, 2)
                        simplification_info["simplified_vertices"] = len(simplified_contour)
                        simplification_info["is_convex"] = True
                        
            except Exception as e:
                if self.debug:
                    print(f"Convex decomposition failed: {e}")
                # Fall back to simplified contour
                pass
        
        return simplified_contour, simplification_info
    
    def draw_output_image(self, original_img: np.ndarray, results: Dict[str, Any], 
                         contours: List[np.ndarray], output_path: str = None) -> np.ndarray:
        """Draw the output image with all detected shapes annotated"""
        # Create output image
        output_img = original_img.copy()
        
        # Draw rigid bodies first (so they appear behind other shapes)
        for i, body in enumerate(results["rigid_bodies"]):
            contour_id = body["contour_id"]
            original_contour = contours[contour_id]
            
            # Get simplified contour from stored points
            simplified_points = body["contour_points"]
            if simplified_points:
                # Convert back to contour format (unscaled for drawing)
                scale_factor = results.get("scale_factor", 1.0)
                unscaled_points = []
                for point in simplified_points:
                    unscaled_point = [int(point[0][0] / scale_factor), int(point[0][1] / scale_factor)]
                    unscaled_points.append([unscaled_point])
                simplified_contour = np.array(unscaled_points, dtype=np.int32)
            else:
                simplified_contour = original_contour
            
            # Draw original contour in light color
            cv2.drawContours(output_img, [original_contour], -1, (200, 200, 255), 1)
            
            # Draw simplified contour in bold color
            cv2.drawContours(output_img, [simplified_contour], -1, (255, 100, 100), 3)
            
            # Draw bounding box
            x, y, w, h = body["bounding_box"]
            # Unscale bounding box for display
            scale_factor = results.get("scale_factor", 1.0)
            display_box = (int(x / scale_factor), int(y / scale_factor), 
                          int(w / scale_factor), int(h / scale_factor))
            cv2.rectangle(output_img, (display_box[0], display_box[1]), 
                         (display_box[0] + display_box[2], display_box[1] + display_box[3]), 
                         (255, 150, 150), 2)
            
            # Add label with vertex count information
            simplification_info = body.get("simplification_info", {})
            original_vertices = simplification_info.get("original_vertices", "?")
            simplified_vertices = simplification_info.get("simplified_vertices", "?")
            method = simplification_info.get("method", "none")
            is_convex = simplification_info.get("is_convex", False)
            
            label = f"BODY_{i+1}: {original_vertices}→{simplified_vertices}v"
            if is_convex:
                label += " (convex)"
            elif method == "convex_decomposition":
                num_pieces = simplification_info.get("num_pieces", 0)
                label += f" ({num_pieces} pieces)"
            
            label_pos = (display_box[0], display_box[1] - 10) if display_box[1] > 20 else (display_box[0], display_box[1] + display_box[3] + 20)
            cv2.putText(output_img, label, label_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
        
        # Draw starting points (triangles/squares) in green
        for i, point in enumerate(results["starting_points"]):
            contour_id = point["contour_id"]
            contour = contours[contour_id]
            x, y = point["coordinates"]
            shape_type = point.get("shape_type", "unknown")
            
            # Draw contour
            cv2.drawContours(output_img, [contour], -1, (0, 255, 0), 4)
            
            # Draw center point
            cv2.circle(output_img, (x, y), 8, (0, 255, 0), -1)
            cv2.circle(output_img, (x, y), 12, (0, 200, 0), 3)
            
            # Add label with coordinates and shape type
            label = f"START_{i+1}({shape_type}) ({x},{y})"
            cv2.putText(output_img, label, (x - 80, y - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw end points (stars/circles) in red
        for i, point in enumerate(results["end_points"]):
            contour_id = point["contour_id"]
            contour = contours[contour_id]
            x, y = point["coordinates"]
            shape_type = point.get("shape_type", "unknown")
            
            # Draw contour
            cv2.drawContours(output_img, [contour], -1, (0, 0, 255), 4)
            
            # Draw center point
            cv2.circle(output_img, (x, y), 8, (0, 0, 255), -1)
            cv2.circle(output_img, (x, y), 12, (0, 0, 200), 3)
            
            # Add label with coordinates and shape type
            label = f"END_{i+1}({shape_type}) ({x},{y})"
            cv2.putText(output_img, label, (x - 70, y - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Add summary text at the top
        total_original_vertices = sum([body["simplification_info"]["original_vertices"] for body in results["rigid_bodies"]])
        total_simplified_vertices = sum([body["simplification_info"]["simplified_vertices"] for body in results["rigid_bodies"]])
        reduction_percentage = ((total_original_vertices - total_simplified_vertices) / total_original_vertices * 100) if total_original_vertices > 0 else 0
        
        summary_text = f"Detected: {len(results['starting_points'])} starts, {len(results['end_points'])} ends, {len(results['rigid_bodies'])} rigid bodies"
        vertex_text = f"Vertex Reduction: {total_original_vertices} → {total_simplified_vertices} ({reduction_percentage:.1f}%)"
        
        cv2.putText(output_img, summary_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(output_img, summary_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        
        cv2.putText(output_img, vertex_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(output_img, vertex_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Add legend
        legend_y_start = output_img.shape[0] - 140
        cv2.putText(output_img, "Legend:", (10, legend_y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(output_img, "Legend:", (10, legend_y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Green legend for starting points
        cv2.circle(output_img, (20, legend_y_start + 25), 8, (0, 255, 0), -1)
        cv2.putText(output_img, "= Starting Points (Triangles/Squares)", (35, legend_y_start + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(output_img, "= Starting Points (Triangles/Squares)", (35, legend_y_start + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Red legend for end points
        cv2.circle(output_img, (20, legend_y_start + 50), 8, (0, 0, 255), -1)
        cv2.putText(output_img, "= End Points (Stars/Circles)", (35, legend_y_start + 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(output_img, "= End Points (Stars/Circles)", (35, legend_y_start + 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Blue legend for rigid bodies with simplification info
        cv2.rectangle(output_img, (15, legend_y_start + 70), (25, legend_y_start + 80), (255, 100, 100), -1)
        cv2.putText(output_img, "= Rigid Bodies (Simplified for Physics)", (35, legend_y_start + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(output_img, "= Rigid Bodies (Simplified for Physics)", (35, legend_y_start + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Light blue for original contours
        cv2.rectangle(output_img, (15, legend_y_start + 95), (25, legend_y_start + 105), (200, 200, 255), -1)
        cv2.putText(output_img, "= Original Contours", (35, legend_y_start + 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(output_img, "= Original Contours", (35, legend_y_start + 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add shape suggestions
        cv2.putText(output_img, "TIP: Simplified shapes improve game performance!", (10, legend_y_start + 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(output_img, "TIP: Simplified shapes improve game performance!", (10, legend_y_start + 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Save output image if path provided
        if output_path:
            cv2.imwrite(output_path, output_img)
            print(f"Output image saved to: {output_path}")
        
        return output_img

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Main processing function to detect all shapes and classify them"""
        # Preprocess image
        original_img, processed_img = self.preprocess_image(image_path)

        # Get contours
        contours = self.get_contours(processed_img)

        # Scale factor to make coordinates smaller (adjust as needed)
        # This reduces the game world size to prevent performance issues
        scale_factor = 0.3

        # Initialize results
        results = {
            "starting_points": [],  # Triangles or squares
            "end_points": [],       # Stars or circles
            "rigid_bodies": [],     # Other shapes/lines (walls, ground, obstacles)
            "image_size": original_img.shape[:2],  # (height, width)
            "scale_factor": scale_factor  # Store scale factor for reference
        }
        
        # Classify each contour
        debug_img = original_img.copy() if self.debug else None
        
        for i, contour in enumerate(contours):
            centroid = self.get_centroid(contour)
            contour_area = cv2.contourArea(contour)

            # Apply scaling factor to centroid coordinates
            scaled_centroid = (int(centroid[0] * scale_factor), int(centroid[1] * scale_factor))

            # Try triangle first, then square as alternative
            if self.is_triangle(contour):
                results["starting_points"].append({
                    "coordinates": scaled_centroid,
                    "area": contour_area,
                    "contour_id": i,
                    "shape_type": "triangle"
                })
                if self.debug:
                    cv2.drawContours(debug_img, [contour], -1, (0, 255, 0), 3)  # Green for triangles
                    cv2.circle(debug_img, centroid, 5, (0, 255, 0), -1)

            elif self.is_circle_or_dot(contour):
                results["end_points"].append({
                    "coordinates": scaled_centroid,
                    "area": contour_area,
                    "contour_id": i,
                    "shape_type": "circle"
                })
                if self.debug:
                    cv2.drawContours(debug_img, [contour], -1, (0, 0, 200), 3)  # Darker red for circles
                    cv2.circle(debug_img, centroid, 5, (0, 0, 200), -1)

            else:
                # Everything else is considered a rigid body (walls, ground, obstacles)
                # Simplify contour for better physics performance
                simplified_contour, simplification_info = self.simplify_contour(contour)
                
                # Get bounding box for rigid bodies and scale it
                x, y, w, h = cv2.boundingRect(simplified_contour)
                scaled_bounding_box = (int(x * scale_factor), int(y * scale_factor),
                                     int(w * scale_factor), int(h * scale_factor))

                # Scale simplified contour points for rigid bodies
                scaled_contour_points = []
                for point in simplified_contour:
                    scaled_point = point[0] * scale_factor
                    scaled_contour_points.append([int(scaled_point[0]), int(scaled_point[1])])

                # Scale convex pieces if available
                scaled_convex_pieces = []
                if simplification_info.get("convex_pieces"):
                    for piece in simplification_info["convex_pieces"]:
                        scaled_piece = []
                        for point in piece:
                            scaled_point = point[0] * scale_factor
                            scaled_piece.append([[int(scaled_point[0]), int(scaled_point[1])]])
                        scaled_convex_pieces.append(scaled_piece)

                results["rigid_bodies"].append({
                    "bounding_box": scaled_bounding_box,
                    "centroid": scaled_centroid,
                    "area": contour_area,
                    "contour_id": i,
                    "contour_points": scaled_contour_points,  # Simplified contour points
                    # "original_contour_points": [[[int(point[0][0] * scale_factor), int(point[0][1] * scale_factor)]] for point in contour],  # Keep original for comparison
                    "simplification_info": {
                        **simplification_info,
                        "convex_pieces": scaled_convex_pieces  # Store scaled convex pieces for physics
                    }
                })
                if self.debug:
                    cv2.drawContours(debug_img, [simplified_contour], -1, (255, 0, 0), 2)  # Blue for simplified rigid bodies
                    cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 0), 1)
                    
                    # Draw original contour in lighter color for comparison
                    cv2.drawContours(debug_img, [contour], -1, (150, 150, 255), 1)  # Light blue for original
        
        if self.debug:
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
            
            # Calculate simplification statistics
            total_original_vertices = sum([body["simplification_info"]["original_vertices"] for body in results["rigid_bodies"]])
            total_simplified_vertices = sum([body["simplification_info"]["simplified_vertices"] for body in results["rigid_bodies"]])
            reduction_percentage = ((total_original_vertices - total_simplified_vertices) / total_original_vertices * 100) if total_original_vertices > 0 else 0
            
            title = f'Detected Shapes: {len(results["starting_points"])} starts, {len(results["end_points"])} ends, {len(results["rigid_bodies"])} rigid bodies\n'
            title += f'Vertex Reduction: {total_original_vertices} → {total_simplified_vertices} ({reduction_percentage:.1f}% reduction)'
            plt.title(title)
            plt.show()
        
        # Store contours for drawing output image
        results["_contours"] = contours
        results["_original_img"] = original_img
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Recognize hand-drawn shapes for game level creation')
    parser.add_argument('image_path', help='Path to the hand-drawn image')
    parser.add_argument('--debug', action='store_true', help='Show debug visualizations')
    parser.add_argument('--output', '-o', help='Output JSON file path (optional)')
    parser.add_argument('--draw', '-d', help='Output image file path with drawn annotations (optional)')
    parser.add_argument('--show-output', action='store_true', help='Display the output image with matplotlib')
    parser.add_argument('--no-simplify', action='store_true', help='Disable contour simplification')
    parser.add_argument('--max-vertices', type=int, default=200, help='Maximum vertices per contour (default: 20)')
    parser.add_argument('--simplification-factor', type=float, default=0.005, help='Contour simplification factor (default: 0.02)')
    parser.add_argument('--no-convex-decomp', action='store_true', help='Disable convex decomposition (requires pymunk)')
    
    args = parser.parse_args()
    
    try:
        # Initialize detector with simplification settings
        detector = ShapeDetector(
            debug=args.debug,
            simplify_contours=not args.no_simplify,
            max_vertices=args.max_vertices,
            simplification_factor=args.simplification_factor,
            use_convex_decomposition=not args.no_convex_decomp
        )
        
        # Process image
        results = detector.process_image(args.image_path)
        
        # Draw output image if requested
        output_img = None
        if args.draw or args.show_output:
            contours = results.pop("_contours")  # Remove from results before saving JSON
            original_img = results.pop("_original_img")
            
            output_img = detector.draw_output_image(
                original_img, results, contours, 
                output_path=args.draw if args.draw else None
            )
            
            # Show output image if requested
            if args.show_output:
                plt.figure(figsize=(15, 10))
                plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
                plt.title('Image Recognition Results')
                plt.axis('off')
                plt.tight_layout()
                plt.show()
        else:
            # Clean up internal data if not drawing
            results.pop("_contours", None)
            results.pop("_original_img", None)
        
        # Print results
        print(f"Image processed: {args.image_path}")
        print(f"Image size: {results['image_size'][1]}x{results['image_size'][0]} (WxH)")
        print(f"\nStarting points (triangles/squares): {len(results['starting_points'])}")
        for i, point in enumerate(results['starting_points']):
            shape_type = point.get('shape_type', 'unknown')
            print(f"  Start {i+1} ({shape_type}): coordinates {point['coordinates']}")
        
        print(f"\nEnd points (stars/circles): {len(results['end_points'])}")
        for i, point in enumerate(results['end_points']):
            shape_type = point.get('shape_type', 'unknown')
            print(f"  End {i+1} ({shape_type}): coordinates {point['coordinates']}")
        
        print(f"\nRigid Bodies (ground/walls/obstacles): {len(results['rigid_bodies'])}")
        total_original_vertices = 0
        total_simplified_vertices = 0
        
        for i, body in enumerate(results['rigid_bodies']):
            x, y, w, h = body['bounding_box']
            simplification_info = body.get('simplification_info', {})
            original_vertices = simplification_info.get('original_vertices', 0)
            simplified_vertices = simplification_info.get('simplified_vertices', 0)
            method = simplification_info.get('method', 'none')
            is_convex = simplification_info.get('is_convex', False)
            
            total_original_vertices += original_vertices
            total_simplified_vertices += simplified_vertices
            
            status = ""
            if method == "convex_decomposition":
                num_pieces = simplification_info.get('num_pieces', 0)
                status = f" (decomposed into {num_pieces} pieces)"
            elif is_convex:
                status = " (convex)"
            elif method != "none":
                status = f" (simplified using {method})"
            
            print(f"  Body {i+1}: bounding box ({x}, {y}, {w}, {h}), vertices {original_vertices}→{simplified_vertices}{status}")
        
        if total_original_vertices > 0:
            reduction_percentage = (total_original_vertices - total_simplified_vertices) / total_original_vertices * 100
            print(f"\nVertex Reduction Summary: {total_original_vertices} → {total_simplified_vertices} ({reduction_percentage:.1f}% reduction)")
            
            if not args.no_simplify:
                print(f"Simplification settings: max_vertices={args.max_vertices}, factor={args.simplification_factor}")
                if not args.no_convex_decomp and PYMUNK_AVAILABLE:
                    print("Convex decomposition: ENABLED")
                elif not args.no_convex_decomp:
                    print("Convex decomposition: DISABLED (pymunk not available)")
                else:
                    print("Convex decomposition: DISABLED")
        
        # Save to JSON if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")
        
        if args.draw:
            print(f"Annotated image saved to: {args.draw}")
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
