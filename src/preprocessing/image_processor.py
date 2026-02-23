import cv2
import numpy as np
from PIL import Image

class ImagePreprocessor:
    """
    Handles image preprocessing for historical document OCR.
    Focuses on isolating main text from embellishments.
    """
    
    @staticmethod
    def preprocess(image_path):
        """
        Full preprocessing pipeline: Grayscale -> Adaptive Threshold -> Contour Detection -> Cropping.
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding to handle uneven lighting in historical sources
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations to group text characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilate = cv2.dilate(thresh, kernel, iterations=3)
        
        # Find contours
        contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size to isolate main text block (heuristics)
        # We assume the largest central contour is the main text block
        if not contours:
            return Image.fromarray(gray)
            
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # Crop to the main text block
        cropped_img = gray[y:y+h, x:x+w]
        
        # Return as PIL Image for TrOCR compatibility
        return Image.fromarray(cropped_img).convert("RGB")

if __name__ == "__main__":
    # Test block
    print("ImagePreprocessor module loaded.")
