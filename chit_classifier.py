import cv2
import numpy as np

class ChitClassifier:
    """
    Classifies and detects small paper chits.
    Relaxed geometric constraints to account for fingers holding the paper,
    while still filtering out major background noise.
    """
    def __init__(self, min_area=150, max_area=6000):
        self.min_area = min_area
        self.max_area = max_area 

    def detect(self, img_bgr):
        detected_chits = []
        
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # Real-world white paper under normal room lighting requires broader thresholds.
        # Saturation is loosened (0-70) for warm lights, but brightness raised to 180 to reject glare.
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 70, 255])
        
        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Morphological operations to close gaps made by text on the chit or shadows
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area < area < self.max_area:
                
                # --- Solidity Check ---
                # Lowered to 0.60 to account for fingers partly occluding the paper
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0: continue
                solidity = float(area) / hull_area
                
                if solidity > 0.60:
                    
                    # --- Geometric Shape Check ---
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
                    
                    # 4 to 10 vertices allows for the paper to be slightly crumpled or partly clamped by fingers
                    if 4 <= len(approx) <= 10:
                        x, y, w, h = cv2.boundingRect(approx)
                        aspect_ratio = float(w)/h if h > 0 else 0
                        
                        # Flexible aspect ratio limits
                        if 0.4 < aspect_ratio < 2.5:
                            # Secondary validation: verify color balance using a small average patch (stable)
                            center_x, center_y = x + w // 2, y + h // 2
                            y1, y2 = max(0, center_y-2), min(img_bgr.shape[0], center_y+3)
                            x1, x2 = max(0, center_x-2), min(img_bgr.shape[1], center_x+3)
                            avg_patch = np.mean(img_bgr[y1:y2, x1:x2], axis=(0, 1))
                            b, g, r = avg_patch
                            
                            # Accept if RGB values are balanced and bright (white paper)
                            if r > 150 and g > 150 and b > 150:
                                rgb_variance = max(r, g, b) - min(r, g, b)
                                if rgb_variance < 35:  # Tightened variance to ensure it's truly neutral white/gray
                                    detected_chits.append((x, y, x+w, y+h, "Paper Chit", 0.90))
                            
        return detected_chits
