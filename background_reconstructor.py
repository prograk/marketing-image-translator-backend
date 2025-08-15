import cv2
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
from sklearn.cluster import KMeans

class BackgroundReconstructor:
    """Advanced background reconstruction for clean text removal"""
    
    def __init__(self):
        pass
    
    def reconstruct_background(self, image: Image.Image, x: int, y: int, w: int, h: int) -> Image.Image:
        """
        Reconstruct background in text region using advanced techniques
        """
        try:
            # Convert to OpenCV format
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Create mask for the text region
            mask = np.zeros(img_cv.shape[:2], dtype=np.uint8)
            mask[y:y+h, x:x+w] = 255
            
            # Use OpenCV's inpainting for background reconstruction
            inpainted = cv2.inpaint(img_cv, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            
            # Convert back to PIL
            inpainted_pil = Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
            
            return inpainted_pil
            
        except Exception as e:
            print(f"Advanced background reconstruction failed: {e}")
            return self._simple_background_fill(image, x, y, w, h)
    
    def _simple_background_fill(self, image: Image.Image, x: int, y: int, w: int, h: int) -> Image.Image:
        """Fallback simple background filling"""
        img_copy = image.copy()
        
        # Sample surrounding area for background color
        margin = 10
        samples = []
        
        # Sample above
        if y > margin:
            above = image.crop((x, y-margin, x+w, y))
            samples.extend(list(above.getdata()))
        
        # Sample below
        if y+h+margin < image.height:
            below = image.crop((x, y+h, x+w, min(image.height, y+h+margin)))
            samples.extend(list(below.getdata()))
        
        # Sample left
        if x > margin:
            left = image.crop((x-margin, y, x, y+h))
            samples.extend(list(left.getdata()))
        
        # Sample right
        if x+w+margin < image.width:
            right = image.crop((x+w, y, min(image.width, x+w+margin), y+h))
            samples.extend(list(right.getdata()))
        
        if samples:
            # Calculate average color
            avg_color = tuple(int(sum(channel)/len(samples)) for channel in zip(*samples))
        else:
            avg_color = (255, 255, 255)  # Default white
        
        # Fill the region
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img_copy)
        draw.rectangle([x, y, x+w, y+h], fill=avg_color)
        
        return img_copy
    
    def analyze_text_region(self, image: Image.Image, x: int, y: int, w: int, h: int) -> dict:
        """Analyze text region for better processing"""
        try:
            # Extract text region
            text_region = image.crop((x, y, x+w, y+h))
            text_array = np.array(text_region)
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(text_array, cv2.COLOR_RGB2GRAY)
            
            # Detect text edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours (text boundaries)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calculate text coverage
            text_mask = np.zeros_like(gray)
            cv2.drawContours(text_mask, contours, -1, 255, -1)
            text_coverage = np.sum(text_mask > 0) / (w * h)
            
            # Estimate background complexity
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            entropy = -np.sum((hist + 1e-7) * np.log(hist + 1e-7))
            
            return {
                'text_coverage': text_coverage,
                'background_complexity': entropy,
                'contours': contours,
                'edges': edges
            }
            
        except Exception as e:
            print(f"Text region analysis failed: {e}")
            return {
                'text_coverage': 0.5,
                'background_complexity': 100,
                'contours': [],
                'edges': None
            }
    
    def smart_color_sampling(self, image: Image.Image, x: int, y: int, w: int, h: int) -> tuple:
        """Smart color sampling avoiding text areas"""
        try:
            # Analyze the region first
            analysis = self.analyze_text_region(image, x, y, w, h)
            
            # If text coverage is low, sample from the region itself
            if analysis['text_coverage'] < 0.3:
                region = image.crop((x, y, x+w, y+h))
                pixels = list(region.getdata())
                
                # Use KMeans to find dominant background color
                if len(pixels) > 10:
                    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                    kmeans.fit(pixels)
                    
                    # Choose the cluster with more pixels (likely background)
                    labels = kmeans.labels_
                    unique, counts = np.unique(labels, return_counts=True)
                    bg_cluster = unique[np.argmax(counts)]
                    bg_color = tuple(map(int, kmeans.cluster_centers_[bg_cluster]))
                    
                    return bg_color
            
            # Otherwise, sample from surrounding areas
            return self._sample_surrounding_areas(image, x, y, w, h)
            
        except Exception as e:
            print(f"Smart color sampling failed: {e}")
            return self._sample_surrounding_areas(image, x, y, w, h)
    
    def _sample_surrounding_areas(self, image: Image.Image, x: int, y: int, w: int, h: int) -> tuple:
        """Sample colors from areas surrounding the text"""
        margin = 5
        samples = []
        
        # Define sampling areas with weights
        sampling_areas = [
            # (x1, y1, x2, y2, weight)
            (max(0, x-margin), max(0, y-margin), x, y, 0.3),  # Top-left
            (x+w, max(0, y-margin), min(image.width, x+w+margin), y, 0.3),  # Top-right
            (max(0, x-margin), y+h, x, min(image.height, y+h+margin), 0.3),  # Bottom-left
            (x+w, y+h, min(image.width, x+w+margin), min(image.height, y+h+margin), 0.3),  # Bottom-right
        ]
        
        weighted_samples = []
        
        for x1, y1, x2, y2, weight in sampling_areas:
            if x2 > x1 and y2 > y1:
                area = image.crop((x1, y1, x2, y2))
                area_pixels = list(area.getdata())
                
                # Add weighted samples
                for pixel in area_pixels:
                    for _ in range(int(weight * 10)):  # Weight simulation
                        weighted_samples.append(pixel)
        
        if weighted_samples:
            # Calculate weighted average
            avg_color = tuple(int(sum(channel)/len(weighted_samples)) 
                            for channel in zip(*weighted_samples))
            return avg_color
        else:
            return (255, 255, 255)  # Default white