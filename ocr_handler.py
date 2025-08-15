import easyocr
import numpy as np
from PIL import Image
import io
import base64
from typing import List, Dict, Tuple, Optional
import cv2

class OCRHandler:
    def __init__(self, languages: List[str] = None):
        """
        Initialize OCR reader with specified languages
        
        Args:
            languages: List of language codes. Default includes English and Hindi
        """
        if languages is None:
            # English + Hindi (covers Marathi with Devanagari script)
            # Add Tamil (ta) and Kannada (kn) as needed
            languages = ['en', 'hi']
        
        print(f"Initializing EasyOCR with languages: {languages}")
        print("This may take a minute on first run to download models...")
        
        # Initialize without GPU for compatibility with i5/16GB setup
        self.reader = easyocr.Reader(languages, gpu=False)
        print("OCR initialization complete!")
    
    def detect_text(self, image_path: str = None, image_bytes: bytes = None) -> Dict:
        """
        Detect text regions in an image
        
        Args:
            image_path: Path to image file
            image_bytes: Image as bytes (for uploaded files)
            
        Returns:
            Dictionary containing detected text regions with metadata
        """
        try:
            # Load image
            if image_bytes:
                # Convert bytes to numpy array
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            elif image_path:
                image = cv2.imread(image_path)
            else:
                raise ValueError("Either image_path or image_bytes must be provided")
            
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Detect text with EasyOCR
            # Returns list of [bbox, text, confidence]
            results = self.reader.readtext(image)
            
            # Process results into structured format
            text_regions = []
            for idx, (bbox, text, confidence) in enumerate(results):
                # bbox is in format [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                # Convert to simple rectangle format
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                x_min = int(min(x_coords))
                y_min = int(min(y_coords))
                x_max = int(max(x_coords))
                y_max = int(max(y_coords))
                
                # Calculate text properties
                text_width = x_max - x_min
                text_height = y_max - y_min
                
                # Estimate font size (rough approximation)
                estimated_font_size = int(text_height * 0.75)
                
                # Extract dominant color from text region
                text_region_img = image[y_min:y_max, x_min:x_max]
                dominant_color = self._get_dominant_color(text_region_img)
                
                region = {
                    'id': f'text_region_{idx}',
                    'text': str(text),  # Ensure string
                    'confidence': float(confidence),
                    'bbox': {
                        'x': int(x_min),
                        'y': int(y_min),
                        'width': int(text_width),
                        'height': int(text_height)
                    },
                    'polygon': [[int(p[0]), int(p[1])] for p in bbox],  # Original polygon
                    'style': {
                        'estimated_font_size': int(estimated_font_size),
                        'text_color': str(dominant_color),
                        'is_bold': bool(self._detect_bold(text_region_img)),  # Convert numpy.bool to Python bool
                        'is_uppercase': bool(text.isupper()),
                        'word_count': int(len(text.split()))
                    },
                    'metadata': {
                        'has_trademark': bool(self._has_trademark_symbols(text)),
                        'is_likely_brand': bool(self._is_likely_brand(text)),
                        'script_type': str(self._detect_script(text))
                    },
                    'selected_for_translation': True  # Default to selected
                }
                
                text_regions.append(region)
            
            return {
                'success': True,
                'image_dimensions': {
                    'width': width,
                    'height': height
                },
                'total_regions': len(text_regions),
                'text_regions': text_regions
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'text_regions': []
            }
    
    def _get_dominant_color(self, image_region: np.ndarray) -> str:
        """
        Extract dominant text color from image region
        
        Args:
            image_region: Cropped image containing text
            
        Returns:
            Hex color code
        """
        try:
            if image_region.size == 0:
                return "#000000"
            
            # Convert to RGB if needed
            if len(image_region.shape) == 3:
                # Use edge detection to find text pixels
                gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                
                # Get colors where edges are detected (likely text)
                text_pixels = image_region[edges > 0]
                
                if len(text_pixels) > 0:
                    # Get mean color of text pixels
                    mean_color = np.mean(text_pixels, axis=0)
                    # Convert BGR to RGB
                    rgb = mean_color[::-1]
                else:
                    # Fallback to darkest pixels (usually text)
                    gray_flat = gray.flatten()
                    dark_threshold = np.percentile(gray_flat, 30)
                    dark_pixels = image_region[gray < dark_threshold]
                    if len(dark_pixels) > 0:
                        mean_color = np.mean(dark_pixels, axis=0)
                        rgb = mean_color[::-1]
                    else:
                        rgb = [0, 0, 0]
            else:
                rgb = [0, 0, 0]
            
            # Convert to hex
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgb[0]), int(rgb[1]), int(rgb[2])
            )
            return hex_color
            
        except Exception:
            return "#000000"  # Default to black
    
    def _detect_bold(self, image_region: np.ndarray) -> bool:
        """
        Detect if text appears to be bold based on stroke width
        
        Args:
            image_region: Cropped image containing text
            
        Returns:
            Boolean indicating if text appears bold
        """
        try:
            if image_region.size == 0:
                return False
            
            # Convert to grayscale
            gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Calculate stroke width using distance transform
            dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            
            # If average distance is high, text is likely bold
            avg_thickness = np.mean(dist_transform[dist_transform > 0]) if dist_transform.any() else 0
            
            # Threshold for bold detection (adjust based on testing)
            return avg_thickness > 2.5
            
        except Exception:
            return False
    
    def _has_trademark_symbols(self, text: str) -> bool:
        """Check if text contains trademark symbols"""
        trademark_symbols = ['®', '™', '©', '℠']
        return any(symbol in text for symbol in trademark_symbols)
    
    def _is_likely_brand(self, text: str) -> bool:
        """
        Determine if text is likely a brand name
        
        Criteria:
        - All uppercase and 1-3 words
        - Contains trademark symbols
        - Follows brand name patterns
        """
        if self._has_trademark_symbols(text):
            return True
        
        words = text.split()
        if len(words) <= 3 and text.isupper():
            return True
        
        # Check for common brand patterns
        brand_patterns = [
            text.endswith('.com'),
            text.endswith('.in'),
            '@' in text,  # Social media handles
            '#' in text and len(words) == 1,  # Hashtags
        ]
        
        return any(brand_patterns)
    
    def _detect_script(self, text: str) -> str:
        """
        Detect the script type of the text
        
        Returns:
            Script type: 'latin', 'devanagari', 'tamil', 'kannada', 'mixed'
        """
        scripts = {
            'latin': 0,
            'devanagari': 0,
            'tamil': 0,
            'kannada': 0
        }
        
        for char in text:
            code_point = ord(char)
            
            # Latin script
            if (0x0041 <= code_point <= 0x007A) or (0x0020 <= code_point <= 0x0040):
                scripts['latin'] += 1
            # Devanagari (Hindi, Marathi)
            elif 0x0900 <= code_point <= 0x097F:
                scripts['devanagari'] += 1
            # Tamil
            elif 0x0B80 <= code_point <= 0x0BFF:
                scripts['tamil'] += 1
            # Kannada
            elif 0x0C80 <= code_point <= 0x0CFF:
                scripts['kannada'] += 1
        
        total_chars = sum(scripts.values())
        if total_chars == 0:
            return 'unknown'
        
        # Find dominant script
        dominant_script = max(scripts, key=scripts.get)
        
        # Check if mixed
        if scripts[dominant_script] < total_chars * 0.8:
            return 'mixed'
        
        return dominant_script


# Test function
def test_ocr_handler():
    """Test the OCR handler with a sample image"""
    
    # Initialize handler
    print("Creating OCR Handler...")
    handler = OCRHandler()
    
    # Test with a sample image (you'll need to provide path)
    test_image_path = "sample_marketing_image.jpg"  # Replace with actual path
    
    print(f"\nTesting with image: {test_image_path}")
    result = handler.detect_text(image_path=test_image_path)
    
    if result['success']:
        print(f"\n✅ Successfully detected {result['total_regions']} text regions!")
        print(f"Image dimensions: {result['image_dimensions']}")
        
        for region in result['text_regions']:
            print(f"\n📝 Region: {region['id']}")
            print(f"   Text: '{region['text']}'")
            print(f"   Confidence: {region['confidence']:.2f}")
            print(f"   Position: x={region['bbox']['x']}, y={region['bbox']['y']}")
            print(f"   Size: {region['bbox']['width']}x{region['bbox']['height']}")
            print(f"   Font Size: ~{region['style']['estimated_font_size']}px")
            print(f"   Color: {region['style']['text_color']}")
            print(f"   Bold: {region['style']['is_bold']}")
            print(f"   Likely Brand: {region['metadata']['is_likely_brand']}")
            print(f"   Script: {region['metadata']['script_type']}")
    else:
        print(f"❌ Error: {result['error']}")


if __name__ == "__main__":
    # Run test
    test_ocr_handler()