import os
import platform
import requests
import zipfile
from fontTools.ttLib import TTFont
from PIL import ImageFont, Image, ImageDraw
from pathlib import Path
from typing import Optional, Dict, List, Union
import numpy as np
import cv2

class FontManager:
    def __init__(self, cache_dir: str = "fonts"):
        """Initialize font manager with online font downloading"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Font cache for PIL ImageFont objects
        self.font_cache = {}
        
        # Online font sources
        self.font_urls = {
            # Noto Sans fonts from Google Fonts
            'noto sans devanagari': 'https://fonts.google.com/download?family=Noto%20Sans%20Devanagari',
            'noto sans tamil': 'https://fonts.google.com/download?family=Noto%20Sans%20Tamil',
            'noto sans telugu': 'https://fonts.google.com/download?family=Noto%20Sans%20Telugu',
            'noto sans kannada': 'https://fonts.google.com/download?family=Noto%20Sans%20Kannada',
            'noto sans bengali': 'https://fonts.google.com/download?family=Noto%20Sans%20Bengali',
            'noto sans': 'https://fonts.google.com/download?family=Noto%20Sans',
            'roboto': 'https://fonts.google.com/download?family=Roboto',
            'open sans': 'https://fonts.google.com/download?family=Open%20Sans',
            'lato': 'https://fonts.google.com/download?family=Lato',
            'montserrat': 'https://fonts.google.com/download?family=Montserrat'
        }
        
        # Language to preferred fonts mapping (using downloadable fonts)
        self.language_fonts = {
            'hi': ['Noto Sans Devanagari', 'Noto Sans'],
            'hindi': ['Noto Sans Devanagari', 'Noto Sans'],
            'mr': ['Noto Sans Devanagari', 'Noto Sans'],
            'marathi': ['Noto Sans Devanagari', 'Noto Sans'],
            'ta': ['Noto Sans Tamil', 'Noto Sans'],
            'tamil': ['Noto Sans Tamil', 'Noto Sans'],
            'te': ['Noto Sans Telugu', 'Noto Sans'],
            'telugu': ['Noto Sans Telugu', 'Noto Sans'],
            'kn': ['Noto Sans Kannada', 'Noto Sans'],
            'kannada': ['Noto Sans Kannada', 'Noto Sans'],
            'bn': ['Noto Sans Bengali', 'Noto Sans'],
            'bengali': ['Noto Sans Bengali', 'Noto Sans'],
            'en': ['Roboto', 'Open Sans', 'Lato', 'Noto Sans'],
            'english': ['Roboto', 'Open Sans', 'Lato', 'Noto Sans']
        }
        
        # Build font database from cache and download missing fonts
        self.available_fonts = self._build_font_database()
        
        print(f"FontManager initialized. Found {len(self.available_fonts)} fonts.")
    
    def _download_font(self, font_name: str) -> bool:
        """Download font from Google Fonts"""
        font_key = font_name.lower()
        
        if font_key not in self.font_urls:
            print(f"Font '{font_name}' not available for download")
            return False
        
        try:
            print(f"Downloading {font_name}...")
            
            # Create font-specific directory
            font_dir = self.cache_dir / font_key.replace(' ', '_')
            font_dir.mkdir(exist_ok=True)
            
            # Download the zip file
            response = requests.get(self.font_urls[font_key], timeout=30)
            response.raise_for_status()
            
            # Save and extract zip
            zip_path = font_dir / f"{font_key.replace(' ', '_')}.zip"
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            # Extract TTF files
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file_info in zip_ref.filelist:
                    if file_info.filename.endswith('.ttf'):
                        zip_ref.extract(file_info, font_dir)
            
            # Clean up zip file
            zip_path.unlink()
            
            print(f"✓ Downloaded {font_name}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to download {font_name}: {e}")
            return False
    
    def _get_cached_fonts(self) -> List[Path]:
        """Get all cached font files"""
        font_files = []
        
        if self.cache_dir.exists():
            # Look for TTF files in cache directory and subdirectories
            font_files.extend(self.cache_dir.rglob("*.ttf"))
            font_files.extend(self.cache_dir.rglob("*.otf"))
        
        return font_files
    
    def _build_font_database(self) -> Dict[str, Dict]:
        """Build database of available fonts from cache"""
        fonts = {}
        
        # Get cached fonts
        cached_fonts = self._get_cached_fonts()
        
        for font_file in cached_fonts:
            try:
                font_info = self._analyze_font(font_file)
                if font_info:
                    fonts[font_info['family'].lower()] = font_info
            except Exception as e:
                continue
        
        return fonts
    
    def _ensure_font_available(self, font_name: str) -> bool:
        """Ensure font is available, download if necessary"""
        font_key = font_name.lower()
        
        # Check if already available
        if font_key in self.available_fonts:
            return True
        
        # Try to download
        if self._download_font(font_name):
            # Rebuild database to include new font
            new_fonts = self._build_font_database()
            self.available_fonts.update(new_fonts)
            return font_key in self.available_fonts
        
        return False
    
    def _get_test_chars(self, language: str) -> str:
        """Get test characters for language"""
        test_chars = {
            'hi': 'हिंदी डिज़ाइन',
            'hindi': 'हिंदी डिज़ाइन', 
            'mr': 'मराठी डिझाइन',
            'marathi': 'मराठी डिझाइन',
            'ta': 'தமிழ் வடிவமைப்பு',
            'tamil': 'தமிழ் வடிவமைப்பு',
            'te': 'తెలుగు డిజైన్',
            'telugu': 'తెలుగు డిజైన్',
            'kn': 'ಕನ್ನಡ ವಿನ್ಯಾಸ',
            'kannada': 'ಕನ್ನಡ ವಿನ್ಯಾಸ',
            'bn': 'বাংলা ডিজাইন',
            'bengali': 'বাংলা ডিজাইন'
        }
        
        return test_chars.get(language.lower(), 'ABCabc')
    
    def find_best_font(self, language: str, size: int = 20, weight: str = 'normal', style: str = 'normal') -> ImageFont.FreeTypeFont:
        """Find best font for language, downloading if necessary"""
        cache_key = f"{language}_{size}_{weight}_{style}"
        
        if cache_key in self.font_cache:
            return self.font_cache[cache_key]
        
        # Get preferred fonts for language
        preferred_fonts = self.language_fonts.get(language.lower(), ['Noto Sans'])
        
        # Test characters for the language
        test_chars = self._get_test_chars(language)
        test_codepoints = {ord(c) for c in test_chars}
        
        best_font_path = None
        best_score = 0
        
        # Try preferred fonts first, download if needed
        for font_family in preferred_fonts:
            # Ensure font is available
            if not self._ensure_font_available(font_family):
                continue
            
            font_key = font_family.lower()
            
            if font_key in self.available_fonts:
                font_info = self.available_fonts[font_key]
                
                # Check character support
                supported = font_info['supported_chars']
                coverage = len(test_codepoints & supported) / len(test_codepoints)
                
                # Weight preference
                weight_score = 1.0
                if weight == 'bold' and font_info['weight'] == 'bold':
                    weight_score = 1.5
                
                # Style preference  
                style_score = 1.0
                if style == 'italic' and font_info['style'] == 'italic':
                    style_score = 1.2
                
                # Calculate total score
                score = coverage * weight_score * style_score
                
                if score > best_score:
                    best_score = score
                    best_font_path = font_info['path']
                    
                    # If perfect match, use it
                    if coverage >= 0.9:
                        break
        
        # If no good match found, ensure we have at least Noto Sans
        if best_score < 0.5:
            if self._ensure_font_available('Noto Sans'):
                font_info = self.available_fonts.get('noto sans')
                if font_info:
                    best_font_path = font_info['path']
                    best_score = 0.8
        
        # Create PIL font
        if best_font_path and os.path.exists(best_font_path):
            try:
                font = ImageFont.truetype(best_font_path, size)
                self.font_cache[cache_key] = font
                print(f"✓ Selected font for {language}: {os.path.basename(best_font_path)} (score: {best_score:.2f})")
                return font
            except Exception as e:
                print(f"Error loading font {best_font_path}: {e}")
        
        # Final fallback - try to get any available font
        return self._get_fallback_font(size)
    
    def _get_fallback_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Get fallback font, download Noto Sans if nothing available"""
        try:
            # If we have any font available, use it
            if self.available_fonts:
                font_info = next(iter(self.available_fonts.values()))
                return ImageFont.truetype(font_info['path'], size)
            
            # Try to download Noto Sans as fallback
            if self._ensure_font_available('Noto Sans'):
                font_info = self.available_fonts.get('noto sans')
                if font_info:
                    return ImageFont.truetype(font_info['path'], size)
            
            # Ultimate fallback
            return ImageFont.load_default()
            
        except Exception as e:
            print(f"Fallback font error: {e}")
            return ImageFont.load_default()
    
    def _analyze_font(self, font_path: Path) -> Optional[Dict]:
        """Analyze font using fonttools"""
        try:
            font = TTFont(font_path)
            
            # Get font family name
            name_table = font.get('name')
            family_name = None
            
            if name_table:
                # Try to get English family name (nameID 1)
                for record in name_table.names:
                    if record.nameID == 1 and record.platformID == 3:  # Microsoft platform
                        family_name = record.toUnicode()
                        break
                
                if not family_name:
                    # Fallback to any family name
                    for record in name_table.names:
                        if record.nameID == 1:
                            try:
                                family_name = record.toUnicode()
                                break
                            except:
                                continue
            
            if not family_name:
                family_name = font_path.stem
            
            # Get character coverage
            cmap = font.getBestCmap()
            supported_chars = set(cmap.keys()) if cmap else set()
            
            # Detect weight and style
            weight = 'normal'
            style = 'normal'
            
            if name_table:
                for record in name_table.names:
                    if record.nameID == 2:  # Subfamily name (style)
                        try:
                            subfamily = record.toUnicode().lower()
                            if 'bold' in subfamily:
                                weight = 'bold'
                            if 'italic' in subfamily or 'oblique' in subfamily:
                                style = 'italic'
                        except:
                            continue
            
            return {
                'family': family_name,
                'path': str(font_path),
                'weight': weight,
                'style': style,
                'supported_chars': supported_chars
            }
            
        except Exception as e:
            return None
    
    def get_font(self, language: str, size: int = 20) -> ImageFont.FreeTypeFont:
        """Main method to get font (backward compatibility)"""
        return self.find_best_font(language, size)
    
    def get_text_dimensions(self, text: str, font: ImageFont.FreeTypeFont) -> tuple:
        """Get text dimensions for proper positioning"""
        try:
            # Use textbbox for accurate measurements
            bbox = font.getbbox(text)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            return width, height
        except:
            # Fallback method
            return font.getsize(text)

    def wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
        """Wrap text to fit within max_width"""
        words = text.split(' ')
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            width, _ = self.get_text_dimensions(test_line, font)
            
            if width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Word is too long, add it anyway
                    lines.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def analyze_original_font(self, image_path: str, x: int, y: int, w: int, h: int, text: str) -> Dict:
        """Enhanced analysis of original font properties from image region"""
        try:
            from PIL import Image
            
            image = Image.open(image_path).convert('RGB')
            img_array = np.array(image)
            
            print(f"\n=== DEBUGGING COLOR DETECTION FOR: '{text}' ===")
            print(f"Text region coordinates: x={x}, y={y}, w={w}, h={h}")
            
            # Extract CORE text region (center 60% to avoid edges and background)
            margin_x = max(1, w // 5)  # 20% margin on each side
            margin_y = max(1, h // 5)  # 20% margin on top/bottom
            
            core_x_start = max(0, x + margin_x)
            core_y_start = max(0, y + margin_y)
            core_x_end = min(img_array.shape[1], x + w - margin_x)
            core_y_end = min(img_array.shape[0], y + h - margin_y)
            
            # Extract regions
            text_region = img_array[core_y_start:core_y_end, core_x_start:core_x_end]
            
            # Context region (larger area around text)
            padding = max(5, min(w, h) // 8)
            context_x_start = max(0, x - padding)
            context_y_start = max(0, y - padding)
            context_x_end = min(img_array.shape[1], x + w + padding)
            context_y_end = min(img_array.shape[0], y + h + padding)
            context_region = img_array[context_y_start:context_y_end, context_x_start:context_x_end]
            
            print(f"Core text region shape: {text_region.shape}")
            print(f"Using core region: x={core_x_start}-{core_x_end}, y={core_y_start}-{core_y_end}")
            
            if text_region.size == 0:
                print("WARNING: Core text region is empty, using full region")
                text_region = img_array[y:y+h, x:x+w] if h > 0 and w > 0 else np.array([[[255,255,255]]])
            
            # Enhanced font size calculation
            estimated_size = self._calculate_optimal_font_size(text, w, h, text_region)
            
            # Enhanced weight detection
            weight = self._detect_font_weight_advanced(text_region, estimated_size)
            
            # Enhanced color extraction WITH DEBUG
            text_color = self._extract_text_color_advanced(text_region, context_region, text)
            
            # Background color analysis
            background_color = self._extract_background_color_advanced(img_array, x, y, w, h)
            
            font_props = {
                'estimated_size': estimated_size,
                'weight': weight,
                'style': 'normal',  # Can be enhanced later for italic detection
                'color': text_color,
                'background': background_color,
                'contrast_ratio': self._calculate_contrast_ratio(text_color, background_color),
                'text_density': self._calculate_text_density(text_region)
            }
            
            print(f"=== FINAL ANALYSIS FOR '{text}' ===")
            print(f"  Size: {font_props['estimated_size']}")
            print(f"  Weight: {font_props['weight']}")
            print(f"  Color: RGB{font_props['color']}")
            print(f"  Background: RGB{font_props['background']}")
            print(f"  Contrast: {font_props['contrast_ratio']:.2f}")
            print(f"=======================================\n")
            
            return font_props
            
        except Exception as e:
            print(f"Error analyzing font: {e}")
            return {
                'estimated_size': 20,
                'weight': 'normal',
                'style': 'normal',
                'color': (255, 255, 255),
                'background': (41, 128, 185),
                'contrast_ratio': 4.5,
                'text_density': 0.3
            }
    
    def _calculate_optimal_font_size(self, text: str, width: int, height: int, region: np.ndarray) -> int:
        """Simplified but robust font size calculation"""
        try:
            # Primary calculation based on height (most reliable)
            height_based_size = max(8, int(height * 0.75))
            
            # Secondary validation using stroke thickness
            stroke_thickness = self._analyze_stroke_thickness(region)
            if stroke_thickness > 2:  # Only use if we have clear stroke data
                stroke_based_size = max(8, int(stroke_thickness * 6))
                # Use stroke size only if it's reasonable (within 50% of height-based)
                if 0.5 * height_based_size <= stroke_based_size <= 1.5 * height_based_size:
                    height_based_size = int(0.8 * height_based_size + 0.2 * stroke_based_size)
            
            # Simple character density check (prevent extreme cases)
            text_length = len(text.strip())
            if text_length > 0:
                char_width = width / text_length
                if char_width < 6:  # Very cramped text
                    height_based_size = max(8, int(height_based_size * 0.9))
                elif char_width > 30:  # Very sparse text  
                    height_based_size = min(48, int(height_based_size * 1.1))
            
            # Ensure reasonable bounds
            final_size = max(8, min(48, height_based_size))
            
            print(f"Size calc: height={height} → base={int(height * 0.75)} → final={final_size}")
            return final_size
            
        except Exception as e:
            print(f"Error calculating font size: {e}")
            return max(12, int(height * 0.75))
    
    def _detect_font_weight_advanced(self, text_region: np.ndarray, font_size: int) -> str:
        """Advanced font weight detection using multiple techniques"""
        try:
            if text_region.size == 0:
                return 'normal'
            
            # Convert to grayscale
            gray = cv2.cvtColor(text_region, cv2.COLOR_RGB2GRAY)
            
            # Method 1: Stroke width analysis
            stroke_width = self._analyze_stroke_thickness(text_region)
            
            # Method 2: Edge density analysis
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Method 3: Intensity gradient analysis
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            avg_gradient = np.mean(gradient_magnitude)
            
            # Method 4: Text pixel density
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text_pixel_ratio = np.sum(binary < 128) / binary.size
            
            # Combine metrics for decision
            bold_score = 0
            
            # Stroke width scoring (normalized by font size)
            if stroke_width > 0:
                normalized_stroke = stroke_width / (font_size / 16)  # Normalize to 16px baseline
                if normalized_stroke > 1.5:
                    bold_score += 2
                elif normalized_stroke > 1.2:
                    bold_score += 1
            
            # Edge density scoring
            if edge_density > 0.15:
                bold_score += 1
            
            # Gradient scoring
            if avg_gradient > 30:
                bold_score += 1
            
            # Text density scoring
            if text_pixel_ratio > 0.35:
                bold_score += 1
            
            # Decision threshold
            weight = 'bold' if bold_score >= 3 else 'normal'
            
            print(f"Weight analysis: stroke={stroke_width:.2f}, edge_density={edge_density:.3f}, "
                  f"gradient={avg_gradient:.1f}, density={text_pixel_ratio:.3f}, score={bold_score} → {weight}")
            
            return weight
            
        except Exception as e:
            print(f"Error detecting font weight: {e}")
            return 'normal'
    
    def _analyze_stroke_thickness(self, region: np.ndarray) -> float:
        """Analyze average stroke thickness of text"""
        try:
            if region.size == 0:
                return 0
            
            # Convert to grayscale
            gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
            
            # Apply threshold to get binary image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Invert so text is white
            binary = cv2.bitwise_not(binary)
            
            # Calculate distance transform
            dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            
            # Find skeleton and calculate average thickness
            non_zero_distances = dist_transform[dist_transform > 0]
            if len(non_zero_distances) > 0:
                # Average thickness is 2 * average distance
                avg_thickness = 2 * np.mean(non_zero_distances.astype(float))
                return float(avg_thickness)
            
            return 0
            
        except Exception as e:
            return 0
    
    def _extract_text_color_advanced(self, text_region: np.ndarray, context_region: np.ndarray, text: str = "") -> tuple:
        """Advanced text color extraction using context WITH DEBUG"""
        try:
            if text_region.size == 0:
                print(f"  DEBUG: Empty text region for '{text}', returning white")
                return (255, 255, 255)
            
            print(f"  DEBUG: Analyzing color for '{text}'")
            print(f"  Text region shape: {text_region.shape}")
            
            # Convert to different color spaces for analysis
            text_rgb = text_region
            
            # Method 1: Simple dominant color approach (PRIORITIZED FOR BRIGHT TEXT)
            pixels = text_rgb.reshape(-1, 3)
            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
            
            print(f"  DEBUG: Found {len(unique_colors)} unique colors")
            print(f"  DEBUG: Top 5 colors by frequency:")
            
            # Sort by frequency
            sorted_indices = np.argsort(counts)[::-1]
            for i in range(min(5, len(unique_colors))):
                idx = sorted_indices[i]
                color = unique_colors[idx]
                count = counts[idx]
                percentage = (count / len(pixels)) * 100
                brightness = np.mean(color)
                print(f"    Color RGB{tuple(color)}: {count} pixels ({percentage:.1f}%) brightness={brightness:.1f}")
            
            # PRIORITY 1: Check for bright colors first (WHITE TEXT DETECTION)
            print(f"  DEBUG: Checking for bright colors (brightness > 200)...")
            for i, idx in enumerate(sorted_indices):
                color = unique_colors[idx]
                count = counts[idx]
                percentage = (count / len(pixels)) * 100
                brightness = np.mean(color)
                
                # Lower threshold for small text regions (adjust dynamically)
                total_pixels = len(pixels)
                min_threshold = max(1.0, min(5.0, 1000.0 / total_pixels))  # Adaptive threshold
                
                # Look for bright colors that represent significant portion of text
                if brightness > 200 and percentage > min_threshold:
                    print(f"  DEBUG: Found bright text color! RGB{tuple(color)} (brightness={brightness:.1f}, {percentage:.1f}%)")
                    final_color = tuple(map(int, color))
                    print(f"  DEBUG: USING BRIGHT COLOR: RGB{final_color}")
                    return final_color
            
            # PRIORITY 2: Check for white-ish colors even if not most frequent (LOWER THRESHOLD)
            print(f"  DEBUG: No dominant bright color found, checking for any white-ish colors...")
            for i, idx in enumerate(sorted_indices[:15]):  # Check top 15 colors
                color = unique_colors[idx]
                count = counts[idx]
                percentage = (count / len(pixels)) * 100
                
                # Even lower threshold for white-ish colors
                min_white_threshold = max(0.5, min(2.0, 500.0 / len(pixels)))
                
                # Check if color is white-ish (all RGB values > 200)
                if np.all(color > 200) and percentage > min_white_threshold:
                    print(f"  DEBUG: Found white-ish color! RGB{tuple(color)} ({percentage:.1f}%) - using adaptive threshold {min_white_threshold:.1f}%")
                    final_color = tuple(map(int, color))
                    print(f"  DEBUG: USING WHITE-ISH COLOR: RGB{final_color}")
                    return final_color
            
            # PRIORITY 2.5: Check for very bright colors (brightness > 180) with even lower threshold
            print(f"  DEBUG: Checking for any bright-ish colors (brightness > 180)...")
            for i, idx in enumerate(sorted_indices[:20]):
                color = unique_colors[idx]
                count = counts[idx]
                percentage = (count / len(pixels)) * 100
                brightness = np.mean(color)
                
                if brightness > 180 and percentage > 0.1:  # Very low threshold for any bright color
                    print(f"  DEBUG: Found bright-ish color! RGB{tuple(color)} (brightness={brightness:.1f}, {percentage:.1f}%)")
                    final_color = tuple(map(int, color))
                    print(f"  DEBUG: USING BRIGHT-ISH COLOR: RGB{final_color}")
                    return final_color
            
            # Determine if we likely have dark background (for edge detection decision)
            bg_color = np.median(context_region.reshape(-1, 3), axis=0) if context_region.size > 0 else np.array([128, 128, 128])
            bg_brightness = np.mean(bg_color)
            print(f"  DEBUG: Background brightness: {bg_brightness:.1f}")
            
            # PRIORITY 3: Only use edge detection for dark text on light backgrounds
            if bg_brightness > 150:  # Light background - likely dark text
                print(f"  DEBUG: Light background detected, using edge-based detection for dark text...")
                
                gray = cv2.cvtColor(text_rgb, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                
                # Dilate edges to capture text strokes
                kernel = np.ones((3,3), np.uint8)
                dilated_edges = cv2.dilate(edges, kernel, iterations=1)
                
                # Extract colors from edge regions
                edge_pixels = text_rgb[dilated_edges > 0]
                
                print(f"  DEBUG: Edge detection found {len(edge_pixels)} edge pixels")
                
                if len(edge_pixels) > 0:
                    edge_color = np.median(edge_pixels, axis=0)
                    contrast_score = np.linalg.norm(edge_color - bg_color)
                    print(f"  DEBUG: Edge-based color: RGB{tuple(map(int, edge_color))}")
                    print(f"  DEBUG: Contrast score: {contrast_score:.2f}")
                    
                    if contrast_score > 50:  # Good contrast
                        print(f"  DEBUG: Using edge-based color for dark text")
                        final_color = tuple(map(int, edge_color))
                        print(f"  DEBUG: FINAL COLOR: RGB{final_color}")
                        return final_color
            
            # PRIORITY 4: For dark backgrounds, look for the most contrasting color
            print(f"  DEBUG: Dark background detected, looking for most contrasting color...")
            
            max_contrast = 0
            best_color = unique_colors[sorted_indices[0]]  # Default to most frequent
            
            for idx in sorted_indices[:5]:  # Check top 5 colors
                color = unique_colors[idx]
                count = counts[idx]
                percentage = (count / len(pixels)) * 100
                
                if percentage > 5.0:  # Only consider colors with significant presence
                    contrast = np.linalg.norm(color - bg_color)
                    print(f"  DEBUG: Color RGB{tuple(color)} contrast: {contrast:.2f}")
                    
                    if contrast > max_contrast:
                        max_contrast = contrast
                        best_color = color
            
            # SMART FALLBACK: If best contrasting color is still very similar to background, force white
            if max_contrast < 20:  # Very low contrast means we're mostly getting background
                avg_detected_brightness = np.mean([np.mean(unique_colors[idx]) for idx in sorted_indices[:3]])
                print(f"  DEBUG: Very low contrast detected ({max_contrast:.2f}), avg brightness: {avg_detected_brightness:.1f}")
                
                if avg_detected_brightness < 120 and bg_brightness < 120:  # Both dark
                    print(f"  DEBUG: FORCING WHITE COLOR due to dark background and low contrast")
                    final_color = (255, 255, 255)
                    print(f"  DEBUG: FINAL COLOR (FORCED): RGB{final_color}")
                    return final_color
            
            print(f"  DEBUG: Best contrasting color: RGB{tuple(best_color)} (contrast: {max_contrast:.2f})")
            final_color = tuple(map(int, best_color))
            print(f"  DEBUG: FINAL COLOR: RGB{final_color}")
            return final_color
            
        except Exception as e:
            print(f"  DEBUG: Error in advanced color extraction: {e}")
            return (255, 255, 255)
    
    def _extract_background_color_advanced(self, img_array: np.ndarray, x: int, y: int, w: int, h: int) -> tuple:
        """Enhanced background color extraction"""
        try:
            samples = []
            offset = max(5, min(w, h) // 4)
            
            # Sample multiple regions around the text
            sample_regions = [
                # Above text
                (x, y - offset, w, offset//2),
                # Below text  
                (x, y + h, w, offset//2),
                # Left of text
                (x - offset, y, offset//2, h),
                # Right of text
                (x + w, y, offset//2, h),
                # Corners
                (x - offset//2, y - offset//2, offset//2, offset//2),
                (x + w, y - offset//2, offset//2, offset//2),
                (x - offset//2, y + h, offset//2, offset//2),
                (x + w, y + h, offset//2, offset//2)
            ]
            
            for sx, sy, sw, sh in sample_regions:
                # Ensure coordinates are within image bounds
                sx = max(0, min(sx, img_array.shape[1] - 1))
                sy = max(0, min(sy, img_array.shape[0] - 1))
                ex = max(sx + 1, min(sx + sw, img_array.shape[1]))
                ey = max(sy + 1, min(sy + sh, img_array.shape[0]))
                
                if ex > sx and ey > sy:
                    region_samples = img_array[sy:ey, sx:ex].reshape(-1, 3)
                    if len(region_samples) > 0:
                        samples.extend(region_samples)
            
            if samples:
                samples = np.array(samples)
                # Use median for robust estimation
                bg_color = np.median(samples, axis=0)
                return tuple(map(int, bg_color))
            
            return (41, 128, 185)  # Default blue background
            
        except Exception as e:
            print(f"Error extracting background color: {e}")
            return (41, 128, 185)
    
    def _calculate_contrast_ratio(self, text_color: tuple, bg_color: tuple) -> float:
        """Calculate contrast ratio between text and background"""
        try:
            def luminance(color):
                """Calculate relative luminance"""
                rgb = [c / 255.0 for c in color]
                rgb = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
                return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            
            l1 = luminance(text_color)
            l2 = luminance(bg_color)
            
            lighter = max(l1, l2)
            darker = min(l1, l2)
            
            contrast_ratio = (lighter + 0.05) / (darker + 0.05)
            return contrast_ratio
            
        except Exception as e:
            return 4.5  # Default reasonable contrast
    
    def _calculate_text_density(self, text_region: np.ndarray) -> float:
        """Calculate text density in the region"""
        try:
            if text_region.size == 0:
                return 0.3
            
            gray = cv2.cvtColor(text_region, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Calculate ratio of text pixels to total pixels
            text_pixels = np.sum(binary < 128)  # Dark pixels are text
            total_pixels = binary.size
            
            density = text_pixels / total_pixels
            return float(density)
            
        except Exception as e:
            return 0.3
    
    def calculate_optimal_font_size_for_translation(self, original_text: str, translated_text: str, 
                                                   original_bbox: Dict, language: str, base_size: Optional[int] = None) -> int:
        """Simplified translation-aware font size calculation"""
        try:
            original_width = original_bbox['width']
            original_height = original_bbox['height']
            
            # Use provided base size or calculate from height
            if base_size is None:
                base_size = max(8, int(original_height * 0.75))
            
            # Only adjust if there's a significant difference in text length
            length_ratio = 1.0  # Default value in case the condition is not met
            orig_len = len(original_text.strip())
            trans_len = len(translated_text.strip())
            
            if orig_len > 0 and trans_len > 0:
                length_ratio = trans_len / orig_len
                
                # Only adjust for significant differences (>30% change)
                if length_ratio > 1.3:  # Text got much longer
                    adjustment = min(0.9, 1.0 / (length_ratio ** 0.3))  # Gentle reduction
                    adjusted_size = int(base_size * adjustment)
                elif length_ratio < 0.7:  # Text got much shorter
                    adjustment = min(1.2, length_ratio ** 0.3 + 0.8)  # Gentle increase
                    adjusted_size = int(base_size * adjustment)
                else:
                    adjusted_size = base_size  # No significant change
            else:
                adjusted_size = base_size
            
            # Ensure reasonable bounds
            final_size = max(8, min(48, adjusted_size))
            
            if final_size != base_size:
                print(f"Translation size adjustment: {base_size} → {final_size} (ratio: {length_ratio:.2f})")
            
            return final_size
            
        except Exception as e:
            print(f"Error calculating translation size: {e}")
            return base_size or max(12, int(original_bbox['height'] * 0.75))
    
    def _extract_text_color(self, region) -> tuple:
        """Extract text color from region (kept for compatibility)"""
        return self._extract_text_color_advanced(region, region)
    
    def _extract_background_color(self, img_array, x, y, w, h) -> tuple:
        """Extract background color around region (kept for compatibility)"""
        return self._extract_background_color_advanced(img_array, x, y, w, h)
    
    def _appears_bold(self, region) -> bool:
        """Detect if text appears bold (kept for compatibility)"""
        weight = self._detect_font_weight_advanced(region, 16)  # Assume 16px for compatibility
        return weight == 'bold'