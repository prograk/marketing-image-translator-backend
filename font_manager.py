import os
import platform
import requests
import zipfile
from fontTools.ttLib import TTFont
from PIL import ImageFont
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np

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
        """Analyze original font properties from image region"""
        try:
            from PIL import Image
            
            image = Image.open(image_path).convert('RGB')
            img_array = np.array(image)
            
            # Extract region
            region = img_array[y:y+h, x:x+w] if h > 0 and w > 0 else np.array([[[255,255,255]]])
            
            font_props = {
                'estimated_size': max(12, int(h * 0.75)),
                'weight': 'normal',
                'style': 'normal',  # Add missing style key
                'color': self._extract_text_color(region),
                'background': self._extract_background_color(img_array, x, y, w, h)
            }
            
            # Detect bold text (thicker strokes)
            if self._appears_bold(region):
                font_props['weight'] = 'bold'
            
            print(f"Analyzed: size={font_props['estimated_size']}, weight={font_props['weight']}, color={font_props['color']}")
            return font_props
            
        except Exception as e:
            print(f"Error analyzing font: {e}")
            return {
                'estimated_size': 20,
                'weight': 'normal',
                'style': 'normal',  # Add missing style key in fallback too
                'color': (255, 255, 255),
                'background': (41, 128, 185)
            }
    
    def _extract_text_color(self, region) -> tuple:
        """Extract text color from region"""
        try:
            pixels = region.reshape(-1, 3)
            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
            
            # Find most contrasting color against blue background
            bg_color = np.array([41, 128, 185])
            max_contrast = 0
            text_color = (255, 255, 255)
            
            for color, count in zip(unique_colors, counts):
                contrast = np.linalg.norm(color - bg_color)
                if contrast > max_contrast and count > len(pixels) * 0.05:
                    max_contrast = contrast
                    text_color = tuple(map(int, color))
            
            return text_color
        except:
            return (255, 255, 255)
    
    def _extract_background_color(self, img_array, x, y, w, h) -> tuple:
        """Extract background color around region"""
        try:
            samples = []
            offset = 15
            
            # Sample around the region
            for dy in [-offset, offset]:
                for dx in range(-offset, w + offset, 5):
                    sample_x, sample_y = x + dx, y + dy
                    if 0 <= sample_x < img_array.shape[1] and 0 <= sample_y < img_array.shape[0]:
                        samples.append(img_array[sample_y, sample_x])
            
            if samples:
                return tuple(map(int, np.median(samples, axis=0)))
            return (41, 128, 185)
        except:
            return (41, 128, 185)
    
    def _appears_bold(self, region) -> bool:
        """Detect if text appears bold"""
        try:
            gray = np.mean(region, axis=2)
            bg_threshold = np.mean(gray)
            text_pixels = np.sum(gray < bg_threshold * 0.8)
            total_pixels = region.shape[0] * region.shape[1]
            return (text_pixels / total_pixels) > 0.25
        except:
            return False