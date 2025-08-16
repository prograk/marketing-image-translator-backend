import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from typing import Optional, Dict, List
from font_manager import FontManager
from ocr_handler import OCRHandler
from translation_handler import TranslationHandler

class ImageProcessor:
    def __init__(self):
        self.font_manager = FontManager()
        self.ocr_handler = OCRHandler()
        self.translation_handler = TranslationHandler()
    
    def process_image(self, image_path: str, translations, target_language: str, text_regions, output_path: str) -> bool:
        """Process image with enhanced OCR, translation, and consistent text replacement"""
        try:
            print(f"\n=== Enhanced Image Processing with Consistency ===")
            print(f"Input: {image_path}")
            print(f"Output: {output_path}")
            print(f"Target Language: {target_language}")
            print(f"Processing image with {len(translations)} translations...")
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(image)
            
            # Use provided text regions or extract them
            if text_regions is None:
                text_regions = self.ocr_handler.extract_text(image_path)
            
            if not text_regions:
                print("No text regions found")
                return False
            
            # Create a mapping of region IDs to translations
            translation_map = {}
            if isinstance(translations, dict):
                translation_map = translations
            elif isinstance(target_language, dict):
                # Parameters are swapped - target_language contains translations
                translation_map = target_language
                target_language = 'hi'  # Set actual target language
            
            # Handle swapped parameters - if translations is a list, it's actually text_regions
            if isinstance(translations, list) and not isinstance(text_regions, list):
                # Swap them
                translations, text_regions = text_regions, translations
            
            # Group text regions for consistent styling
            text_groups = self._group_text_regions(text_regions)
            print(f"Grouped {len(text_regions)} regions into {len(text_groups)} groups")
            
            # Process each group for consistency
            for group_id, group in text_groups.items():
                print(f"\nProcessing group {group_id} with {len(group['regions'])} regions")
                
                # Calculate consistent styling for the group
                group_style = self._calculate_group_style(group, image_path)
                
                # Process each region in the group
                for region in group['regions']:
                    if isinstance(region, str):
                        continue
                    
                    if not region.get('selected_for_translation', False):
                        continue
                    
                    region_id = region['id']
                    original_text = region['text'].strip()
                    
                    if len(original_text) < 2:
                        continue
                    
                    # Get translation
                    translated_text = translation_map.get(region_id, original_text)
                    if isinstance(translated_text, str):
                        if translated_text.startswith('"') and translated_text.endswith('"'):
                            translated_text = translated_text[1:-1]
                    else:
                        translated_text = original_text
                    
                    print(f"  Processing region {region_id}: '{original_text}' → '{translated_text}'")
                    
                    # Get region coordinates
                    bbox = region['bbox']
                    x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                    
                    # Use group style with minor individual adjustments
                    font_props = self._get_individual_font_props(group_style, region, image_path, x, y, w, h)
                    
                    # Calculate size with group consistency
                    optimal_size = self._calculate_consistent_size(
                        original_text, translated_text, bbox, group_style, target_language
                    )
                    
                    # Get font
                    font = self.font_manager.find_best_font(
                        target_language,
                        optimal_size,
                        font_props['weight'],
                        font_props['style']
                    )
                    
                    # Clear and render
                    image = self._clear_text_region_inpaint(image, x, y, w, h)
                    draw = ImageDraw.Draw(image)
                    
                    self._render_text_with_consistent_styling(
                        draw, translated_text, font, x, y, w, h,
                        font_props, target_language, group_style
                    )
            
            # Save processed image
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            dpi = image.info.get('dpi', (300, 300))
            image.save(output_path, format='PNG', optimize=True, compress_level=1, dpi=dpi)
            print(f"✓ Image processed successfully: {output_path}")
            return True
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            print("✗ Failed to process image")
            return False
    
    def _group_text_regions(self, text_regions: List[Dict]) -> Dict[str, Dict]:
        """Group text regions by similarity for consistent styling"""
        try:
            groups = {}
            
            for region in text_regions:
                if isinstance(region, str) or not region.get('selected_for_translation', False):
                    continue
                
                bbox = region['bbox']
                text = region['text'].strip()
                
                # Determine group type
                group_key = self._determine_group_key(region, bbox, text)
                
                if group_key not in groups:
                    groups[group_key] = {
                        'type': group_key.split('_')[0],
                        'regions': [],
                        'avg_height': 0,
                        'avg_width': 0,
                        'sample_region': region
                    }
                
                groups[group_key]['regions'].append(region)
            
            # Calculate group averages
            for group_id, group in groups.items():
                heights = [r['bbox']['height'] for r in group['regions']]
                widths = [r['bbox']['width'] for r in group['regions']]
                group['avg_height'] = int(np.mean(heights))
                group['avg_width'] = int(np.mean(widths))
            
            return groups
            
        except Exception as e:
            print(f"Error grouping regions: {e}")
            # Fallback: put everything in one group
            return {'default_group': {
                'type': 'mixed',
                'regions': [r for r in text_regions if isinstance(r, dict)],
                'avg_height': 20,
                'avg_width': 100,
                'sample_region': text_regions[0] if text_regions else None
            }}
    
    def _determine_group_key(self, region: Dict, bbox: Dict, text: str) -> str:
        """Determine which group a text region belongs to"""
        try:
            height = bbox['height']
            width = bbox['width']
            y_pos = bbox['y']
            
            # Check if it's likely a list item (positioned similarly vertically)
            if self._is_likely_list_item(region, bbox, text):
                # Group by approximate height and position
                height_bucket = (height // 5) * 5  # Round to nearest 5
                return f"list_{height_bucket}"
            
            # Check if it's a heading (larger text, positioned differently)
            elif self._is_likely_heading(region, bbox, text):
                height_bucket = (height // 10) * 10  # Round to nearest 10
                return f"heading_{height_bucket}"
            
            # Default grouping by size
            else:
                height_bucket = (height // 8) * 8  # Round to nearest 8
                return f"text_{height_bucket}"
                
        except Exception as e:
            return "default"
    
    def _is_likely_list_item(self, region: Dict, bbox: Dict, text: str) -> bool:
        """Detect if text region is likely a list item"""
        try:
            # Check for bullet-like characteristics
            if len(text.split()) <= 6:  # Short text
                return True
            
            # Check if text looks like a service/feature
            list_indicators = ['डिज़ाइन', 'सेवा', 'Service', 'Design', '•', '◦', '-']
            if any(indicator in text for indicator in list_indicators):
                return True
            
            return False
            
        except Exception as e:
            return True  # Default to list item for consistency
    
    def _is_likely_heading(self, region: Dict, bbox: Dict, text: str) -> bool:
        """Detect if text region is likely a heading"""
        try:
            height = bbox['height']
            
            # Large text is likely heading
            if height > 40:
                return True
            
            # Check for heading-like text
            if text.isupper() and len(text.split()) <= 4:
                return True
            
            # Check for title words
            title_words = ['Services', 'Design', 'Creative', 'Solutions']
            if any(word in text for word in title_words):
                return True
            
            return False
            
        except Exception as e:
            return False
    
    def _calculate_group_style(self, group: Dict, image_path: str) -> Dict:
        """Calculate consistent style for a group of text regions"""
        try:
            regions = group['regions']
            if not regions:
                return self._get_default_style()
            
            # Use the first region as reference for detailed analysis
            sample_region = regions[0]
            bbox = sample_region['bbox']
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            
            # Get detailed analysis from sample
            sample_style = self.font_manager.analyze_original_font(
                image_path, x, y, w, h, sample_region['text']
            )
            
            # Calculate consistent size for the group
            heights = [r['bbox']['height'] for r in regions]
            consistent_height = int(np.median(heights))  # Use median for robustness
            
            # Base the group size on the median height
            group_base_size = max(8, int(consistent_height * 0.75))
            
            group_style = {
                'base_size': group_base_size,
                'weight': sample_style['weight'],
                'style': sample_style['style'],
                'color': sample_style['color'],
                'background': sample_style['background'],
                'contrast_ratio': sample_style.get('contrast_ratio', 4.5),
                'group_type': group['type'],
                'avg_height': consistent_height
            }
            
            print(f"Group style: type={group['type']}, base_size={group_base_size}, "
                  f"weight={sample_style['weight']}, regions={len(regions)}")
            
            return group_style
            
        except Exception as e:
            print(f"Error calculating group style: {e}")
            return self._get_default_style()
    
    def _get_default_style(self) -> Dict:
        """Get default style for fallback"""
        return {
            'base_size': 16,
            'weight': 'normal',
            'style': 'normal',
            'color': (255, 255, 255),
            'background': (41, 128, 185),
            'contrast_ratio': 4.5,
            'group_type': 'default',
            'avg_height': 20
        }
    
    def _get_individual_font_props(self, group_style: Dict, region: Dict, image_path: str, 
                                  x: int, y: int, w: int, h: int) -> Dict:
        """Get individual font properties while maintaining group consistency"""
        try:
            # For most properties, use group style
            font_props = {
                'estimated_size': group_style['base_size'],
                'weight': group_style['weight'],
                'style': group_style['style'],
                'color': group_style['color'],
                'background': group_style['background'],
                'contrast_ratio': group_style['contrast_ratio']
            }
            
            # Only analyze individual color if needed (for variation within group)
            if group_style['group_type'] in ['heading', 'mixed']:
                individual_style = self.font_manager.analyze_original_font(image_path, x, y, w, h, region['text'])
                font_props['color'] = individual_style['color']  # Allow color variation for headings
            
            return font_props
            
        except Exception as e:
            return group_style
    
    def _calculate_consistent_size(self, original_text: str, translated_text: str, 
                                 bbox: Dict, group_style: Dict, language: str) -> int:
        """Calculate size with group consistency"""
        try:
            base_size = group_style['base_size']
            
            # For list items, maintain very consistent sizing
            if group_style['group_type'] == 'list':
                # Only minor adjustments for translation length
                adjusted_size = self.font_manager.calculate_optimal_font_size_for_translation(
                    original_text, translated_text, bbox, language, base_size
                )
                # Limit adjustment to ±15% for consistency
                min_size = int(base_size * 0.85)
                max_size = int(base_size * 1.15)
                return max(min_size, min(max_size, adjusted_size))
            
            # For headings, allow more flexibility
            elif group_style['group_type'] == 'heading':
                return self.font_manager.calculate_optimal_font_size_for_translation(
                    original_text, translated_text, bbox, language, base_size
                )
            
            # For other text, moderate consistency
            else:
                adjusted_size = self.font_manager.calculate_optimal_font_size_for_translation(
                    original_text, translated_text, bbox, language, base_size
                )
                # Limit adjustment to ±25%
                min_size = int(base_size * 0.75)
                max_size = int(base_size * 1.25)
                return max(min_size, min(max_size, adjusted_size))
                
        except Exception as e:
            return group_style['base_size']
        """
        Enhanced text removal using OpenCV inpainting with better masking
        """
        try:
            # Convert PIL image to OpenCV format (numpy array, BGR color)
            img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # Create enhanced mask with slight padding
            mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
            padding = max(1, min(w, h) // 20)  # Adaptive padding
            
            # Ensure coordinates are within bounds
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(img_bgr.shape[1], x + w + padding)
            y_end = min(img_bgr.shape[0], y + h + padding)
            
            mask[y_start:y_end, x_start:x_end] = 255

            # Use adaptive inpaint radius based on text size
            inpaint_radius = max(3, min(7, min(w, h) // 10))
            
            # Perform inpainting using TELEA algorithm (better for text)
            img_inpainted = cv2.inpaint(img_bgr, mask, inpaint_radius, cv2.INPAINT_TELEA)

            # Convert back to PIL (RGB)
            img_rgb = cv2.cvtColor(img_inpainted, cv2.COLOR_BGR2RGB)
            pil_img_out = Image.fromarray(img_rgb)
            return pil_img_out
            
        except Exception as e:
            print(f"Error in inpainting: {e}")
            # Fallback to simple rectangle fill
            draw = ImageDraw.Draw(pil_image)
            self._clear_text_region(draw, x, y, w, h, (41, 128, 185))
            return pil_image
    
    def _clear_text_region(self, draw: ImageDraw.Draw, x: int, y: int, w: int, h: int, bg_color: tuple):
        """Clear the original text region (fallback method)"""
        # Add some padding to ensure complete clearing
        padding = 2
        draw.rectangle([
            x - padding, y - padding, 
            x + w + padding, y + h + padding
        ], fill=bg_color)
    
    def _render_text_with_enhanced_styling(self, draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont,
                                         x: int, y: int, w: int, h: int, font_props: dict, 
                                         language: str, original_text: str):
        """
        Enhanced text rendering with better style matching and adaptive fitting
        """
        try:
            color = font_props['color']
            padding = max(2, min(w, h) // 20)  # Adaptive padding
            max_width = w - 2 * padding
            max_height = h - 2 * padding
            
            # Smart text wrapping with line optimization
            lines = self._smart_text_wrapping(text, font, max_width, max_height)
            
            if not lines:
                lines = [text]
            
            # Calculate line metrics
            line_height = self._get_enhanced_line_height(font, language)
            total_text_height = len(lines) * line_height
            
            # Adaptive font size adjustment if text doesn't fit
            if total_text_height > max_height or any(self.font_manager.get_text_dimensions(line, font)[0] > max_width for line in lines):
                adjusted_font = self._adjust_font_for_fit(text, font, max_width, max_height, language, lines)
                if adjusted_font:
                    font = adjusted_font
                    lines = self._smart_text_wrapping(text, font, max_width, max_height)
                    line_height = self._get_enhanced_line_height(font, language)
                    total_text_height = len(lines) * line_height
            
            # Determine text alignment based on original text characteristics
            text_align = self._detect_text_alignment(original_text, w, h)
            vertical_align = self._detect_vertical_alignment(font_props.get('text_density', 0.3))
            
            # Enhanced positioning with proper baseline alignment
            start_y = self._calculate_start_y(y, h, total_text_height, vertical_align, padding)
            
            # Render each line with enhanced styling
            for i, line in enumerate(lines):
                line_y = start_y + i * line_height
                
                # Calculate horizontal position
                line_x = self._calculate_line_x(x, w, line, font, text_align, padding)
                
                # Apply enhanced text effects if needed
                if font_props.get('contrast_ratio', 4.5) < 3.0:
                    # Add text outline for better readability
                    self._draw_text_with_outline(draw, line_x, line_y, line, font, color)
                else:
                    # Regular text rendering
                    draw.text((line_x, line_y), line, font=font, fill=color)
                
                print(f"Rendered line {i+1}: '{line}' at ({line_x}, {line_y})")
            
        except Exception as e:
            print(f"Error in enhanced text rendering: {e}")
            # Fallback to simple rendering
            self._render_text_with_wrapping(draw, text, font, x, y, w, h, color, language)
    
    def _smart_text_wrapping(self, text: str, font: ImageFont.FreeTypeFont, max_width: int, max_height: int) -> List[str]:
        """Smart text wrapping that optimizes line breaks for better visual balance"""
        try:
            words = text.split()
            if len(words) <= 1:
                return [text]
            
            # Try different line break strategies
            strategies = [
                self._wrap_balanced(words, font, max_width),
                self._wrap_greedy(words, font, max_width),
                self._wrap_even_split(words, font, max_width)
            ]
            
            # Choose the best strategy based on visual balance
            best_lines = []
            best_score = float('-inf')
            
            for lines in strategies:
                if lines:
                    score = self._evaluate_line_balance(lines, font, max_width)
                    if score > best_score:
                        best_score = score
                        best_lines = lines
            
            return best_lines or [text]
            
        except Exception as e:
            print(f"Error in smart wrapping: {e}")
            return self.font_manager.wrap_text(text, font, max_width)
    
    def _wrap_balanced(self, words: List[str], font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
        """Balanced line wrapping for even distribution"""
        if len(words) <= 2:
            return [' '.join(words)]
        
        # Try to split into roughly equal parts
        mid = len(words) // 2
        line1 = ' '.join(words[:mid])
        line2 = ' '.join(words[mid:])
        
        width1, _ = self.font_manager.get_text_dimensions(line1, font)
        width2, _ = self.font_manager.get_text_dimensions(line2, font)
        
        if width1 <= max_width and width2 <= max_width:
            return [line1, line2]
        
        return self.font_manager.wrap_text(' '.join(words), font, max_width)
    
    def _wrap_greedy(self, words: List[str], font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
        """Standard greedy wrapping"""
        return self.font_manager.wrap_text(' '.join(words), font, max_width)
    
    def _wrap_even_split(self, words: List[str], font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
        """Even character-based split for very long words"""
        text = ' '.join(words)
        if len(text) <= 20:
            return [text]
        
        # Split roughly in the middle
        mid = len(text) // 2
        # Find nearest space
        split_pos = mid
        for i in range(max(0, mid-10), min(len(text), mid+10)):
            if text[i] == ' ':
                split_pos = i
                break
        
        line1 = text[:split_pos].strip()
        line2 = text[split_pos:].strip()
        
        if line1 and line2:
            width1, _ = self.font_manager.get_text_dimensions(line1, font)
            width2, _ = self.font_manager.get_text_dimensions(line2, font)
            
            if width1 <= max_width and width2 <= max_width:
                return [line1, line2]
        
        return [text]
    
    def _evaluate_line_balance(self, lines: List[str], font: ImageFont.FreeTypeFont, max_width: int) -> float:
        """Evaluate visual balance of line breaks"""
        try:
            if not lines:
                return 0
            
            widths = [self.font_manager.get_text_dimensions(line, font)[0] for line in lines]
            
            # Penalize lines that are too different in length
            width_variance = np.var(widths) if len(widths) > 1 else 0
            
            # Reward good space utilization
            avg_utilization = np.mean([w / max_width for w in widths])
            
            # Penalize too many lines
            line_count_penalty = len(lines) * 0.1
            
            score = avg_utilization - (width_variance / 10000) - line_count_penalty
            return score
            
        except Exception as e:
            return 0
    
    def _adjust_font_for_fit(self, text: str, font: ImageFont.FreeTypeFont, max_width: int, 
                           max_height: int, language: str, lines: List[str]) -> Optional[ImageFont.FreeTypeFont]:
        """Adjust font size to fit text in available space"""
        try:
            current_size = font.size
            min_size = max(8, current_size // 2)
            
            # Binary search for optimal size
            low, high = min_size, current_size
            best_font = None
            
            while low <= high:
                test_size = (low + high) // 2
                test_font = self.font_manager.find_best_font(language, test_size)
                
                # Test if text fits
                test_lines = self._smart_text_wrapping(text, test_font, max_width, max_height)
                line_height = self._get_enhanced_line_height(test_font, language)
                total_height = len(test_lines) * line_height
                
                fits_width = all(self.font_manager.get_text_dimensions(line, test_font)[0] <= max_width for line in test_lines)
                fits_height = total_height <= max_height
                
                if fits_width and fits_height:
                    best_font = test_font
                    low = test_size + 1  # Try larger
                else:
                    high = test_size - 1  # Try smaller
            
            if best_font:
                print(f"Adjusted font size: {current_size} → {best_font.size}")
            
            return best_font
            
        except Exception as e:
            print(f"Error adjusting font: {e}")
            return None
    
    def _detect_text_alignment(self, original_text: str, width: int, height: int) -> str:
        """Detect likely text alignment based on text characteristics"""
        try:
            # Rules for alignment detection
            if original_text.isupper() and len(original_text.split()) <= 3:
                return 'center'  # Titles/headers are usually centered
            
            if width > height * 3:  # Very wide text region
                return 'left'  # Long text is usually left-aligned
            
            if len(original_text) < 15:  # Short text
                return 'center'  # Short text is often centered
            
            return 'center'  # Default to center for most cases
            
        except Exception as e:
            return 'center'
    
    def _detect_vertical_alignment(self, text_density: float) -> str:
        """Detect vertical alignment based on text density"""
        try:
            if text_density > 0.4:
                return 'center'  # Dense text usually centered
            elif text_density < 0.2:
                return 'top'  # Sparse text might be top-aligned
            else:
                return 'center'  # Default
                
        except Exception as e:
            return 'center'
    
    def _calculate_start_y(self, y: int, h: int, total_text_height: int, vertical_align: str, padding: int) -> int:
        """Calculate vertical start position with proper alignment"""
        if vertical_align == 'top':
            return y + padding
        elif vertical_align == 'bottom':
            return y + h - total_text_height - padding
        else:  # center
            return y + (h - total_text_height) // 2
    
    def _calculate_line_x(self, x: int, w: int, line: str, font: ImageFont.FreeTypeFont, 
                         text_align: str, padding: int) -> int:
        """Calculate horizontal position for a line"""
        try:
            line_width, _ = self.font_manager.get_text_dimensions(line, font)
            
            if text_align == 'left':
                return x + padding
            elif text_align == 'right':
                return x + w - line_width - padding
            else:  # center
                return x + (w - line_width) // 2
                
        except Exception as e:
            return x + padding
    
    def _draw_text_with_outline(self, draw: ImageDraw.Draw, x: int, y: int, text: str, 
                               font: ImageFont.FreeTypeFont, color: tuple):
        """Draw text with outline for better readability"""
        try:
            # Calculate outline color (opposite brightness)
            avg_brightness = sum(color) / 3
            outline_color = (255, 255, 255) if avg_brightness < 128 else (0, 0, 0)
            
            # Draw outline
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
            
            # Draw main text
            draw.text((x, y), text, font=font, fill=color)
            
        except Exception as e:
            # Fallback to regular text
            draw.text((x, y), text, font=font, fill=color)
    
    def _get_enhanced_line_height(self, font: ImageFont.FreeTypeFont, language: str) -> int:
        """Get enhanced line height based on language characteristics"""
        try:
            # Use language-specific test characters
            test_chars = {
                'hi': 'हिंग्लीy',
                'ta': 'தமிழ்y', 
                'te': 'తెలుగుy',
                'kn': 'ಕನ್ನಡy',
                'bn': 'বাংলাy'
            }
            
            test_text = test_chars.get(language.lower(), 'Ayg')
            bbox = font.getbbox(test_text)
            height = bbox[3] - bbox[1]
            
            # Add language-specific spacing
            language_multipliers = {
                'hi': 1.3,  # Devanagari needs more space
                'ta': 1.25, # Tamil needs some extra space
                'te': 1.3,  # Telugu needs more space
                'kn': 1.25, # Kannada needs some extra space
                'bn': 1.3   # Bengali needs more space
            }
            
            multiplier = language_multipliers.get(language.lower(), 1.2)
            return int(height * multiplier)
            
        except Exception as e:
            # Fallback
            return int(font.size * 1.2)
    
    def _render_text_with_wrapping(self, draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont,
                                x: int, y: int, w: int, h: int, color: tuple, language: str,
                                text_align: str = 'center', vertical_align: str = 'center',
                                padding: int = 5):
        """
        Original text rendering method (kept for compatibility)
        """
        try:
            max_width = w - 2 * padding
            lines = self.font_manager.wrap_text(text, font, max_width)
            if not lines:
                lines = [text]

            line_height = self._get_line_height(font)
            total_text_height = len(lines) * line_height

            # Adjust font size if too tall (optional, as you have)
            if total_text_height > h and len(lines) > 1:
                new_size = max(8, int(font.size * 0.8))
                font = self.font_manager.find_best_font(language, new_size)
                lines = self.font_manager.wrap_text(text, font, max_width)
                line_height = self._get_line_height(font)
                total_text_height = len(lines) * line_height

            # Vertical start position
            if vertical_align == 'top':
                start_y = y + padding
            elif vertical_align == 'bottom':
                start_y = y + h - total_text_height - padding
            else:  # center
                start_y = y + (h - total_text_height) // 2

            for i, line in enumerate(lines):
                line_y = start_y + i * line_height

                line_width, _ = self.font_manager.get_text_dimensions(line, font)

                # Horizontal position
                if text_align == 'left':
                    line_x = x + padding
                elif text_align == 'right':
                    line_x = x + w - line_width - padding
                else:  # center
                    line_x = x + (w - line_width) // 2

                # Clamp to bbox
                line_x = max(x + padding, min(line_x, x + w - line_width - padding))
                line_y = max(y + padding, min(line_y, y + h - line_height - padding))

                draw.text((line_x, line_y), line, font=font, fill=color)

        except Exception as e:
            print(f"Error rendering text alignment: {e}")
            # fallback single line center
            try:
                text_width, text_height = self.font_manager.get_text_dimensions(text, font)
                text_x = x + (w - text_width) // 2
                text_y = y + (h - text_height) // 2
                draw.text((text_x, text_y), text, font=font, fill=color)
            except:
                pass

    def _get_line_height(self, font: ImageFont.FreeTypeFont) -> int:
        """Get line height for font with proper spacing (kept for compatibility)"""
        try:
            # Use a sample character to get height
            bbox = font.getbbox("Ag")
            height = bbox[3] - bbox[1]
            # Add some line spacing (20% of font height)
            return int(height * 1.2)
        except:
            # Fallback to font size
            return int(font.size * 1.2)
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages"""
        return list(self.font_manager.language_fonts.keys())
    
    def _get_consistent_line_height(self, font: ImageFont.FreeTypeFont, language: str, group_style: Dict) -> int:
        """Get consistent line height for group"""
        try:
            # Base line height from font
            base_height = self._get_enhanced_line_height(font, language)
            
            # Adjust based on group type
            if group_style['group_type'] == 'list':
                # Slightly tighter spacing for lists
                return int(base_height * 0.95)
            elif group_style['group_type'] == 'heading':
                # More generous spacing for headings
                return int(base_height * 1.1)
            else:
                return base_height
                
        except Exception as e:
            return int(font.size * 1.2)
    
    def _clear_text_region_inpaint(self, pil_image, x, y, w, h):
        """
        Enhanced text removal using OpenCV inpainting with better masking
        """
        try:
            # Convert PIL image to OpenCV format (numpy array, BGR color)
            img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # Create enhanced mask with slight padding
            mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
            padding = max(1, min(w, h) // 20)  # Adaptive padding
            
            # Ensure coordinates are within bounds
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(img_bgr.shape[1], x + w + padding)
            y_end = min(img_bgr.shape[0], y + h + padding)
            
            mask[y_start:y_end, x_start:x_end] = 255

            # Use adaptive inpaint radius based on text size
            inpaint_radius = max(3, min(7, min(w, h) // 10))
            
            # Perform inpainting using TELEA algorithm (better for text)
            img_inpainted = cv2.inpaint(img_bgr, mask, inpaint_radius, cv2.INPAINT_TELEA)

            # Convert back to PIL (RGB)
            img_rgb = cv2.cvtColor(img_inpainted, cv2.COLOR_BGR2RGB)
            pil_img_out = Image.fromarray(img_rgb)
            return pil_img_out
            
        except Exception as e:
            print(f"Error in inpainting: {e}")
            # Fallback to simple rectangle fill
            draw = ImageDraw.Draw(pil_image)
            self._clear_text_region(draw, x, y, w, h, (41, 128, 185))
            return pil_image
    
    def _clear_text_region(self, draw: ImageDraw.Draw, x: int, y: int, w: int, h: int, bg_color: tuple):
        """Clear the original text region (fallback method)"""
        # Add some padding to ensure complete clearing
        padding = 2
        draw.rectangle([
            x - padding, y - padding, 
            x + w + padding, y + h + padding
        ], fill=bg_color)
    
    def _render_text_with_consistent_styling(self, draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont,
                                           x: int, y: int, w: int, h: int, font_props: dict, 
                                           language: str, group_style: Dict):
        """
        Render text with group-consistent styling but individual positioning
        """
        try:
            color = font_props['color']
            padding = max(2, min(w, h) // 25)  # Consistent padding
            max_width = w - 2 * padding
            max_height = h - 2 * padding
            
            # Simplified text wrapping for consistency
            lines = self.font_manager.wrap_text(text, font, max_width)
            if not lines:
                lines = [text]
            
            # Use consistent line height calculation
            line_height = self._get_consistent_line_height(font, language, group_style)
            total_text_height = len(lines) * line_height
            
            # Consistent alignment based on group type
            if group_style['group_type'] == 'list':
                text_align = 'left'
                vertical_align = 'center'
            elif group_style['group_type'] == 'heading':
                text_align = 'center'
                vertical_align = 'center'
            else:
                text_align = 'center'
                vertical_align = 'center'
            
            # Calculate start position
            start_y = self._calculate_start_y(y, h, total_text_height, vertical_align, padding)
            
            # Render each line consistently
            for i, line in enumerate(lines):
                line_y = start_y + i * line_height
                line_x = self._calculate_line_x(x, w, line, font, text_align, padding)
                
                # Ensure line stays within bounds
                line_x = max(x + padding, min(line_x, x + w - padding - 10))
                line_y = max(y + padding, min(line_y, y + h - padding - 10))
                
                # Apply consistent rendering
                if font_props.get('contrast_ratio', 4.5) < 2.5:
                    self._draw_text_with_outline(draw, line_x, line_y, line, font, color)
                else:
                    draw.text((line_x, line_y), line, font=font, fill=color)
            
        except Exception as e:
            print(f"Error in consistent rendering: {e}")
            # Fallback to simple center rendering
            try:
                text_width, text_height = self.font_manager.get_text_dimensions(text, font)
                text_x = x + (w - text_width) // 2
                text_y = y + (h - text_height) // 2
                draw.text((text_x, text_y), text, font=font, fill=color)
            except:
                pass