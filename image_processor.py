import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from font_manager import FontManager
from ocr_handler import OCRHandler
from translation_handler import TranslationHandler

class ImageProcessor:
    def __init__(self):
        self.font_manager = FontManager()
        self.ocr_handler = OCRHandler()
        self.translation_handler = TranslationHandler()
    
    def process_image(self, image_path: str, translations, target_language: str, text_regions, output_path: str) -> bool:
        """Process image with OCR, translation, and text replacement"""
        try:
            print(f"\n=== Image Processing with FontTools ===")
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
            
            # Debug: Print what we received
            print(f"DEBUG: translations type: {type(translations)}")
            print(f"DEBUG: target_language type: {type(target_language)}")
            print(f"DEBUG: translation_map: {translation_map}")
            print(f"DEBUG: text_regions count: {len(text_regions) if isinstance(text_regions, list) else 'Not a list'}")
            
            # Handle swapped parameters - if translations is a list, it's actually text_regions
            if isinstance(translations, list) and not isinstance(text_regions, list):
                # Swap them
                translations, text_regions = text_regions, translations
            
            # Process each text region
            for i, region in enumerate(text_regions):
                # Handle case where text_regions might be a list of strings or dictionaries
                if isinstance(region, str):
                    # Skip string entries, we need dictionary objects
                    continue
                
                # Skip if not selected for translation
                if not region.get('selected_for_translation', False):
                    continue
                
                region_id = region['id']
                original_text = region['text'].strip()
                
                if len(original_text) < 2:  # Skip very short text
                    continue
                
                # Get translation for this region
                translated_text = translation_map.get(region_id, original_text)
                if isinstance(translated_text, str):
                    if translated_text.startswith('"') and translated_text.endswith('"'):
                        translated_text = translated_text[1:-1]  # Remove quotes
                else:
                    translated_text = original_text  # Fallback
                
                print(f"\nProcessing region {region_id}: '{original_text}' → '{translated_text}'")
                
                # Get region coordinates from bbox
                bbox = region['bbox']
                x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                
                # Analyze original font properties
                font_props = self.font_manager.analyze_original_font(image_path, x, y, w, h, original_text)
                
                # Get appropriate font
                font = self.font_manager.find_best_font(
                    target_language,
                    font_props['estimated_size'],
                    font_props['weight'],
                    font_props['style']
                )
                
                # Clear original text area
                # self._clear_text_region(draw, x, y, w, h, font_props['background'])
                image = self._clear_text_region_inpaint(image, x, y, w, h)
                draw = ImageDraw.Draw(image)

                # Render new text with proper positioning
                self._render_text_with_wrapping(
                    draw, translated_text, font, x, y, w, h,
                    font_props['color'], target_language,
                    text_align='left',            # or 'center', 'right' as per detected style
                    vertical_align='top',         # or 'center', 'bottom'
                    padding=5                     # adjust as needed
                )
            
            # Save processed image
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            dpi = image.info.get('dpi', (300, 300))  # Use original or default 300 dpi
            image.save(output_path, format='PNG', optimize=True, compress_level=1, dpi=dpi)
            print(f"✓ Image processed successfully: {output_path}")
            return True
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            print("✗ Failed to process image")
            return False
    
    def _clear_text_region_inpaint(self, pil_image, x, y, w, h):
        """
        Removes text from the specified region using OpenCV inpainting.
        Args:
            pil_image (PIL.Image.Image): The full image (RGB).
            x, y, w, h (int): Bounding box of the text region.
        Returns:
            PIL.Image.Image: Image with text region inpainted.
        """
        # Convert PIL image to OpenCV format (numpy array, BGR color)
        img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Create mask
        mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        mask[y:y+h, x:x+w] = 255  # Mask region to inpaint

        # Perform inpainting
        inpaint_radius = 3  # 3-5 works well for text
        img_inpainted = cv2.inpaint(img_bgr, mask, inpaint_radius, cv2.INPAINT_TELEA)

        # Convert back to PIL (RGB)
        img_rgb = cv2.cvtColor(img_inpainted, cv2.COLOR_BGR2RGB)
        pil_img_out = Image.fromarray(img_rgb)
        return pil_img_out
    
    def _clear_text_region(self, draw: ImageDraw.Draw, x: int, y: int, w: int, h: int, bg_color: tuple):
        """Clear the original text region"""
        # Add some padding to ensure complete clearing
        padding = 2
        draw.rectangle([
            x - padding, y - padding, 
            x + w + padding, y + h + padding
        ], fill=bg_color)
    
    def _render_text_with_wrapping(self, draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont,
                                x: int, y: int, w: int, h: int, color: tuple, language: str,
                                text_align: str = 'center', vertical_align: str = 'center',
                                padding: int = 5):
        """
        Render text with wrapping and positioning aligned inside bbox.

        text_align options: 'left', 'center', 'right'  
        vertical_align options: 'top', 'center', 'bottom'
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
        """Get line height for font with proper spacing"""
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