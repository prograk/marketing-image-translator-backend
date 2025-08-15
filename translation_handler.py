
import requests
import json
from typing import List, Dict, Optional
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

class TranslationHandler:
    def __init__(self, api_key: str = None):
        """
        Initialize Translation Handler with OpenRouter API
        
        Args:
            api_key: OpenRouter API key
        """
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Marketing Image Translator",
            "Content-Type": "application/json"
        }
        
        # Language name mappings
        self.language_names = {
            'hi': 'Hindi',
            'mr': 'Marathi', 
            'ta': 'Tamil',
            'kn': 'Kannada',
            'te': 'Telugu',
            'gu': 'Gujarati',
            'bn': 'Bengali',
            'pa': 'Punjabi',
            'ml': 'Malayalam',
            'ur': 'Urdu'
        }
        
    def create_translation_prompt(self, text: str, target_language: str, context: str = "") -> str:
        """
        Create optimized prompt for translation
        
        Args:
            text: Text to translate
            target_language: Target language code
            context: Additional context about the text
            
        Returns:
            Formatted prompt for the LLM
        """
        lang_name = self.language_names.get(target_language, target_language)
        
        prompt = f"""You are a professional translator specializing in marketing content.
        
Task: Translate the following English text to {lang_name}.

Requirements:
1. Maintain the marketing tone and impact
2. Keep brand names, product names, and trademarks in English
3. Ensure the translation is culturally appropriate
4. Preserve any formatting (like ALL CAPS for emphasis)
5. Keep the translation concise - similar length to original

Original text: "{text}"

{f"Context: {context}" if context else ""}

Provide ONLY the translated text without any explanation or notes.
Translation:"""
        
        return prompt
    
    def clean_translation_text(self, text):
        """Clean translation text by removing quotes and extra formatting"""
        if not text:
            return text
        
        # Remove common quote patterns
        text = text.strip()
        text = text.strip('"').strip("'")
        text = text.strip('""').strip("''")
        text = text.strip('「」').strip('『』')
        
        # Remove any remaining quote patterns
        import re
        text = re.sub(r'^["\'""`´''""]+|["\'""`´''""]+$', '', text)
        
        return text.strip()
    
    async def translate_single(self, 
                               text: str, 
                               target_language: str,
                               model: str = "openai/gpt-3.5-turbo",
                               context: str = "") -> Dict:
        """
        Translate a single text using OpenRouter API
        
        Args:
            text: Text to translate
            target_language: Target language code
            model: Model to use for translation
            context: Additional context
            
        Returns:
            Translation result dictionary
        """
        if not self.api_key:
            return {
                'success': False,
                'error': 'API key not provided',
                'original': text,
                'translated': text
            }
        
        try:
            prompt = self.create_translation_prompt(text, target_language, context)
            
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional translator for marketing content."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,  # Lower temperature for consistent translations
                "max_tokens": 500
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        translated_text = data['choices'][0]['message']['content'].strip()
                        
                        # Clean the translation text
                        translated_text = self.clean_translation_text(translated_text)
                        
                        return {
                            'success': True,
                            'original': text,
                            'translated': translated_text,
                            'language': target_language,
                            'model': model
                        }
                    else:
                        error_text = await response.text()
                        return {
                            'success': False,
                            'error': f"API error: {response.status} - {error_text}",
                            'original': text,
                            'translated': text
                        }
                        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'original': text,
                'translated': text
            }
    
    async def translate_batch(self,
                             texts: List[Dict],
                             target_language: str,
                             model: str = "openai/gpt-3.5-turbo") -> List[Dict]:
        """
        Translate multiple texts in parallel
        
        Args:
            texts: List of text dictionaries with 'id' and 'text'
            target_language: Target language code
            model: Model to use
            
        Returns:
            List of translation results
        """
        tasks = []
        for text_item in texts:
            task = self.translate_single(
                text_item['text'],
                target_language,
                model,
                text_item.get('context', '')
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Add IDs back to results
        for i, result in enumerate(results):
            result['id'] = texts[i]['id']
        
        return results
    
    def translate_sync(self, text: str, target_language: str, model: str = "openai/gpt-3.5-turbo") -> Dict:
        """
        Synchronous translation - COMPLETELY SYNCHRONOUS, NO ASYNC
        """
        if not self.api_key:
            return {
                'success': False,
                'error': 'API key not provided',
                'original': text,
                'translated': text
            }
        
        try:
            prompt = self.create_translation_prompt(text, target_language)
            
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional translator for marketing content."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 500
            }
            
            # Use SYNCHRONOUS requests - NOT async
            import requests  # Make sure requests is imported
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                translated_text = data['choices'][0]['message']['content'].strip()
                
                return {
                    'success': True,
                    'original': text,
                    'translated': translated_text,
                    'language': target_language,
                    'model': model
                }
            else:
                return {
                    'success': False,
                    'error': f"API error: {response.status_code}",
                    'original': text,
                    'translated': text
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'original': text,
                'translated': text
            }

class MockTranslationHandler:
    """
    Mock translation handler for testing without API key
    Uses simple translations for demonstration
    """
    
    def __init__(self):
        # Sample translations for testing
        self.mock_translations = {
            'hi': {
                'Special Offer': 'विशेष ऑफर',
                'Limited Time': 'सीमित समय',
                'Buy Now': 'अभी खरीदें',
                'Save Money': 'पैसे बचाएं',
                'Free Delivery': 'मुफ्त डिलीवरी',
                'Best Quality': 'सर्वोत्तम गुणवत्ता',
                'New Arrival': 'नया आगमन',
                'Discount': 'छूट',
                'Sale': 'सेल'
            },
            'mr': {
                'Special Offer': 'विशेष ऑफर',
                'Limited Time': 'मर्यादित वेळ',
                'Buy Now': 'आता खरेदी करा',
                'Save Money': 'पैसे वाचवा',
                'Free Delivery': 'मोफत वितरण'
            },
            'ta': {
                'Special Offer': 'சிறப்பு சலுகை',
                'Limited Time': 'குறைந்த நேரம்',
                'Buy Now': 'இப்போது வாங்கவும்',
                'Save Money': 'பணத்தை சேமிக்கவும்'
            }
        }
    
    def translate_sync(self, text: str, target_language: str, model: str = None) -> Dict:
        """
        Mock translation for testing
        
        Args:
            text: Text to translate
            target_language: Target language code
            model: Ignored in mock
            
        Returns:
            Mock translation result
        """
        # Check if we have a mock translation
        translations = self.mock_translations.get(target_language, {})
        translated = translations.get(text, None)
        
        if translated:
            return {
                'success': True,
                'original': text,
                'translated': translated,
                'language': target_language,
                'model': 'mock'
            }
        else:
            # Return a simple transliteration for demo
            # In real scenario, this would be actual translation
            return {
                'success': True,
                'original': text,
                'translated': f"[{target_language}] {text}",
                'language': target_language,
                'model': 'mock'
            }
    
    async def translate_single(self, text: str, target_language: str, model: str = None, context: str = "") -> Dict:
        """Async wrapper for mock translation"""
        return self.translate_sync(text, target_language, model)
    
    async def translate_batch(self, texts: List[Dict], target_language: str, model: str = None) -> List[Dict]:
        """Mock batch translation"""
        results = []
        for text_item in texts:
            result = self.translate_sync(text_item['text'], target_language, model)
            result['id'] = text_item['id']
            results.append(result)
        return results


# Test function
def test_translation():
    """Test translation functionality"""
    
    # Use mock handler for testing
    handler = MockTranslationHandler()
    
    # Test single translation
    result = handler.translate_sync("Special Offer", "hi")
    print(f"Translation result: {result}")
    
    # Test with API key (if available)
    api_key = input("Enter OpenRouter API key (or press Enter to skip): ").strip()
    if api_key:
        real_handler = TranslationHandler(api_key)
        result = real_handler.translate_sync("Special Offer on all products", "hi")
        print(f"Real translation: {result}")


if __name__ == "__main__":
    test_translation()