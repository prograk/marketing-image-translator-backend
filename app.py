from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import os
import shutil
import json
import uuid
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import logging

from translation_handler import TranslationHandler, MockTranslationHandler
from image_processor import ImageProcessor

# Import our OCR handler
from ocr_handler import OCRHandler

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Marketing Image Translator",
    description="Translate text in marketing images while preserving brand elements",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
os.makedirs("temp", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Initialize OCR handler (load once to avoid reloading models)
print("Initializing OCR Handler...")
ocr_handler = OCRHandler(languages=['en', 'hi'])  # Add more languages as needed
print("OCR Handler ready!")

# Store session data (in production, use Redis or database)
sessions = {}

# Pydantic models for request/response
class TextRegion(BaseModel):
    id: str
    text: str
    selected_for_translation: bool
    bbox: Dict
    style: Dict
    metadata: Dict

class DetectionResponse(BaseModel):
    session_id: str
    image_dimensions: Dict
    total_regions: int
    text_regions: List[Dict]
    image_url: str

class TranslationRequest(BaseModel):
    session_id: str
    selected_regions: List[str]  # List of region IDs to translate
    target_language: str
    preserve_styling: bool = True

class SessionData(BaseModel):
    session_id: str
    original_image_path: str
    text_regions: List[Dict]
    image_dimensions: Dict
    created_at: str

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Marketing Image Translator API",
        "version": "1.0.0",
        "endpoints": {
            "/upload": "POST - Upload image for text detection",
            "/get-regions/{session_id}": "GET - Get detected text regions",
            "/update-selections": "POST - Update text selection status",
            "/translate": "POST - Translate selected regions",
            "/download/{session_id}": "GET - Download processed image",
            "/docs": "GET - API documentation (Swagger UI)"
        }
    }

@app.post("/upload", response_model=DetectionResponse)
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image and detect text regions
    
    Args:
        file: Image file (jpg, png, etc.)
    
    Returns:
        Detection results with session ID
    """
    try:
        # Validate file type
        allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/bmp", "image/webp"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
            )
        
        # Validate file size (max 10MB)
        contents = await file.read()
        file_size = len(contents)
        if file_size > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(
                status_code=400,
                detail="File too large. Maximum size is 10MB"
            )
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Save original image
        temp_path = f"temp/{session_id}_original{os.path.splitext(file.filename)[1]}"
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        # Detect text using OCR handler
        print(f"Processing image for session {session_id}...")
        result = ocr_handler.detect_text(image_bytes=contents)
        
        if not result['success']:
            raise HTTPException(
                status_code=500,
                detail=f"OCR detection failed: {result.get('error', 'Unknown error')}"
            )
        
        # Create image preview (base64 for embedding in frontend)
        img = Image.open(BytesIO(contents))
        
        # Resize for preview if too large
        max_size = (800, 800)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        image_url = f"data:image/png;base64,{img_base64}"
        
        # Store session data
        sessions[session_id] = {
            "session_id": session_id,
            "original_image_path": temp_path,
            "text_regions": result['text_regions'],
            "image_dimensions": result['image_dimensions'],
            "created_at": datetime.now().isoformat(),
            "original_filename": file.filename,
            "image_base64": img_base64
        }
        
        print(f"✅ Detected {result['total_regions']} text regions")
        
        return DetectionResponse(
            session_id=session_id,
            image_dimensions=result['image_dimensions'],
            total_regions=result['total_regions'],
            text_regions=result['text_regions'],
            image_url=image_url
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in upload_image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-regions/{session_id}")
async def get_text_regions(session_id: str):
    """
    Get detected text regions for a session
    
    Args:
        session_id: Session identifier
    
    Returns:
        Text regions and image data
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[session_id]
    
    return {
        "success": True,
        "session_id": session_id,
        "text_regions": session_data['text_regions'],
        "image_dimensions": session_data['image_dimensions'],
        "image_url": f"data:image/png;base64,{session_data['image_base64']}",
        "total_regions": len(session_data['text_regions'])
    }

@app.post("/update-selections")
async def update_text_selections(
    session_id: str = Form(...),
    selections: str = Form(...)  # JSON string of selections
):
    """
    Update which text regions are selected for translation
    
    Args:
        session_id: Session identifier
        selections: JSON string with region selections
                   Format: {"region_id": true/false, ...}
    
    Returns:
        Updated session data
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Parse selections
        selection_dict = json.loads(selections)
        
        # Update text regions
        session_data = sessions[session_id]
        for region in session_data['text_regions']:
            if region['id'] in selection_dict:
                region['selected_for_translation'] = selection_dict[region['id']]
        
        # Count selected regions
        selected_count = sum(
            1 for r in session_data['text_regions'] 
            if r.get('selected_for_translation', False)
        )
        
        return {
            "success": True,
            "session_id": session_id,
            "selected_regions": selected_count,
            "total_regions": len(session_data['text_regions']),
            "message": f"Updated {len(selection_dict)} regions, {selected_count} selected for translation"
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in selections")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session-summary/{session_id}")
async def get_session_summary(session_id: str):
    """
    Get summary of detected text for review
    
    Args:
        session_id: Session identifier
    
    Returns:
        Summary of text regions with selection status
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[session_id]
    
    # Create summary
    summary = {
        "session_id": session_id,
        "original_filename": session_data.get('original_filename', 'unknown'),
        "image_dimensions": session_data['image_dimensions'],
        "created_at": session_data['created_at'],
        "statistics": {
            "total_regions": len(session_data['text_regions']),
            "selected_for_translation": 0,
            "preserved_brands": 0,
            "by_script": {},
            "by_confidence": {
                "high": 0,  # > 0.9
                "medium": 0,  # 0.7 - 0.9
                "low": 0  # < 0.7
            }
        },
        "text_summary": []
    }
    
    # Analyze regions
    for region in session_data['text_regions']:
        # Count selections
        if region.get('selected_for_translation', False):
            summary['statistics']['selected_for_translation'] += 1
        else:
            if region['metadata'].get('is_likely_brand', False):
                summary['statistics']['preserved_brands'] += 1
        
        # Count by script
        script = region['metadata'].get('script_type', 'unknown')
        summary['statistics']['by_script'][script] = \
            summary['statistics']['by_script'].get(script, 0) + 1
        
        # Count by confidence
        conf = region['confidence']
        if conf > 0.9:
            summary['statistics']['by_confidence']['high'] += 1
        elif conf > 0.7:
            summary['statistics']['by_confidence']['medium'] += 1
        else:
            summary['statistics']['by_confidence']['low'] += 1
        
        # Add to text summary
        summary['text_summary'].append({
            "id": region['id'],
            "text": region['text'],
            "selected": region.get('selected_for_translation', False),
            "confidence": round(region['confidence'], 2),
            "is_brand": region['metadata'].get('is_likely_brand', False),
            "has_trademark": region['metadata'].get('has_trademark', False)
        })
    
    return summary

@app.post("/clear-session/{session_id}")
async def clear_session(session_id: str):
    """
    Clear session data and temporary files
    
    Args:
        session_id: Session identifier
    
    Returns:
        Success status
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Delete temporary files
        session_data = sessions[session_id]
        if os.path.exists(session_data['original_image_path']):
            os.remove(session_data['original_image_path'])
        
        # Remove from sessions
        del sessions[session_id]
        
        return {
            "success": True,
            "message": f"Session {session_id} cleared successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ocr_ready": ocr_handler is not None,
        "active_sessions": len(sessions),
        "temp_files": len(os.listdir("temp")) if os.path.exists("temp") else 0
    }
    """Translate selected text regions"""
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Use mock handler if no API key
    if api_key:
        translator = TranslationHandler(api_key)
    else:
        translator = MockTranslationHandler()
    
    # Get selected regions
    session_data = sessions[session_id]
    texts_to_translate = []
    
    for region in session_data['text_regions']:
        if region.get('selected_for_translation', False):
            texts_to_translate.append({
                'id': region['id'],
                'text': region['text']
            })
    
    # Translate texts
    translations = {}
    for text_item in texts_to_translate:
        result = translator.translate_sync(
            text_item['text'], 
            target_language, 
            model
            # Remove the context parameter since it's not supported
        )
        if result['success']:
            translations[text_item['id']] = result['translated']
    
    # Process image
    processor = ImageProcessor()
    output_path = f"output/{session_id}_{target_language}.png"
    
    success = processor.process_image(
        session_data['original_image_path'],
        translations,
        target_language,
        session_data['text_regions'],
        output_path
    )
    
    return {
        "success": success,
        "translations": translations,
        "output_path": output_path
    }

@app.post("/translate")
async def translate_regions(
    session_id: str = Form(...),
    target_language: str = Form(...),
    api_key: str = Form(None),
    model: str = Form("openai/gpt-3.5-turbo")
):
    """
    Translate selected text regions and generate translated image
    
    Args:
        session_id: Session identifier
        target_language: Target language code (hi, mr, ta, etc.)
        api_key: OpenRouter API key (optional, uses mock if not provided)
        model: LLM model to use for translation
    
    Returns:
        Translation results and output image path
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Add validation for target language
        supported_languages = ['hi', 'mr', 'ta', 'te', 'kn', 'ml', 'gu', 'bn', 'pa', 'or', 'as']
        if target_language not in supported_languages:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported language: {target_language}. Supported: {supported_languages}"
            )
        
        # Use mock handler if no API key provided
        if api_key and api_key.strip():
            translator = TranslationHandler(api_key)
            print(f"Using OpenRouter API with model: {model}")
        else:
            translator = MockTranslationHandler()
            print("⚠️  Using mock translator - translations will be fake!")
        
        # Get session data
        session_data = sessions[session_id]
        texts_to_translate = []
        
        # Collect selected texts with better context
        for region in session_data['text_regions']:
            if region.get('selected_for_translation', False):
                # Add more context for better translations
                context = {
                    'text': region['text'],
                    'confidence': region['confidence'],
                    'is_brand': region['metadata'].get('is_likely_brand', False),
                    'word_count': region['style'].get('word_count', 1),
                    'font_size': region['style'].get('estimated_font_size', 'medium')
                }
                texts_to_translate.append({
                    'id': region['id'],
                    'text': region['text'],
                    'context': context
                })
        
        if not texts_to_translate:
            return {
                "success": False,
                "error": "No text regions selected for translation",
                "message": "Please select at least one text region to translate"
            }
        
        print(f"Translating {len(texts_to_translate)} regions to {target_language}")
        
        # Translate with better error handling
        translations = {}
        failed_translations = []
        
        for text_item in texts_to_translate:
            try:
                # Skip very short or likely brand text
                if (len(text_item['text'].strip()) < 2 or 
                    text_item['context']['is_brand']):
                    translations[text_item['id']] = text_item['text']
                    print(f"⏭️  Skipped (brand/short): '{text_item['text']}'")
                    continue
                
                # Remove the context parameter - not supported by translate_sync
                result = translator.translate_sync(
                    text_item['text'], 
                    target_language, 
                    model
                )
                
                if result['success'] and result['translated'].strip():
                    translations[text_item['id']] = result['translated']
                    print(f"✅ '{text_item['text']}' → '{result['translated']}'")
                else:
                    translations[text_item['id']] = text_item['text']
                    failed_translations.append(text_item['text'])
                    print(f"❌ Failed: '{text_item['text']}'")
                    
            except Exception as e:
                print(f"❌ Translation error for '{text_item['text']}': {e}")
                translations[text_item['id']] = text_item['text']
                failed_translations.append(text_item['text'])
        
        # Validate we have translations
        if not translations:
            raise HTTPException(
                status_code=500,
                detail="All translations failed. Please check your API key and try again."
            )
        
        # Process image with better error handling
        processor = ImageProcessor()
        output_filename = f"{session_id}_{target_language}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        output_path = f"output/{output_filename}"
        
        # Ensure output directory exists
        os.makedirs("output", exist_ok=True)
        
        print(f"🖼️  Processing image: {session_data['original_image_path']}")
        
        # Add validation for image processing
        if not os.path.exists(session_data['original_image_path']):
            raise HTTPException(
                status_code=500,
                detail="Original image file not found"
            )
        
        success = processor.process_image(
            session_data['original_image_path'],
            translations,
            target_language,
            session_data['text_regions'],
            output_path
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Image processing failed. Check if fonts are installed for the target language."
            )
        
        if not os.path.exists(output_path):
            raise HTTPException(
                status_code=500,
                detail="Output image was not created successfully"
            )
        
        # Store results
        if 'translations' not in session_data:
            session_data['translations'] = {}
        
        session_data['translations'][target_language] = {
            'translations': translations,
            'output_path': output_path,
            'filename': output_filename,
            'failed_count': len(failed_translations),
            'success_count': len(translations) - len(failed_translations)
        }
        
        print(f"✅ Translation complete! Output: {output_path}")
        
        return {
            "success": True,
            "session_id": session_id,
            "language": target_language,
            "translated_count": len(translations) - len(failed_translations),
            "failed_count": len(failed_translations),
            "total_regions": len(session_data['text_regions']),
            "selected_regions": len(texts_to_translate),
            "translations": translations,
            "output_filename": output_filename,
            "download_url": f"/download/{session_id}/{target_language}",
            "failed_texts": failed_translations if failed_translations else None,
            "message": f"Successfully processed {len(translations)} regions ({len(failed_translations)} failed)" if failed_translations else f"Successfully translated all {len(translations)} regions"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Translation endpoint error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Translation failed: {str(e)}"
        )

@app.get("/test-fonts/{language}")
async def test_fonts(language: str = "hindi"):
    """Test font rendering for a language"""
    try:
        from image_processor import ImageProcessor
        processor = ImageProcessor()
        processor.test_rendering(language)
        
        return {
            "success": True,
            "message": f"Font test completed for {language}",
            "test_image": f"fonts/test_{language}.png"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/save-selections")
async def save_text_selections(request: dict):
    """
    Save user's text selection choices
    
    Request body:
    {
        "session_id": "xxx",
        "selected_regions": ["text_region_0", "text_region_5", ...]
    }
    """
    session_id = request.get('session_id')
    selected_ids = request.get('selected_regions', [])
    
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required")
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session_data = sessions[session_id]
        
        # Update ALL regions based on selection
        # First, set all to false
        for region in session_data['text_regions']:
            region['selected_for_translation'] = False
        
        # Then set selected ones to true
        for region in session_data['text_regions']:
            if region['id'] in selected_ids:
                region['selected_for_translation'] = True
        
        # Count and log for debugging
        selected_count = len(selected_ids)
        total_count = len(session_data['text_regions'])
        
        print(f"\n=== Selections Saved for session {session_id} ===")
        print(f"Selected regions: {selected_ids}")
        print(f"Total selected: {selected_count} out of {total_count}")
        
        # Debug: Show what's selected
        for region in session_data['text_regions']:
            status = "✓ SELECTED" if region['selected_for_translation'] else "✗ Not selected"
            print(f"  {region['id']}: {status} - '{region['text'][:30]}...'")
        
        return {
            "success": True,
            "session_id": session_id,
            "selected_count": selected_count,
            "total_regions": total_count,
            "message": f"Saved {selected_count} selections out of {total_count} regions"
        }
        
    except Exception as e:
        print(f"Error saving selections: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{session_id}/{language}")
async def download_translated_image(session_id: str, language: str):
    """
    Download the translated image for a specific language
    
    Args:
        session_id: Session identifier
        language: Language code of the translated image
    
    Returns:
        The translated image file
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[session_id]
    
    # Check if translation exists
    if 'translations' not in session_data or language not in session_data['translations']:
        raise HTTPException(status_code=404, detail=f"Translation for {language} not found")
    
    # Get the output path
    output_path = session_data['translations'][language]['output_path']
    
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Translated image file not found")
    
    return FileResponse(
        output_path,
        media_type="image/png",
        filename=f"translated_{language}_{session_id}.png"
    )

@app.get("/translation-status/{session_id}")
async def get_translation_status(session_id: str):
    """
    Get the status of translations for a session
    
    Args:
        session_id: Session identifier
    
    Returns:
        List of completed translations
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[session_id]
    
    if 'translations' not in session_data:
        return {
            "session_id": session_id,
            "translations": [],
            "message": "No translations completed yet"
        }
    
    completed_translations = []
    for lang, data in session_data['translations'].items():
        completed_translations.append({
            "language": lang,
            "filename": data['filename'],
            "download_url": f"/download/{session_id}/{lang}",
            "region_count": len(data['translations'])
        })
    
    return {
        "session_id": session_id,
        "translations": completed_translations,
        "total": len(completed_translations)
    }

# Cleanup old sessions periodically (in production, use background tasks)
@app.on_event("startup")
async def startup_event():
    """Clean temp directory on startup"""
    print("🚀 Starting Marketing Image Translator API...")
    
    # Clean old temp files
    temp_dir = "temp"
    if os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            try:
                os.remove(file_path)
                print(f"Cleaned up: {file}")
            except:
                pass
    
    print("✅ API Ready!")

# Run the application
if __name__ == "__main__":
    # Run with: python app.py
    # Or: uvicorn app:app --reload --port 8000
    
    print("=" * 60)
    print("🚀 Marketing Image Translator API")
    print("=" * 60)
    print("Starting server on http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 60)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )