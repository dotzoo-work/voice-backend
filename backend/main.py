import sys
sys.stdout.reconfigure(encoding='utf-8')

from fastapi import FastAPI, HTTPException, File, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
import openai
import httpx
import traceback
from sarvamai import SarvamAI
from sarvamai.play import save
####Request models
class ChatRequest(BaseModel):
    message: str
    bot_id: str = "default"

class RelayRequest(BaseModel):
    message: str
    bot_id: str = "default"
    language: str = "en"
import os
from dotenv import load_dotenv
import httpx
import json
import io
import base64
import re
import unicodedata
import asyncio
import hashlib
import time
import tempfile
import subprocess
import aiofiles
# FFmpeg setup using imageio-ffmpeg
FFMPEG_AVAILABLE = False
ffmpeg_path = None

try:
    from imageio_ffmpeg import get_ffmpeg_exe
    ffmpeg_path = get_ffmpeg_exe()
    
    # Test if it works
    result = subprocess.run([ffmpeg_path, "-version"], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        FFMPEG_AVAILABLE = True
        print(f"✅ FFmpeg available: {ffmpeg_path}")
    else:
        print(f"❌ FFmpeg test failed")
except ImportError:
    print(f"❌ imageio-ffmpeg not installed")
except Exception as e:
    print(f"❌ FFmpeg setup failed: {e}")
    
print(f"FFmpeg status: {'ENABLED' if FFMPEG_AVAILABLE else 'DISABLED'}")


load_dotenv()



# Session-wise audio buffers - not needed for full blob approach
# SESSION_AUDIO_BUFFERS = {}

# Allowed languages
ALLOWED_LANGUAGES = ["en", "es", "hi", "pa"]

# STT Router
async def stt_router(audio_data, lang):
    if lang not in ALLOWED_LANGUAGES:
        raise ValueError(f"Language {lang} not supported")

    if lang in ["en", "es"]:
        # GPT-4o Transcribe (Whisper-1 backend)
        return await gpt4o_transcribe(audio_data, lang)

    if lang in ["hi", "pa"]:
        # Sarika v2.5
        return await sarvam_stt(audio_data, lang)

async def gpt4o_transcribe(audio_data, lang):
    """GPT-4o transcription for English and Spanish"""
    audio_file = io.BytesIO(audio_data)
    audio_file.name = "audio.wav"
    
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language=lang,
        response_format="json",
        temperature=0
    )
    return getattr(transcript, 'text', '').strip()

async def sarvam_stt(audio_data, lang):
    """Sarvam STT for Hindi and Punjabi"""
    lang_map = {
        "hi": "hi-IN",
        "pa": "pa-IN"
    }
    
    if lang not in lang_map:
        raise ValueError(f"Language {lang} not supported for Sarvam STT")
    
    try:
        # Save audio to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
        
        # Use Sarvam STT API
        with open(temp_path, "rb") as audio_file:
            response = sarvam_client.speech_to_text.transcribe(
                file=audio_file,
                language_code=lang_map[lang]
            )
        
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass
            
        # Extract transcript
        transcript = getattr(response, 'transcript', '').strip()
        return transcript if transcript else ""
        
    except Exception as e:
        print(f"Sarvam STT error: {e}")
        # Clean up temp file on error
        try:
            os.unlink(temp_path)
        except:
            pass
        return ""



async def convert_audio_to_wav(audio_data, input_format="webm"):
    """Convert audio to WAV format using imageio-ffmpeg"""
    if not FFMPEG_AVAILABLE:
        print("FFmpeg not available, using original audio")
        return audio_data
        
    try:
        print(f"Converting {input_format} to WAV")
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{input_format}") as original_file:
            original_file.write(audio_data)
            original_file.flush()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_file:
                convert_cmd = [
                    ffmpeg_path,
                    "-y",
                    "-i", original_file.name,
                    "-ar", "16000",
                    "-ac", "1",
                    wav_file.name
                ]
                
                process = subprocess.run(
                    convert_cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    timeout=30,
                    encoding="utf-8"
                )
                
                if process.returncode != 0:
                    error_msg = process.stderr if process.stderr else "FFmpeg error"
                    print(f"FFmpeg failed: {error_msg}")
                    try:
                        os.unlink(original_file.name)
                        os.unlink(wav_file.name)
                    except:
                        pass
                    return audio_data
                
                with open(wav_file.name, "rb") as f:
                    wav_data = f.read()
                
                print(f"Conversion successful: {len(wav_data)} bytes")
                
                try:
                    os.unlink(original_file.name)
                    os.unlink(wav_file.name)
                except:
                    pass
                
                return wav_data
                
    except Exception as e:
        print(f"Audio conversion error: {e}")
        return audio_data

app = FastAPI(title="Voice Backend Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # Allow all origins for development
        "https://www.edmondsbaydental.com",
        "https://voice.yesitisfree.com",
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "https://localhost:3000",
        "https://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=30.0
)

# Sarvam client
sarvam_client = SarvamAI(
    api_subscription_key=os.getenv("SARVAM_API_KEY")
)

# Multiple chatbots configuration - Add your projects here
CHATBOT_URLS = {
     "dr-tomar": "https://edmonds.yesitisfree.com/api/chat",
    
    
    "default": "https://edmonds.yesitisfree.com/api/chat"
}

# Redis disabled - using local cache only

# Performance optimizations
tts_cache = {}
CACHE_MAX_SIZE = 100
CACHE_TTL = 86400  # 24 hours
MAX_TTS_LENGTH = 2500

# Connection pooling for faster API calls
connection_pool = httpx.AsyncClient(
    limits=httpx.Limits(max_keepalive_connections=5, max_connections=20),
    timeout=httpx.Timeout(30.0, connect=5.0, read=30.0),
    follow_redirects=True
)



def safe_text(text):
    """Guaranteed safe text conversion for TTS"""
    if text is None:
        return ""
    return str(text)

def get_cache_key(text, lang="en", provider="openai"):
    """Generate cache key with text, language, and provider"""
    cache_string = f"{text}_{lang}_{provider}"
    return hashlib.md5(cache_string.encode()).hexdigest()

def get_cached_tts(text, lang="en", provider="openai"):
    cache_key = get_cache_key(text, lang, provider)
    if cache_key in tts_cache:
        cached_item = tts_cache[cache_key]
        if time.time() - cached_item['timestamp'] < CACHE_TTL:
            return cached_item['audio']
        else:
            del tts_cache[cache_key]
    return None

def cache_tts(text, audio_content, lang="en", provider="openai"):
    if len(tts_cache) >= CACHE_MAX_SIZE:
        oldest_key = min(tts_cache.keys(), key=lambda k: tts_cache[k]['timestamp'])
        del tts_cache[oldest_key]
    
    cache_key = get_cache_key(text, lang, provider)
    tts_cache[cache_key] = {
        'audio': audio_content,
        'timestamp': time.time()
    }

def optimize_text_for_tts(text):
    if len(text) <= MAX_TTS_LENGTH:
        return text
    
    sentences = text.split('. ')
    result = ""
    for sentence in sentences:
        if len(result + sentence + '. ') <= MAX_TTS_LENGTH:
            result += sentence + '. '
        else:
            break
    
    return result.strip() or text[:MAX_TTS_LENGTH]

def clean_text_for_tts(text):
    if not text:
        return "Hello"
    
    text = str(text)
    
    try:
        # Remove HTML entities and decode them properly
        import html
        text = html.unescape(text)  # Convert &#39; to ' etc.
        text = re.sub(r'&#\d+;', '', text)
        text = re.sub(r'&\w+;', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) < 3:
            return "I apologize for the formatting issue."
            
        return text
        
    except Exception as e:
        print(f"Cleaning error: {e}")
        return "There was a text processing error."

# Language normalization
def normalize_language(lang):
    """Normalize and validate language input"""
    if not lang or not isinstance(lang, str):
        return "en"
    
    lang = lang.strip().lower()
    if lang in ALLOWED_LANGUAGES:
        return lang
    
    return "en"

# TTS Router
async def tts_router(text, lang):
    if lang not in ALLOWED_LANGUAGES:
        raise ValueError("Unsupported language")

    if lang in ["en", "es"]:
        return await openai_tts(text, lang)   # tts-1

    if lang in ["hi", "pa"]:
        return await sarvam_tts(text, lang)   # Bulbul v2

async def elevenlabs_or_gpt_tts(text, lang):
    """ElevenLabs or GPT TTS for English and Spanish"""
    # For now, use OpenAI TTS - can be switched to ElevenLabs later
    return await openai_tts(text, lang)

async def sarvam_tts(text, lang):
    lang_map = {
        "hi": "hi-IN",
        "pa": "pa-IN"
    }

    if lang not in lang_map:
        raise ValueError("Unsupported language for Sarvam TTS")

    try:
        response = sarvam_client.text_to_speech.convert(
            text=text,
            target_language_code=lang_map[lang],
            enable_preprocessing=True
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            save(response, f.name)
            temp_path = f.name

        with open(temp_path, "rb") as audio:
            audio_bytes = audio.read()

        os.unlink(temp_path)
        return audio_bytes

    except Exception as e:
        print("Sarvam SDK TTS failed:", e)
        raise

async def openai_tts(text, lang):
    """OpenAI TTS implementation"""
    # Only English and Spanish use alloy voice
    voice = "alloy"
    
    # Use safe_text to prevent encoding issues
    tts_text = safe_text(text)
    
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=tts_text,
        response_format="mp3"  # Ensure MP3 format
    )
    return response.content
# LLM with forced response language
async def get_chatbot_response(message: str, bot_id: str = "default", lang: str = "en"):
    """
    Call the chatbot backend API with strict language enforcement.
    """
    chatbot_url = CHATBOT_URLS.get(bot_id, CHATBOT_URLS["default"])
    
    payload = {
        "message": message,
        "bot_id": bot_id,
        "language": lang
    }

    print(f"Sending to chatbot: {chatbot_url}")
    print(f"Message: {message}")

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            response = await client.post(chatbot_url, json=payload)

        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                bot_response = data.get("response") or data.get("answer") or "Sorry, no response"
                
                # Keep Unicode characters for multilingual support
                cleaned_response = str(bot_response)
                if not cleaned_response.strip():
                    cleaned_response = "I understand your message."
                
                print(f"Bot response: {cleaned_response}")
                return cleaned_response
                
            except Exception as e:
                print(f"JSON parse error: {e}")
                return "Response parsing failed"
        else:
            print(f"API error: {response.status_code}")
            return "API connection failed"

    except Exception as e:
        print(f"Request error: {e}")
        # Return a more user-friendly fallback message
        if lang == "pa":
            return "ਮਾਫ਼ ਕਰਨਾ, ਮੈਂ ਇਸ ਸਮੇਂ ਜਵਾਬ ਨਹੀਂ ਦੇ ਸਕਦਾ।"
        elif lang == "hi":
            return "माफ़ करें, मैं इस समय जवाब नहीं दे सकता।"
        else:
            return "I apologize, I cannot respond at this time."



# API Endpoints
@app.get("/session-ephemeral")
async def get_session_ephemeral(bot_id: str = "default"):
    """Create ephemeral OpenAI Realtime token"""
    try:
        response = client.beta.realtime.sessions.create(
            model="gpt-4o-mini",
            voice="nova"
        )
        return {
            "client_secret": {
                "value": response.client_secret.value,
                "expires_at": response.client_secret.expires_at
            },
            "bot_id": bot_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/relay-message")
async def relay_message(request: RelayRequest):
    """Relay message to chatbot backend"""
    try:
        print(f"🔍 Relay request - Message: {request.message}, Language: {request.language}, Bot: {request.bot_id}")
        response = await get_chatbot_response(request.message, request.bot_id, request.language)
        print(f"🔍 Relay response: {response}")
        return {"response": response, "bot_id": request.bot_id, "language": request.language}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-multilingual")
async def test_multilingual(message: str = "नमस्ते", bot_id: str = "default", lang: str = "hi"):
    """Test multilingual chatbot response"""
    try:
        print(f"Testing multilingual - Message: {message}, Language: {lang}")
        response = await get_chatbot_response(message, bot_id, lang)
        print(f"Test response: {response}")
        return {
            "success": True,
            "input_message": message,
            "input_language": lang,
            "bot_response": response,
            "bot_id": bot_id
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "input_message": message,
            "input_language": lang
        }

@app.get("/test-chatbot-simple")
async def test_chatbot_simple():
    """Simple test for chatbot connection"""
    try:
        response = await get_chatbot_response("Hello", "dr-tomar", "en")
        return {
            "success": True,
            "response": response,
            "message": "Chatbot working!"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/test-chatbot")
async def test_chatbot(message: str = "Hello", bot_id: str = "dr-tomar", lang: str = "en"):
    """Simple test endpoint for chatbot"""
    try:
        print(f"Testing chatbot - Message: {message}, Bot: {bot_id}, Language: {lang}")
        response = await get_chatbot_response(message, bot_id, lang)
        return {
            "success": True,
            "message": message,
            "bot_id": bot_id,
            "language": lang,
            "response": response
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": message,
            "bot_id": bot_id,
            "language": lang
        }



def split_text_for_streaming(text, chunk_size=100):
    """Split text into chunks for streaming TTS"""
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk + sentence + '. ') <= chunk_size:
            current_chunk += sentence + '. '
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

async def generate_single_tts_chunk(chunk, chunk_index, lang="en"):
    """Generate TTS for a single chunk using TTS router"""
    # Ensure chunk is UTF-8 string
    tts_chunk = str(chunk)
    
    # Check cache first
    local_cached = get_cached_tts(tts_chunk, lang, "tts_router")
    if local_cached:
        return chunk_index, local_cached, tts_chunk
    
    try:
        audio_content = await tts_router(tts_chunk, lang)
        
        # Cache the chunk with language and provider
        cache_tts(tts_chunk, audio_content, lang, "tts_router")
        
        return chunk_index, audio_content, tts_chunk
    except Exception as e:
        print(f"TTS Chunk Error: {e}")
        return chunk_index, None, tts_chunk

async def generate_tts_audio_streaming(text, websocket, lang="en"):
    """Generate TTS audio in parallel chunks and stream to websocket"""
    chunks = split_text_for_streaming(text, 150)
    
    # Generate all chunks in parallel
    tasks = [
        generate_single_tts_chunk(chunk, i, lang) 
        for i, chunk in enumerate(chunks)
    ]
    
    # Process chunks as they complete
    for task in asyncio.as_completed(tasks):
        chunk_index, audio_content, chunk_text = await task
        
        if audio_content:
            audio_b64 = base64.b64encode(audio_content).decode()
            await websocket.send_json({
                "type": "audio_chunk",
                "data": audio_b64,
                "chunk_index": chunk_index,
                "total_chunks": len(chunks),
                "text_chunk": chunk_text
            })
        else:
            print(f"Failed to generate audio for chunk {chunk_index}")

# Removed duplicate endpoint - using voice-realtime instead



async def generate_tts_audio(text, lang="en"):
    """Generate TTS audio using TTS router"""
    # Ensure text is UTF-8 string
    tts_text = str(text)
    
    # Check local cache
    local_cached = get_cached_tts(tts_text, lang, "tts_router")
    if local_cached:
        return local_cached
    
    try:
        print(f"🎵 Generating TTS for language: {lang}")
        audio_content = await tts_router(tts_text, lang)
        
        # Cache with language and provider
        cache_tts(tts_text, audio_content, lang, "tts_router")
        
        return audio_content
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...), language: str = "en"):
    """STT endpoint using STT router"""
    lang = normalize_language(language)
    if lang not in ALLOWED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")
    
    try:
        audio_data = await file.read()
        wav_data = await convert_audio_to_wav(audio_data, "webm")
        
        transcript = await stt_router(wav_data, lang)
        
        return {
            "text": transcript,
            "language": lang
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT failed: {str(e)}")

@app.post("/tts")
async def text_to_speech(request: dict):
    """TTS endpoint using TTS router"""
    lang = normalize_language(request.get("language"))
    if lang not in ALLOWED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")
    
    try:
        text = request.get("text", "")
        clean_text = clean_text_for_tts(text)
        optimized_text = optimize_text_for_tts(clean_text)
        
        audio_content = await generate_tts_audio(optimized_text, lang)
        if not audio_content:
            raise HTTPException(status_code=500, detail="TTS generation failed")
        
        return StreamingResponse(
            io.BytesIO(audio_content),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")

@app.websocket("/ws/voice-realtime")
async def voice_realtime_websocket(websocket: WebSocket, bot_id: str = "default"):
    """Single WebSocket endpoint for voice processing"""
    await websocket.accept()
    session_language = "en"  # Default language
    
    print(f"WebSocket connected for bot_id: {bot_id}")
    
    try:
        while True:
            try:
                # Wait for any message
                message_data = await websocket.receive()
                print(f"Raw message received: {type(message_data)}, keys: {list(message_data.keys()) if isinstance(message_data, dict) else 'not dict'}")
                
                # Handle different message types
                if "text" in message_data:
                    # Text message - try to parse as JSON
                    try:
                        message = json.loads(message_data["text"])
                        print(f"Parsed JSON message type: {message.get('type', 'unknown')}")
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse JSON: {e}")
                        continue
                        
                elif "json" in message_data:
                    # Direct JSON message
                    message = message_data["json"]
                    print(f"Direct JSON message type: {message.get('type', 'unknown')}")
                    
                elif "bytes" in message_data:
                    # Full audio blob received - process STT immediately
                    audio_bytes = message_data["bytes"]
                    session_id = int(time.time())
                    print(f"Received full audio blob: {len(audio_bytes)} bytes")
                    
                    # Process full audio immediately
                    try:
                        # Convert to WAV once
                        wav_data = await convert_audio_to_wav(audio_bytes, "webm")
                        
                        # Run STT once on complete audio
                        user_text = await stt_router(wav_data, session_language)
                        
                        if not user_text:
                            await websocket.send_json({
                                "type": "no_speech_detected",
                                "session_id": session_id
                            })
                            continue
                        
                        # Send transcript
                        await websocket.send_json({
                            "type": "transcript",
                            "text": user_text,
                            "language": session_language,
                            "session_id": session_id
                        })
                        
                        # Get bot response
                        bot_response = await get_chatbot_response(user_text, bot_id, session_language)
                        
                        await websocket.send_json({
                            "type": "bot_response",
                            "text": bot_response,
                            "session_id": session_id
                        })
                        
                        # Generate TTS
                        clean_text = clean_text_for_tts(bot_response)
                        optimized_text = optimize_text_for_tts(clean_text)
                        audio_content = await generate_tts_audio(optimized_text, session_language)
                        
                        if audio_content:
                            audio_b64 = base64.b64encode(audio_content).decode()
                            await websocket.send_json({
                                "type": "audio_response",
                                "data": audio_b64,
                                "format": "wav" if session_language in ["hi", "pa"] else "mp3",
                                "session_id": session_id
                            })
                        
                    except Exception as e:
                        print(f"Audio processing error: {e}")
                        await websocket.send_json({
                            "type": "processing_error",
                            "error": str(e),
                            "session_id": session_id
                        })
                    
                    continue
                else:
                    print(f"Unknown message format: {message_data}")
                    continue
                # Process the parsed message
                if message.get("type") == "config":
                    # Language configuration from frontend
                    session_language = normalize_language(message.get("language", "en"))
                    print(f"Language configured: {session_language}")
                    await websocket.send_json({
                        "type": "config_received",
                        "language": session_language
                    })
                    
                elif message.get("type") == "processing_complete":
                    # Just a signal that recording finished
                    session_id = message.get("session_id", int(time.time()))
                    print(f"Recording finished for session {session_id}")
                    # STT already processed when audio bytes were received
                    
                elif message.get("type") == "audio":
                    audio_b64 = message.get("audio", "")
                    lang = normalize_language(message.get("language", session_language))
                    session_id = message.get("session_id", int(time.time()))
                    
                    print(f"Processing audio message - Language: {lang}, Audio size: {len(audio_b64)} chars")
                    
                    # Language validation
                    if lang not in ALLOWED_LANGUAGES:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Unsupported language",
                            "session_id": session_id
                        })
                        continue
                    
                    if not audio_b64:
                        await websocket.send_json({
                            "type": "error",
                            "message": "No audio data provided",
                            "session_id": session_id
                        })
                        continue
                    
                    # Decode audio data and process immediately
                    try:
                        audio_data = base64.b64decode(audio_b64)
                        print(f"Decoded audio data: {len(audio_data)} bytes")
                        
                        # Process immediately like the bytes handler above
                        wav_data = await convert_audio_to_wav(audio_data, "webm")
                        user_text = await stt_router(wav_data, lang)
                        
                        if not user_text:
                            await websocket.send_json({
                                "type": "no_speech_detected",
                                "session_id": session_id
                            })
                            continue
                        
                        await websocket.send_json({
                            "type": "transcript",
                            "text": user_text,
                            "language": lang,
                            "session_id": session_id
                        })
                        
                        bot_response = await get_chatbot_response(user_text, bot_id, lang)
                        await websocket.send_json({
                            "type": "bot_response",
                            "text": bot_response,
                            "session_id": session_id
                        })
                        
                        clean_text = clean_text_for_tts(bot_response)
                        optimized_text = optimize_text_for_tts(clean_text)
                        audio_content = await generate_tts_audio(optimized_text, lang)
                        
                        if audio_content:
                            audio_b64 = base64.b64encode(audio_content).decode()
                            await websocket.send_json({
                                "type": "audio_response",
                                "data": audio_b64,
                                "format": "wav" if lang in ["hi", "pa"] else "mp3",
                                "session_id": session_id
                            })
                    except Exception as e:
                        print(f"Audio decode error: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Invalid audio data: {e}",
                            "session_id": session_id
                        })
                        
            except Exception as e:
                print(f"WebSocket message processing error: {e}")
                break
            
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()

@app.post("/voice-chat")
@app.get("/voice-chat")
async def voice_chat(file: UploadFile = File(None), bot_id: str = "default", language: str = "en"):
    """Voice chat endpoint with forced language"""
    lang = normalize_language(language)
    if lang not in ALLOWED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")
    
    # Handle GET request for testing
    if file is None:
        return {
            "message": "Voice chat endpoint is working",
            "supported_languages": ALLOWED_LANGUAGES,
            "current_language": lang,
            "bot_id": bot_id
        }
    
    try:
        audio_data = await file.read()
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Convert audio to WAV
        wav_data = await convert_audio_to_wav(audio_data, "webm")
        
        # Use STT router with forced language
        user_text = await stt_router(wav_data, lang)
        if not user_text:
            user_text = "Hello"
        
        print(f"User: {user_text} (Language: {lang})")
        
        # Get chatbot response with forced language
        bot_response = await get_chatbot_response(user_text, bot_id, lang)
        
        # Clean and optimize response
        clean_response = clean_text_for_tts(bot_response)
        optimized_response = optimize_text_for_tts(clean_response)
        
        # Generate TTS with forced language
        audio_content = await generate_tts_audio(optimized_response, lang)
        
        if audio_content:
            # Remove headers to avoid latin-1 encoding issues
            return Response(
                content=audio_content,
                media_type="audio/mpeg"
            )
        else:
            return {
                "transcript": user_text,
                "response": optimized_response,
                "audio_error": "TTS generation failed"
            }
        
    except Exception as e:
        print(f"Voice chat error: {e}")
        return {
            "transcript": "",
            "response": "I had trouble processing your request. Please try again.",
            "error": str(e)
        }

@app.get("/")
async def root():
    redis_status = "disabled"
    
    return {
        "message": "Voice Backend Service is running", 
        "status": "active",
        "redis_status": redis_status,
        "optimizations": {
            "tts_cache": f"{len(tts_cache)} items cached",
            "max_tts_length": MAX_TTS_LENGTH,
            "cache_ttl": "24 hours"
        },
        "endpoints": {
            "voice_realtime": "/ws/voice-realtime (Single WebSocket Endpoint)",
            "voice_chat": "/voice-chat",
            "stt": "/stt",
            "tts": "/tts",
            "relay": "/relay-message"
        },
        "architecture": {
            "stt_routing": "Language-based STT routing (GPT-4o, Sarvam, ElevenLabs)",
            "tts_routing": "Language-based TTS routing (OpenAI, Sarvam, Ethiopic)",
            "language_detection": "Disabled - UI forces language",
            "cache_strategy": "Text + Language + Provider based caching"
        }
    }

@app.on_event("shutdown")
async def shutdown_event():
    await connection_pool.aclose()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)