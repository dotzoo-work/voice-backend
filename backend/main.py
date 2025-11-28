from fastapi import FastAPI, HTTPException, File, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
import openai
import httpx
import traceback
##Request models
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
# import redis.asyncio as redis  # Commented out for now

load_dotenv()

app = FastAPI(title="Voice Backend Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.edmondsbaydental.com",
        "https://voice.yesitisfree.com",
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=30.0
)

# Multiple chatbots configuration - Add your projects here
CHATBOT_URLS = {
    "dr-tomar": "https://edmonds.yesitisfree.com/api/chat",
    
    
    "default": "https://edmonds.yesitisfree.com/api/chat"
}

# Redis configuration - Disabled for now
# redis_client = redis.Redis(
#     host="redis-15221.c16.us-east-1-2.ec2.redns.redis-cloud.com",
#     port=15221,
#     password=os.getenv("REDIS_PASSWORD"),
#     decode_responses=True
# )

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

# Precompute common responses
COMMON_RESPONSES = {
    "hello": "Hello! How can I help you today?",
    "hi": "Hi there! What can I do for you?",
    "thanks": "You're welcome!",
    "bye": "Goodbye! Have a great day!",
    
}

def get_cache_key(text):
    return hashlib.md5(text.encode()).hexdigest()

def get_cached_tts(text):
    cache_key = get_cache_key(text)
    if cache_key in tts_cache:
        cached_item = tts_cache[cache_key]
        if time.time() - cached_item['timestamp'] < CACHE_TTL:
            return cached_item['audio']
        else:
            del tts_cache[cache_key]
    return None

def cache_tts(text, audio_content):
    if len(tts_cache) >= CACHE_MAX_SIZE:
        oldest_key = min(tts_cache.keys(), key=lambda k: tts_cache[k]['timestamp'])
        del tts_cache[oldest_key]
    
    cache_key = get_cache_key(text)
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
        # Remove HTML entities but keep Unicode characters for multilingual support
        text = re.sub(r'&#\d+;', '', text)
        text = re.sub(r'&\w+;', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) < 3:
            return "I apologize for the formatting issue."
            
        return text
        
    except Exception as e:
        print(f"Cleaning error: {e}")
        return "There was a text processing error."
async def get_chatbot_response(message: str, bot_id: str = "default"):
    """
    Call the chatbot backend API safely and return the response text.
    """
    # Use bot_id to get the correct URL
    chatbot_url = CHATBOT_URLS.get(bot_id, CHATBOT_URLS["default"])
    payload = {
        "message": message,
        "bot_id": bot_id
    }

    print(f"📡 Sending message to chatbot API: {chatbot_url}")
    print(f"📦 Payload: {payload}")

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
            response = await client.post(chatbot_url, json=payload, headers={"Content-Type": "application/json"})

        print(f"✅ Chatbot API status: {response.status_code}")
        print(f"✅ Chatbot API raw response: {response.text}")
        print(f"🔍 Response length: {len(response.text)} characters")

        if response.status_code == 200:
            data = response.json()
            return data.get("response") or data.get("answer") or "I'm processing your message."
        else:
            print(f"⚠️ Chatbot API returned non-200: {response.status_code}")
            return "I'm having trouble connecting to the assistant. Please try again."

    except httpx.ConnectError as e:
        print(f"❌ Connection error: {e}")
        traceback.print_exc()
        return "I couldn’t reach the chatbot service. Please check your connection."

    except httpx.TimeoutException:
        print("⚠️ Timeout contacting chatbot API.")
        return "The chatbot took too long to respond. Please try again."

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        return "An unexpected error occurred. Please try again later."

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
        response = await get_chatbot_response(request.message, request.bot_id)
        print(f"🔍 Relay response: {response}")
        return {"response": response, "bot_id": request.bot_id, "language": request.language}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-multilingual")
async def test_multilingual(message: str = "नमस्ते", bot_id: str = "default", lang: str = "hi"):
    """Test multilingual chatbot response"""
    try:
        print(f"🧪 Testing multilingual - Message: {message}, Language: {lang}")
        response = await get_chatbot_response(message, bot_id)
        print(f"🧪 Test response: {response}")
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

async def get_redis_cache(key):
    # Redis disabled - return None
    return None

async def set_redis_cache(key, value):
    # Redis disabled - do nothing
    pass

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
    """Generate TTS for a single chunk"""
    cache_key = get_cache_key(f"{chunk}_{lang}")
    
    # Check cache first
    cached_audio = await get_redis_cache(cache_key)
    if cached_audio:
        return chunk_index, base64.b64decode(cached_audio), chunk
    
    local_cached = get_cached_tts(f"{chunk}_{lang}")
    if local_cached:
        return chunk_index, local_cached, chunk
    
    try:
        # Select voice based on language
        voice_map = {
            "hi": "nova", "hindi": "nova", "ur": "nova", "urdu": "nova",
            "pa": "nova", "punjabi": "nova", "panjabi": "nova",
            "es": "nova", "spanish": "nova", "fr": "shimmer", "french": "shimmer",
            "de": "echo", "german": "echo", "it": "alloy", "italian": "alloy",
            "pt": "onyx", "portuguese": "onyx", "ja": "nova", "japanese": "nova",
            "ko": "nova", "korean": "nova", "zh": "nova", "chinese": "nova",
            "ar": "nova", "arabic": "nova", "ru": "echo", "russian": "echo"
        }
        voice = voice_map.get(lang.lower(), "alloy")
        
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            input=chunk
        )
        audio_content = response.content
        
        # Cache the chunk with language
        cache_tts(f"{chunk}_{lang}", audio_content)
        await set_redis_cache(cache_key, base64.b64encode(audio_content).decode())
        
        return chunk_index, audio_content, chunk
    except Exception as e:
        print(f"TTS Chunk Error: {e}")
        return chunk_index, None, chunk

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
    cache_key = get_cache_key(f"{text}_{lang}")
    
    # Check Redis first
    cached_audio = await get_redis_cache(cache_key)
    if cached_audio:
        return base64.b64decode(cached_audio)
    
    # Check local cache
    local_cached = get_cached_tts(f"{text}_{lang}")
    if local_cached:
        return local_cached
    
    try:
        # Select voice based on language
        voice_map = {
            "hi": "nova", "hindi": "nova", "ur": "nova", "urdu": "nova",
            "pa": "nova", "punjabi": "nova", "panjabi": "nova",
            "es": "nova", "spanish": "nova", "fr": "shimmer", "french": "shimmer",
            "de": "echo", "german": "echo", "it": "alloy", "italian": "alloy",
            "pt": "onyx", "portuguese": "onyx", "ja": "nova", "japanese": "nova",
            "ko": "nova", "korean": "nova", "zh": "nova", "chinese": "nova",
            "ar": "nova", "arabic": "nova", "ru": "echo", "russian": "echo"
        }
        voice = voice_map.get(lang.lower(), "alloy")
        print(f"🎵 Generating TTS with voice: {voice} for language: {lang}")
        
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            input=text
        )
        audio_content = response.content
        
        # Cache in both Redis and local with language
        cache_tts(f"{text}_{lang}", audio_content)
        await set_redis_cache(cache_key, base64.b64encode(audio_content).decode())
        
        return audio_content
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    try:
        audio_data = await file.read()
        audio_file = io.BytesIO(audio_data)
        audio_file.name = "audio.wav"
        
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file,
            response_format="json"
        )
        
        return {
            "text": getattr(transcript, 'text', ''),
            "language": getattr(transcript, 'language', None)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT failed: {str(e)}")

@app.post("/tts")
async def text_to_speech(request: dict):
    try:
        text = request.get("text", "")
        lang = request.get("language", "en")
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
    """Real-time voice processing with 3-second auto-stop"""
    await websocket.accept()
    
    audio_buffer = b""
    recording_start_time = None
    is_recording = False
    session_id = int(time.time())
    auto_stop_task = None
    
    async def auto_stop_recording():
        """Auto-stop recording after 3 seconds"""
        nonlocal is_recording, audio_buffer, session_id
        
        await asyncio.sleep(6.0)
        if is_recording:
            print(f"Auto-stopping recording after 6 seconds")
            
            # Stop recording first
            is_recording = False
            
            await websocket.send_json({
                "type": "recording_stopped",
                "duration": 6.0,
                "session_id": session_id,
                "reason": "auto_stop"
            })
            
            # Process complete audio recording
            if len(audio_buffer) > 0:
                asyncio.create_task(process_realtime_audio(
                    audio_buffer, websocket, session_id, bot_id
                ))
            
            # Send ready signal
            await websocket.send_json({
                "type": "ready_for_next",
                "session_id": session_id
            })
            
            # Reset for next recording
            audio_buffer = b""
            session_id = int(time.time())
    
    try:
        while True:
            # Receive audio chunk
            data = await websocket.receive_bytes()
            current_time = time.time()
            
            # Start recording timer on first audio
            if not is_recording:
                recording_start_time = current_time
                is_recording = True
                
                print(f"Starting recording for session {session_id}")
                
                # Start auto-stop timer
                auto_stop_task = asyncio.create_task(auto_stop_recording())
                
                await websocket.send_json({
                    "type": "recording_started",
                    "session_id": session_id
                })
            
            # Only add to buffer if still recording
            if is_recording:
                audio_buffer += data
                
                # Skip real-time chunk processing to avoid format issues
                # Just accumulate audio for final processing
                
                # Check if we hit 6 seconds (backup check)
                recording_duration = current_time - recording_start_time
                if recording_duration >= 6.0:
                    is_recording = False
                    if auto_stop_task:
                        auto_stop_task.cancel()
                    
                    # Process the audio before reset
                    if len(audio_buffer) > 0:
                        asyncio.create_task(process_realtime_audio(
                            audio_buffer, websocket, session_id, bot_id
                        ))
                    
                    # Reset for next recording
                    audio_buffer = b""
                    session_id = int(time.time())
            
    except Exception as e:
        print(f"Real-time WebSocket error: {e}")
        if auto_stop_task:
            auto_stop_task.cancel()
        await websocket.close()

async def process_realtime_audio(audio_data, websocket, session_id, bot_id="default"):
    """Process audio in real-time with immediate response"""
    try:
        # Send processing status
        await websocket.send_json({
            "type": "processing_started",
            "session_id": session_id
        })
        
        # STT processing with proper audio format handling
        audio_file = io.BytesIO(audio_data)
        
        # Detect audio format from data header
        if audio_data[:4] == b'RIFF':
            audio_file.name = f"realtime_{session_id}.wav"
        elif audio_data[:4] == b'OggS':
            audio_file.name = f"realtime_{session_id}.ogg"
        elif audio_data[:3] == b'ID3' or audio_data[:2] == b'\xff\xfb':
            audio_file.name = f"realtime_{session_id}.mp3"
        else:
            # Default to webm for WebSocket audio
            audio_file.name = f"realtime_{session_id}.webm"
        
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file,
            response_format="json"
        )
        
        user_text = getattr(transcript, 'text', '').strip()
        spoken_lang = getattr(transcript, 'language', None)
        
        if user_text:
            # Send transcript immediately
            await websocket.send_json({
                "type": "transcript",
                "text": user_text,
                "language": spoken_lang,
                "session_id": session_id
            })
            
            # Get chatbot response
            bot_response = await get_chatbot_response(user_text, bot_id)
            
            # Send bot response
            await websocket.send_json({
                "type": "bot_response",
                "text": bot_response,
                "session_id": session_id
            })
            
            # Generate TTS with detected language
            clean_text = clean_text_for_tts(bot_response)
            optimized_text = optimize_text_for_tts(clean_text)
            
            lang = spoken_lang if spoken_lang else "en"
            audio_content = await generate_tts_audio(optimized_text, lang)
            
            if audio_content:
                audio_b64 = base64.b64encode(audio_content).decode()
                await websocket.send_json({
                    "type": "audio_response",
                    "data": audio_b64,
                    "session_id": session_id
                })
        else:
            await websocket.send_json({
                "type": "no_speech_detected",
                "session_id": session_id
            })
        
        # Processing complete
        await websocket.send_json({
            "type": "processing_complete",
            "session_id": session_id
        })
        
        # Send ready signal
        await websocket.send_json({
            "type": "ready_for_next",
            "session_id": session_id
        })
            
    except Exception as e:
        print(f"Realtime audio processing error: {e}")
        await websocket.send_json({
            "type": "processing_error",
            "error": str(e),
            "session_id": session_id
        })
        
        # Still send ready signal even on error
        await websocket.send_json({
            "type": "ready_for_next",
            "session_id": session_id
        })

async def process_complete_audio(audio_data, websocket, session_id, bot_id="default"):
    """Process complete 3-second audio recording"""
    try:
        # Send processing status
        await websocket.send_json({
            "type": "processing_started",
            "session_id": session_id
        })
        
        # STT processing
        audio_file = io.BytesIO(audio_data)
        # Always use .wav for WebSocket audio chunks
        audio_file.name = f"session_{session_id}.wav"
        
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file,
            response_format="json"
        )
        
        user_text = getattr(transcript, 'text', '').strip()
        spoken_lang = getattr(transcript, 'language', None)
        
        if user_text:
            # Send transcript
            await websocket.send_json({
                "type": "transcript",
                "text": user_text,
                "language": spoken_lang,
                "session_id": session_id
            })
            
            # Get chatbot response
            bot_response = await get_chatbot_response(user_text, bot_id)
            
            # Send bot response
            await websocket.send_json({
                "type": "bot_response",
                "text": bot_response,
                "session_id": session_id
            })
            
            # Generate TTS with detected language
            clean_text = clean_text_for_tts(bot_response)
            optimized_text = optimize_text_for_tts(clean_text)
            
            lang = spoken_lang if spoken_lang else "en"
            audio_content = await generate_tts_audio(optimized_text, lang)
            
            if audio_content:
                audio_b64 = base64.b64encode(audio_content).decode()
                await websocket.send_json({
                    "type": "audio_response",
                    "data": audio_b64,
                    "session_id": session_id
                })
        else:
            await websocket.send_json({
                "type": "no_speech_detected",
                "session_id": session_id
            })
        
        # Processing complete
        await websocket.send_json({
            "type": "processing_complete",
            "session_id": session_id
        })
            
    except Exception as e:
        print(f"Audio processing error: {e}")
        await websocket.send_json({
            "type": "processing_error",
            "error": str(e),
            "session_id": session_id
        })

async def process_audio_chunk(audio_data, websocket, chunk_id):
    """Process individual audio chunk in real-time"""
    try:
        # Skip if audio data is too small
        if len(audio_data) < 1000:
            return
            
        # Quick STT processing
        audio_file = io.BytesIO(audio_data)
        # Try different extensions based on the audio data
        if len(audio_data) > 0:
            # Check if it looks like WAV (starts with RIFF)
            if audio_data[:4] == b'RIFF':
                audio_file.name = f"chunk_{chunk_id}.wav"
            else:
                audio_file.name = f"chunk_{chunk_id}.webm"
        
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file,
            response_format="json"
        )
        
        # Handle both text response and object response
        if isinstance(transcript, str):
            chunk_text = transcript.strip()
        else:
            chunk_text = getattr(transcript, 'text', '').strip()
        
        if chunk_text:
            # Send partial transcript immediately
            await websocket.send_json({
                "type": "partial_transcript",
                "text": chunk_text,
                "chunk_id": chunk_id
            })
            
            # Process with chatbot immediately (don't wait)
            asyncio.create_task(process_chunk_response(chunk_text, websocket, chunk_id))
            
    except Exception as e:
        print(f"Chunk processing error: {e}")
        # Don't fail completely, just skip this chunk

async def process_chunk_response(text, websocket, chunk_id, bot_id="default"):
    """Process chatbot response for chunk"""
    try:
        # Get bot response
        bot_response = await get_chatbot_response(text, bot_id)
        
        # Send response immediately
        await websocket.send_json({
            "type": "chunk_response",
            "text": bot_response,
            "chunk_id": chunk_id
        })
        
        # Generate TTS in background
        asyncio.create_task(generate_chunk_tts(bot_response, websocket, chunk_id))
        
    except Exception as e:
        print(f"Chunk response error: {e}")

async def generate_chunk_tts(text, websocket, chunk_id, lang="en"):
    """Generate TTS for chunk response"""
    try:
        clean_text = clean_text_for_tts(text)
        optimized_text = optimize_text_for_tts(clean_text)
        
        audio_content = await generate_tts_audio(optimized_text, lang)
        
        if audio_content:
            audio_b64 = base64.b64encode(audio_content).decode()
            await websocket.send_json({
                "type": "chunk_audio",
                "data": audio_b64,
                "chunk_id": chunk_id
            })
            
    except Exception as e:
        print(f"Chunk TTS error: {e}")

@app.websocket("/ws/voice-stream")
async def voice_stream_websocket(websocket: WebSocket, bot_id: str = "default"):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_bytes()
            
            audio_file = io.BytesIO(data)
            audio_file.name = "stream.wav"
            
            # STT with high accuracy language detection
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file,
                response_format="json"
            )
            
            user_text = getattr(transcript, 'text', '').strip()
            spoken_lang = getattr(transcript, 'language', None)
            
            await websocket.send_json({
                "type": "transcript",
                "text": user_text,
                "language": spoken_lang
            })
            
            # Ultra-fast parallel processing
            async def get_and_send_response():
                bot_response = await get_chatbot_response(user_text, bot_id)
                await websocket.send_json({
                    "type": "bot_response",
                    "text": bot_response
                })
                return bot_response
            
            async def process_tts(bot_response):
                clean_response = clean_text_for_tts(bot_response)
                optimized_response = optimize_text_for_tts(clean_response)
                await generate_tts_audio_streaming(optimized_response, websocket, spoken_lang)
                await websocket.send_json({"type": "audio_complete"})
            
            # Start chat response immediately
            response_task = asyncio.create_task(get_and_send_response())
            
            # When response is ready, start TTS in parallel
            bot_response = await response_task
            asyncio.create_task(process_tts(bot_response))
            
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()

@app.websocket("/ws/voice-stream-legacy")
async def voice_stream_legacy(websocket: WebSocket, bot_id: str = "default"):
    """Legacy endpoint with full audio response"""
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_bytes()
            
            audio_file = io.BytesIO(data)
            audio_file.name = "stream.wav"
            
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file,
                response_format="json"
            )
            
            user_text = getattr(transcript, 'text', '').strip()
            spoken_lang = getattr(transcript, 'language', None)
            
            await websocket.send_json({
                "type": "transcript",
                "text": user_text,
                "language": spoken_lang
            })
            
            async def process_response():
                bot_response = await get_chatbot_response(user_text, bot_id)
                
                await websocket.send_json({
                    "type": "bot_response",
                    "text": bot_response
                })
                
                clean_response = clean_text_for_tts(bot_response)
                optimized_response = optimize_text_for_tts(clean_response)
                
                audio_content = await generate_tts_audio(optimized_response, spoken_lang)
                if audio_content:
                    audio_b64 = base64.b64encode(audio_content).decode()
                    await websocket.send_json({
                        "type": "audio",
                        "data": audio_b64
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "TTS failed, but text response available"
                    })
            
            asyncio.create_task(process_response())
            
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()

@app.websocket("/ws/{bot_id}")
async def websocket_bot_endpoint_old(websocket: WebSocket, bot_id: str = "default"):
    await websocket.accept()
    print(f"WebSocket connected for bot_id: {bot_id}")
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "message":
                message = data.get("message", "")
                
                # Get chatbot response
                response_text = await get_chatbot_response(message, bot_id)
                
                # Send text response
                await websocket.send_json({
                    "type": "text_response",
                    "text": response_text,
                    "bot_id": bot_id
                })
                
                # Generate TTS audio
                cleaned_text = clean_text_for_tts(response_text)
                optimized_text = optimize_text_for_tts(cleaned_text)
                audio_content = await generate_tts_audio(optimized_text)
                
                if audio_content:
                    audio_b64 = base64.b64encode(audio_content).decode()
                    await websocket.send_json({
                        "type": "audio_response",
                        "data": audio_b64,
                        "bot_id": bot_id
                    })
                
    except Exception as e:
        print(f"WebSocket error for bot_id {bot_id}: {e}")
    finally:
        print(f"WebSocket disconnected for bot_id: {bot_id}")

@app.post("/voice-chat")
async def voice_chat(file: UploadFile = File(...), bot_id: str = "default"):
    try:
        print(f"Processing voice chat for bot_id: {bot_id}")
        
        # Read and validate audio
        audio_data = await file.read()
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        print(f"Audio file size: {len(audio_data)} bytes")
        
        # STT Processing
        audio_file = io.BytesIO(audio_data)
        audio_file.name = "audio.wav"
        
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file,
            response_format="json"
        )
        
        user_text = getattr(transcript, 'text', '').strip()
        spoken_lang = getattr(transcript, 'language', None)
        if not user_text:
            user_text = "Hello"
        
        print(f"User: {user_text} (Language: {spoken_lang})")
        
        # Get chatbot response
        bot_response = await get_chatbot_response(user_text, bot_id)
        print(f"Bot original: {bot_response}")
        
        # Clean and optimize response
        clean_response = clean_text_for_tts(bot_response)
        optimized_response = optimize_text_for_tts(clean_response)
        print(f"Bot optimized: {optimized_response}")
        
        # Generate TTS with detected language
        lang = spoken_lang if spoken_lang else "en"
        audio_content = await generate_tts_audio(optimized_response, lang)
        
        if audio_content:
            return Response(
                content=audio_content,
                media_type="audio/mpeg",
                headers={
                    "X-Transcript": user_text,
                    "X-Bot-Response": optimized_response
                }
            )
        else:
            return {
                "transcript": user_text,
                "response": optimized_response,
                "audio_error": "TTS generation failed"
            }
        
    except Exception as e:
        print(f"Voice chat error: {type(e).__name__}: {str(e)}")
        return {
            "transcript": "",
            "response": "I had trouble processing your request. Please try again.",
            "error": str(e)
        }

@app.get("/")
async def root():
    redis_status = "disabled"
    # try:
    #     await redis_client.ping()
    # except:
    #     redis_status = "disconnected"
    
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
            "voice_realtime": "/ws/voice-realtime (Real-time Processing)",
            "voice_stream": "/ws/voice-stream (Streaming TTS)",
            "voice_stream_legacy": "/ws/voice-stream-legacy (Full Audio)",
            "voice_chat": "/voice-chat",
            "stt": "/stt",
            "tts": "/tts",
            "relay": "/relay-message"
        }
    }

@app.on_event("shutdown")
async def shutdown_event():
    await connection_pool.aclose()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
