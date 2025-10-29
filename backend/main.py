from fastapi import FastAPI, HTTPException, File, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import openai

# Request models
class ChatRequest(BaseModel):
    message: str
    bot_id: str = "default"

class RelayRequest(BaseModel):
    message: str
    bot_id: str = "default"
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
    allow_origins=["*"],
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
    "dr-tomar": "http://localhost:8000/api/chat",
    "edmonds": "http://localhost:8001/api/chat", 
    "project2": "http://localhost:8002/api/chat",
    "project3": "http://localhost:8003/api/chat",
    "dental-clinic": "http://localhost:8004/api/chat",
    "law-firm": "http://localhost:8005/api/chat",
    "restaurant": "http://localhost:8006/api/chat",
    "default": "http://localhost:8000/api/chat"
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
CACHE_TTL = 1296000  # 15 days
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
        text = unicodedata.normalize('NFKD', text)
        ascii_text = ''.join(c for c in text if ord(c) < 128)
        
        ascii_text = re.sub(r'&#\d+;', '', ascii_text)
        ascii_text = re.sub(r'&\w+;', '', ascii_text)
        ascii_text = re.sub(r'\s+', ' ', ascii_text).strip()
        
        if len(ascii_text) < 3:
            return "I apologize for the formatting issue."
            
        return ascii_text
        
    except Exception as e:
        print(f"Cleaning error: {e}")
        return "There was a text processing error."

async def get_chatbot_response(message, bot_id="default"):
    message_lower = message.lower().strip()
    
    # Get chatbot URL based on bot_id
    chatbot_url = CHATBOT_URLS.get(bot_id, CHATBOT_URLS["default"])
    print(f"Using chatbot URL for bot_id '{bot_id}': {chatbot_url}")
    
    # Fast common responses first
    for key, response in COMMON_RESPONSES.items():
        if key in message_lower:
            return response
    
    # Try chatbot API - test different payload formats
    try:
        # First try with simple message payload
        payload = {"message": message}
        
        print(f"Calling chatbot API with payload: {payload}")
        
        # Direct API call without timeout wrapper
        response = await connection_pool.post(
            chatbot_url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Chatbot API response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Chatbot API response: {result}")
            return result.get("response", result.get("answer", "I received your message and I'm processing it."))
        else:
            response_text = response.text
            print(f"Chatbot API error: {response.status_code} - {response_text}")
            
            # Try with session_id if first attempt fails
            import uuid
            session_id = str(uuid.uuid4())[:8]
            payload_with_session = {
                "message": message,
                "session_id": session_id
            }
            
            print(f"Retrying with session_id: {payload_with_session}")
            
            response2 = await connection_pool.post(
                chatbot_url,
                json=payload_with_session,
                headers={"Content-Type": "application/json"}
            )
            
            if response2.status_code == 200:
                result = response2.json()
                print(f"Chatbot API response (retry): {result}")
                return result.get("response", result.get("answer", "I received your message and I'm processing it."))
            else:
                print(f"Retry also failed: {response2.status_code} - {response2.text}")
            
    except httpx.TimeoutException:
        print("Chatbot API timeout - using fallback")
    except httpx.ConnectError as e:
        print(f"Chatbot API connection refused: {str(e)}")
    except Exception as e:
        print(f"Chatbot API unexpected error: {type(e).__name__}: {str(e)}")
    
    # Enhanced fallback responses
    """if any(word in message_lower for word in ["office", "hours", "open", "time"]):
        return "Our office hours are Monday to Friday 9 AM to 6 PM, and Saturday 9 AM to 2 PM. How can I help you?"
    elif any(word in message_lower for word in ["appointment", "schedule", "book", "see"]):
        return "I'd be happy to help you schedule an appointment. Please call us at 425-775-5162 or let me know your preferred date."
    elif "help" in message_lower:
        return "I'm here to assist you. What can I help you with today?"
    else:
        return "Thank you for your question. For detailed information, please call our office at 425-775-5162."
"""
# API Endpoints
@app.get("/session-ephemeral")
async def get_session_ephemeral(bot_id: str = "default"):
    """Create ephemeral OpenAI Realtime token"""
    try:
        response = client.beta.realtime.sessions.create(
            model="gpt-4o-realtime-preview-2024-10-01",
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
        response = await get_chatbot_response(request.message, request.bot_id)
        return {"response": response, "bot_id": request.bot_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

async def generate_single_tts_chunk(chunk, chunk_index):
    """Generate TTS for a single chunk"""
    cache_key = get_cache_key(chunk)
    
    # Check cache first
    cached_audio = await get_redis_cache(cache_key)
    if cached_audio:
        return chunk_index, base64.b64decode(cached_audio), chunk
    
    local_cached = get_cached_tts(chunk)
    if local_cached:
        return chunk_index, local_cached, chunk
    
    try:
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice="nova",
            input=chunk
        )
        audio_content = response.content
        
        # Cache the chunk
        cache_tts(chunk, audio_content)
        await set_redis_cache(cache_key, base64.b64encode(audio_content).decode())
        
        return chunk_index, audio_content, chunk
    except Exception as e:
        print(f"TTS Chunk Error: {e}")
        return chunk_index, None, chunk

async def generate_tts_audio_streaming(text, websocket):
    """Generate TTS audio in parallel chunks and stream to websocket"""
    chunks = split_text_for_streaming(text, 150)
    
    # Generate all chunks in parallel
    tasks = [
        generate_single_tts_chunk(chunk, i) 
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



async def generate_tts_audio(text):
    cache_key = get_cache_key(text)
    
    # Check Redis first
    cached_audio = await get_redis_cache(cache_key)
    if cached_audio:
        return base64.b64decode(cached_audio)
    
    # Check local cache
    local_cached = get_cached_tts(text)
    if local_cached:
        return local_cached
    
    try:
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice="nova",
            input=text
        )
        audio_content = response.content
        
        # Cache in both Redis and local
        cache_tts(text, audio_content)
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
            model="whisper-1",
            file=audio_file
        )
        
        return {"text": transcript.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT failed: {str(e)}")

@app.post("/tts")
async def text_to_speech(request: dict):
    try:
        text = request.get("text", "")
        clean_text = clean_text_for_tts(text)
        optimized_text = optimize_text_for_tts(clean_text)
        
        audio_content = await generate_tts_audio(optimized_text)
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
        
        await asyncio.sleep(3.0)
        if is_recording:
            print(f"Auto-stopping recording after 3 seconds")
            
            # Stop recording first
            is_recording = False
            
            await websocket.send_json({
                "type": "recording_stopped",
                "duration": 3.0,
                "session_id": session_id,
                "reason": "auto_stop"
            })
            
            # Process complete audio recording
            if len(audio_buffer) > 0:
                asyncio.create_task(process_complete_audio(
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
                
                # Check if we hit 3 seconds (backup check)
                recording_duration = current_time - recording_start_time
                if recording_duration >= 3.0:
                    is_recording = False
                    if auto_stop_task:
                        auto_stop_task.cancel()
                    
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
        # STT processing
        audio_file = io.BytesIO(audio_data)
        audio_file.name = f"realtime_{session_id}.wav"
        
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
            language="en"
        )
        
        user_text = transcript.text.strip()
        
        if user_text:
            # Send transcript immediately
            await websocket.send_json({
                "type": "transcript",
                "text": user_text,
                "session_id": session_id
            })
            
            # Get chatbot response and TTS in parallel
            async def get_response():
                bot_response = await get_chatbot_response(user_text, bot_id)
                await websocket.send_json({
                    "type": "bot_response",
                    "text": bot_response,
                    "session_id": session_id
                })
                return bot_response
            
            async def generate_audio(bot_text):
                clean_text = clean_text_for_tts(bot_text)
                optimized_text = optimize_text_for_tts(clean_text)
                audio_content = await generate_tts_audio(optimized_text)
                
                if audio_content:
                    audio_b64 = base64.b64encode(audio_content).decode()
                    await websocket.send_json({
                        "type": "audio_response",
                        "data": audio_b64,
                        "session_id": session_id
                    })
            
            # Run response and TTS generation
            bot_response = await get_response()
            await generate_audio(bot_response)
            
        else:
            await websocket.send_json({
                "type": "no_speech_detected",
                "session_id": session_id
            })
        
        # Processing complete - ready for next
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
        # Detect format and use appropriate extension
        if len(audio_data) > 4 and audio_data[:4] == b'RIFF':
            audio_file.name = f"session_{session_id}.wav"
        else:
            audio_file.name = f"session_{session_id}.webm"
        
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
            language="en"
        )
        
        # Handle transcript response safely
        try:
            if hasattr(transcript, 'text'):
                user_text = transcript.text.strip()
            elif isinstance(transcript, str):
                user_text = transcript.strip()
            else:
                user_text = str(transcript).strip()
        except Exception as e:
            print(f"Transcript parsing error: {e}")
            user_text = ""
        
        if user_text:
            # Send transcript
            await websocket.send_json({
                "type": "transcript",
                "text": user_text,
                "session_id": session_id
            })
            
            # Get chatbot response using bot_id
            bot_response = await get_chatbot_response(user_text, bot_id)
            
            # Send bot response
            await websocket.send_json({
                "type": "bot_response",
                "text": bot_response,
                "session_id": session_id
            })
            
            # Generate TTS
            clean_text = clean_text_for_tts(bot_response)
            optimized_text = optimize_text_for_tts(clean_text)
            
            audio_content = await generate_tts_audio(optimized_text)
            
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
            model="whisper-1",
            file=audio_file,
            response_format="text",
            language="en"
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

async def process_chunk_response(text, websocket, chunk_id):
    """Process chatbot response for chunk"""
    try:
        # Get bot response
        bot_response = await get_chatbot_response(text)
        
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

async def generate_chunk_tts(text, websocket, chunk_id):
    """Generate TTS for chunk response"""
    try:
        clean_text = clean_text_for_tts(text)
        optimized_text = optimize_text_for_tts(clean_text)
        
        audio_content = await generate_tts_audio(optimized_text)
        
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
async def voice_stream_websocket(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_bytes()
            
            audio_file = io.BytesIO(data)
            audio_file.name = "stream.wav"
            
            # Faster STT with language hint
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
                language="en"
            )
            
            user_text = transcript.text
            
            await websocket.send_json({
                "type": "transcript",
                "text": user_text
            })
            
            # Ultra-fast parallel processing
            async def get_and_send_response():
                bot_response = await get_chatbot_response(user_text)
                await websocket.send_json({
                    "type": "bot_response",
                    "text": bot_response
                })
                return bot_response
            
            async def process_tts(bot_response):
                clean_response = clean_text_for_tts(bot_response)
                optimized_response = optimize_text_for_tts(clean_response)
                await generate_tts_audio_streaming(optimized_response, websocket)
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
async def voice_stream_legacy(websocket: WebSocket):
    """Legacy endpoint with full audio response"""
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_bytes()
            
            audio_file = io.BytesIO(data)
            audio_file.name = "stream.wav"
            
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            
            user_text = transcript.text
            
            await websocket.send_json({
                "type": "transcript",
                "text": user_text
            })
            
            async def process_response():
                bot_response = await get_chatbot_response(user_text)
                
                await websocket.send_json({
                    "type": "bot_response",
                    "text": bot_response
                })
                
                clean_response = clean_text_for_tts(bot_response)
                optimized_response = optimize_text_for_tts(clean_response)
                
                audio_content = await generate_tts_audio(optimized_response)
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
            model="whisper-1",
            file=audio_file
        )
        
        user_text = transcript.text.strip()
        if not user_text:
            user_text = "Hello"
        
        print(f"User: {user_text}")
        
        # Get chatbot response
        bot_response = await get_chatbot_response(user_text)
        print(f"Bot original: {bot_response}")
        
        # Clean and optimize response
        clean_response = clean_text_for_tts(bot_response)
        optimized_response = optimize_text_for_tts(clean_response)
        print(f"Bot optimized: {optimized_response}")
        
        # Generate TTS
        audio_content = await generate_tts_audio(optimized_response)
        
        if audio_content:
            return StreamingResponse(
                io.BytesIO(audio_content),
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
            "cache_ttl": "15 days"
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