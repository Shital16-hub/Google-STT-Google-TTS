# twilio_app_openai_pinecone.py
#!/usr/bin/env python3
"""
Optimized Twilio application using OpenAI LLM + Pinecone for sub-2-second latency.
"""
import os
import sys
import asyncio
import logging
import json
import time
from flask import Flask, request, Response, jsonify
from simple_websocket import Server
from dotenv import load_dotenv
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream

import simple_websocket

# Import optimized components
from telephony.simple_websocket_handler import SimpleWebSocketHandler
from voice_ai_agent import VoiceAIAgent
from integration.pipeline import VoiceAIAgentPipeline
from integration.tts_integration import TTSIntegration

# Load environment variables
load_dotenv()

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger('google.cloud').setLevel(logging.WARNING)
logging.getLogger('grpc').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)

# Flask app
app = Flask(__name__)

# Global variables
voice_ai_agent = None
base_url = None
active_calls = {}
call_metrics = {}

async def initialize_optimized_system():
    """Initialize Voice AI system with OpenAI + Pinecone for minimal latency."""
    global voice_ai_agent, base_url
    
    logger.info("Initializing optimized Voice AI system (OpenAI + Pinecone)...")
    
    # Validate environment
    base_url = os.getenv('BASE_URL')
    if not base_url:
        raise ValueError("BASE_URL environment variable must be set")
    
    openai_key = os.getenv('OPENAI_API_KEY')
    pinecone_key = os.getenv('PINECONE_API_KEY')
    
    if not openai_key:
        raise ValueError("OPENAI_API_KEY environment variable must be set")
    if not pinecone_key:
        raise ValueError("PINECONE_API_KEY environment variable must be set")
    
    # Initialize optimized Voice AI Agent
    voice_ai_agent = VoiceAIAgent(
        openai_api_key=openai_key,
        pinecone_api_key=pinecone_key,
        pinecone_index_name=os.getenv('PINECONE_INDEX_NAME', 'voice-ai-knowledge'),
        openai_model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
        credentials_file=os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    )
    
    # Initialize with minimal startup time
    await voice_ai_agent.init()
    
    logger.info("Optimized system ready - Target latency: <2 seconds")

@app.route('/', methods=['GET'])
def index():
    """Health check with performance metrics."""
    return {
        "status": "running",
        "system": "OpenAI + Pinecone",
        "target_latency": "< 2 seconds",
        "active_calls": len(active_calls),
        "version": "3.0.0"
    }

@app.route('/health', methods=['GET'])
def health_check():
    """Detailed health check with latency metrics."""
    health_data = {
        "status": "healthy" if voice_ai_agent and voice_ai_agent.initialized else "initializing",
        "timestamp": time.time(),
        "system": {
            "llm": "OpenAI GPT-4o-mini",
            "vector_db": "Pinecone",
            "stt": "Google Cloud STT v2",
            "tts": "Google Cloud TTS"
        },
        "active_calls": len(active_calls),
        "performance": {
            "target_latency": "2.0s",
            "calls_processed": len(call_metrics)
        }
    }
    
    # Add average latency if we have metrics
    if call_metrics:
        latencies = [metrics.get('total_time', 0) for metrics in call_metrics.values()]
        health_data["performance"]["avg_latency"] = f"{sum(latencies) / len(latencies):.2f}s"
        health_data["performance"]["under_target"] = len([l for l in latencies if l < 2.0])
    
    return jsonify(health_data)

@app.route('/voice/incoming', methods=['POST'])
def handle_incoming_call():
    """Handle incoming voice calls with optimized TwiML."""
    call_sid = request.form.get('CallSid')
    from_number = request.form.get('From')
    
    logger.info(f"Incoming call: {call_sid} from {from_number}")
    
    if not voice_ai_agent or not voice_ai_agent.initialized:
        logger.error("System not initialized")
        response = VoiceResponse()
        response.say("System is initializing. Please try again in a moment.", voice="alice")
        response.hangup()
        return Response(str(response), mimetype='text/xml')
    
    try:
        # Create optimized TwiML for minimal latency
        response = VoiceResponse()
        
        # WebSocket URL for media stream
        ws_url = f'{base_url.replace("https://", "wss://")}/ws/stream/{call_sid}'
        
        # Create bidirectional stream for immediate processing
        connect = Connect()
        stream = Stream(
            url=ws_url,
            track="inbound_track"
        )
        connect.append(stream)
        response.append(connect)
        
        # Initialize call metrics
        call_metrics[call_sid] = {
            "start_time": time.time(),
            "from_number": from_number
        }
        
        logger.info(f"Created optimized TwiML for call {call_sid}")
        return Response(str(response), mimetype='text/xml')
        
    except Exception as e:
        logger.error(f"Error handling call: {e}")
        response = VoiceResponse()
        response.say("An error occurred. Please try again.", voice="alice")
        response.hangup()
        return Response(str(response), mimetype='text/xml')

@app.route('/voice/status', methods=['POST'])
def handle_status_callback():
    """Handle call status with latency tracking."""
    call_sid = request.form.get('CallSid')
    call_status = request.form.get('CallStatus')
    call_duration = request.form.get('CallDuration', '0')
    
    logger.info(f"Call {call_sid} status: {call_status}")
    
    # Update metrics
    if call_sid in call_metrics:
        call_metrics[call_sid]["status"] = call_status
        call_metrics[call_sid]["duration"] = call_duration
        call_metrics[call_sid]["end_time"] = time.time()
    
    # Cleanup on call end
    if call_status in ['completed', 'failed', 'busy', 'no-answer']:
        if call_sid in active_calls:
            try:
                # Cleanup handler
                handler = active_calls[call_sid]
                asyncio.create_task(handler._cleanup())
                del active_calls[call_sid]
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
        
        # Keep metrics for analysis
        logger.info(f"Call {call_sid} ended after {call_duration}s")
    
    return Response('', status=204)

@app.route('/ws/stream/<call_sid>', websocket=True)
def handle_media_stream(call_sid):
    """Handle WebSocket with optimized processing pipeline."""
    logger.info(f"WebSocket connection for call {call_sid}")
    
    if not voice_ai_agent or not voice_ai_agent.initialized:
        logger.error("System not initialized")
        return ""
    
    ws = None
    handler = None
    
    try:
        # Accept WebSocket connection
        ws = Server.accept(request.environ)
        logger.info(f"WebSocket established for {call_sid}")
        
        # Create optimized handler
        handler = SimpleWebSocketHandler(call_sid, voice_ai_agent)
        active_calls[call_sid] = handler
        
        # Process messages with latency tracking
        while True:
            try:
                message = ws.receive(timeout=30.0)
                
                if message is None:
                    continue
                
                # Parse message
                try:
                    data = json.loads(message)
                    event_type = data.get('event')
                    
                    if event_type == 'connected':
                        logger.info(f"Media connected: {call_sid}")
                        
                    elif event_type == 'start':
                        stream_sid = data.get('streamSid')
                        handler.stream_sid = stream_sid
                        
                        # Start conversation with minimal delay
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(handler.start_conversation(ws))
                        finally:
                            loop.close()
                        
                        logger.info(f"Conversation started: {call_sid}")
                        
                    elif event_type == 'media':
                        # Process audio with latency tracking
                        process_start = time.time()
                        
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(handler._handle_audio(data, ws))
                        finally:
                            loop.close()
                        
                        # Track processing time
                        process_time = time.time() - process_start
                        if call_sid in call_metrics:
                            if 'audio_processing_times' not in call_metrics[call_sid]:
                                call_metrics[call_sid]['audio_processing_times'] = []
                            call_metrics[call_sid]['audio_processing_times'].append(process_time)
                        
                    elif event_type == 'stop':
                        logger.info(f"Stream stopped: {call_sid}")
                        
                        # Final cleanup
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(handler._cleanup())
                        finally:
                            loop.close()
                        break
                        
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from {call_sid}")
                    continue
                    
            except simple_websocket.ws.ConnectionClosed:
                logger.info(f"WebSocket closed: {call_sid}")
                break
            except Exception as e:
                logger.error(f"WebSocket error for {call_sid}: {e}")
                break
        
    except Exception as e:
        logger.error(f"Error in media stream for {call_sid}: {e}")
    
    finally:
        # Cleanup
        if handler:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(handler._cleanup())
                finally:
                    loop.close()
            except Exception as e:
                logger.error(f"Error in handler cleanup: {e}")
        
        if call_sid in active_calls:
            del active_calls[call_sid]
        
        if ws:
            try:
                ws.close()
            except:
                pass
        
        logger.info(f"WebSocket cleanup complete: {call_sid}")
        return ""

@app.route('/stats', methods=['GET'])
def get_system_stats():
    """Get comprehensive system statistics."""
    stats = {
        "timestamp": time.time(),
        "system": {
            "llm": "OpenAI GPT-4o-mini",
            "vector_db": "Pinecone",
            "target_latency": "2.0s"
        },
        "calls": {
            "active": len(active_calls),
            "total_processed": len(call_metrics)
        },
        "performance": {}
    }
    
    # Calculate performance metrics
    if call_metrics:
        durations = []
        latencies = []
        
        for call_id, metrics in call_metrics.items():
            if 'end_time' in metrics and 'start_time' in metrics:
                durations.append(metrics['end_time'] - metrics['start_time'])
            
            if 'audio_processing_times' in metrics:
                latencies.extend(metrics['audio_processing_times'])
        
        if durations:
            stats["performance"]["avg_call_duration"] = f"{sum(durations) / len(durations):.2f}s"
        
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            stats["performance"]["avg_audio_latency"] = f"{avg_latency:.2f}s"
            stats["performance"]["under_target_ratio"] = f"{len([l for l in latencies if l < 2.0]) / len(latencies) * 100:.1f}%"
    
    # Add individual call stats
    active_call_stats = {}
    for call_sid, handler in active_calls.items():
        try:
            active_call_stats[call_sid] = handler.get_stats()
        except Exception as e:
            active_call_stats[call_sid] = {"error": str(e)}
    
    stats["active_calls"] = active_call_stats
    
    return jsonify(stats)

@app.route('/performance', methods=['GET'])
def get_performance_metrics():
    """Get detailed performance analytics."""
    performance_data = {
        "target_latency": 2.0,
        "total_calls": len(call_metrics),
        "active_calls": len(active_calls)
    }
    
    # Calculate detailed metrics
    if call_metrics:
        all_latencies = []
        successful_calls = 0
        
        for metrics in call_metrics.values():
            if 'audio_processing_times' in metrics:
                all_latencies.extend(metrics['audio_processing_times'])
            
            if metrics.get('status') == 'completed':
                successful_calls += 1
        
        if all_latencies:
            performance_data.update({
                "avg_latency": round(sum(all_latencies) / len(all_latencies), 2),
                "min_latency": round(min(all_latencies), 2),
                "max_latency": round(max(all_latencies), 2),
                "under_target_count": len([l for l in all_latencies if l < 2.0]),
                "under_target_percentage": round(len([l for l in all_latencies if l < 2.0]) / len(all_latencies) * 100, 1)
            })
        
        performance_data["success_rate"] = round(successful_calls / len(call_metrics) * 100, 1)
    
    return jsonify(performance_data)

def init_system():
    """Initialize system synchronously."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(initialize_optimized_system())
        logger.info("Optimized system initialization complete")
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        sys.exit(1)
    finally:
        loop.close()

# Error handlers
@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({"error": "Internal server error", "system": "OpenAI + Pinecone"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found", "system": "OpenAI + Pinecone"}), 404

if __name__ == '__main__':
    print("Starting Optimized Voice AI Agent (OpenAI + Pinecone)")
    print(f"Target Latency: < 2 seconds")
    print(f"Base URL: {os.getenv('BASE_URL', 'Not set')}")
    print(f"OpenAI Model: {os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}")
    print(f"Pinecone Index: {os.getenv('PINECONE_INDEX_NAME', 'voice-ai-knowledge')}")
    
    # Initialize optimized system
    init_system()
    
    # Run Flask app
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting optimized server on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True)