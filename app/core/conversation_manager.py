"""
Enhanced Conversation Manager - Advanced memory management and session handling
Implements continuous conversation support with multi-agent orchestration
"""
import asyncio
import json
import time
import uuid
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from collections import deque

# Enhanced imports for multi-agent support
from app.core.latency_optimizer import LatencyOptimizer
from speech_to_text.google_cloud_stt import GoogleCloudStreamingSTT, StreamingTranscriptionResult
from text_to_speech.google_cloud_tts import GoogleCloudTTS

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Enhanced conversation turn with multi-agent support"""
    turn_id: str
    timestamp: float
    user_input: str
    agent_id: str
    agent_response: str
    confidence: float
    latency_ms: float
    tools_used: List[str] = field(default_factory=list)
    context_retrieved: int = 0
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SessionContext:
    """Enhanced session context for multi-agent conversations"""
    session_id: str
    call_sid: str
    start_time: float
    current_agent: Optional[str] = None
    agent_history: List[str] = field(default_factory=list)
    conversation_turns: List[ConversationTurn] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    conversation_summary: str = ""
    total_latency: float = 0.0
    error_count: int = 0
    
    # Enhanced memory management
    short_term_memory: deque = field(default_factory=lambda: deque(maxlen=5))  # Last 5 turns
    long_term_memory: List[Dict[str, Any]] = field(default_factory=list)  # Key insights
    
    # Performance tracking
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

class EnhancedConversationManager:
    """
    Enhanced conversation manager with multi-agent orchestration support
    Handles continuous conversations with advanced memory management
    """
    
    def __init__(
        self,
        orchestrator,  # MultiAgentOrchestrator
        call_sid: str,
        session_id: str,
        websocket,
        latency_optimizer: Optional[LatencyOptimizer] = None
    ):
        self.orchestrator = orchestrator
        self.call_sid = call_sid
        self.session_id = session_id
        self.websocket = websocket
        self.latency_optimizer = latency_optimizer
        
        # Enhanced session context
        self.context = SessionContext(
            session_id=session_id,
            call_sid=call_sid,
            start_time=time.time()
        )
        
        # Conversation state
        self.conversation_active = True
        self.is_processing = False
        self.stream_sid: Optional[str] = None
        
        # Enhanced audio processing
        self.audio_buffer = bytearray()
        self.audio_chunk_count = 0
        self.last_transcription_time = 0.0
        self.silence_detection_threshold = 3.0  # seconds
        
        # Echo detection and prevention
        self.last_tts_time: Optional[float] = None
        self.echo_prevention_window = 2.0  # seconds
        self.last_response_text = ""
        
        # Performance optimization
        self.response_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Enhanced statistics
        self.stats = {
            "session_start": time.time(),
            "audio_chunks_processed": 0,
            "transcriptions_received": 0,
            "valid_transcriptions": 0,
            "responses_generated": 0,
            "agent_switches": 0,
            "tools_executed": 0,
            "average_latency": 0.0,
            "echo_detections": 0,
            "cache_hits": 0,
            "errors": 0
        }
        
        logger.info(f"🎯 Enhanced Conversation Manager initialized for session {session_id}")
    
    async def init(self):
        """Initialize conversation manager components"""
        logger.info(f"🔄 Initializing conversation manager for {self.session_id}")
        
        # Initialize STT with optimized settings
        self.stt_client = GoogleCloudStreamingSTT(
            language="en-US",
            sample_rate=8000,
            encoding="MULAW",
            channels=1,
            interim_results=False,  # Only final results for accuracy
            project_id=self.orchestrator.project_id if hasattr(self.orchestrator, 'project_id') else None,
            location="global",
            credentials_file=self.orchestrator.credentials_file
        )
        
        # Initialize TTS with enhanced settings
        self.tts_client = GoogleCloudTTS(
            credentials_file=self.orchestrator.credentials_file,
            voice_name="en-US-Neural2-C",
            voice_gender=None,  # Neural2 voices don't need gender
            language_code="en-US",
            container_format="mulaw",
            sample_rate=8000,
            enable_caching=True,
            voice_type="NEURAL2"
        )
        
        logger.info(f"✅ Conversation manager initialized for {self.session_id}")
    
    async def start_conversation(self, stream_sid: str):
        """Start the conversation with enhanced initialization"""
        self.stream_sid = stream_sid
        self.conversation_active = True
        
        logger.info(f"🎙️ Starting conversation for session {self.session_id}")
        
        # Start STT streaming
        await self.stt_client.start_streaming()
        
        # Send welcome message with dynamic agent selection
        welcome_message = await self._generate_welcome_message()
        await self._send_response(welcome_message)
        
        logger.info(f"✅ Conversation started for session {self.session_id}")
    
    async def _generate_welcome_message(self) -> str:
        """Generate contextual welcome message"""
        # Check for returning user or context
        if self.context.agent_history:
            return "Welcome back! I'm ready to continue helping you. What would you like to know?"
        else:
            return "Hello! I'm your AI assistant ready to help with any questions. What can I do for you today?"
    
    async def process_audio_chunk(self, data: Dict[str, Any]):
        """Process audio chunk with enhanced echo prevention and optimization"""
        if not self.conversation_active or self.is_processing:
            return
        
        # Extract audio data
        media = data.get('media', {})
        payload = media.get('payload')
        
        if not payload:
            return
        
        try:
            # Decode audio
            import base64
            audio_data = base64.b64decode(payload)
            self.audio_chunk_count += 1
            self.stats["audio_chunks_processed"] += 1
            
            # Enhanced echo prevention
            current_time = time.time()
            if self.last_tts_time and (current_time - self.last_tts_time) < self.echo_prevention_window:
                logger.debug("🔇 Skipping audio - echo prevention active")
                return
            
            # Process with STT
            await self.stt_client.process_audio_chunk(
                audio_data,
                callback=self._handle_transcription_result
            )
            
        except Exception as e:
            logger.error(f"❌ Error processing audio chunk: {e}")
            self.stats["errors"] += 1
    
    async def _handle_transcription_result(self, result: StreamingTranscriptionResult):
        """Handle transcription results with enhanced validation and processing"""
        if not result.is_final or not result.text.strip():
            return
        
        self.stats["transcriptions_received"] += 1
        transcription = result.text.strip()
        confidence = result.confidence
        
        logger.info(f"🎤 Transcription: '{transcription}' (confidence: {confidence:.2f})")
        
        # Enhanced validation with context awareness
        if not await self._is_valid_transcription(transcription, confidence):
            logger.debug(f"❌ Invalid transcription rejected: '{transcription}'")
            return
        
        self.stats["valid_transcriptions"] += 1
        self.last_transcription_time = time.time()
        
        # Process the validated transcription
        await self._process_user_input(transcription, confidence)
    
    async def _is_valid_transcription(self, transcription: str, confidence: float) -> bool:
        """Enhanced transcription validation with context awareness"""
        # Basic validation
        if len(transcription.strip()) < 2:
            return False
        
        # Confidence threshold (adaptive based on conversation context)
        base_threshold = 0.3
        context_boost = 0.1 if self.context.conversation_turns else 0
        threshold = base_threshold - context_boost
        
        if confidence < threshold:
            return False
        
        # Echo detection with enhanced algorithms
        if await self._is_likely_echo(transcription):
            self.stats["echo_detections"] += 1
            return False
        
        # Context-aware filtering
        if await self._is_context_inappropriate(transcription):
            return False
        
        return True
    
    async def _is_likely_echo(self, transcription: str) -> bool:
        """Enhanced echo detection using multiple heuristics"""
        current_time = time.time()
        
        # Time-based echo detection
        if self.last_tts_time and (current_time - self.last_tts_time) < self.echo_prevention_window:
            # Check for content similarity with last response
            if self.last_response_text:
                similarity = await self._calculate_text_similarity(transcription, self.last_response_text)
                if similarity > 0.7:  # 70% similarity threshold
                    return True
        
        # Pattern-based echo detection
        echo_patterns = [
            "i'm ready to help",
            "what would you like to know",
            "how can i assist",
            "voice assistant",
            "ai assistant"
        ]
        
        transcription_lower = transcription.lower()
        for pattern in echo_patterns:
            if pattern in transcription_lower:
                return True
        
        return False
    
    async def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    async def _is_context_inappropriate(self, transcription: str) -> bool:
        """Check if transcription is inappropriate for current context"""
        # Filter common false positives
        false_positives = [
            "um", "uh", "hmm", "ah", "er", "oh",
            "okay", "ok", "yes", "no", "hello", "hi"
        ]
        
        if transcription.lower().strip() in false_positives:
            return True
        
        # Check for very short responses that might be misheard
        if len(transcription.split()) == 1 and len(transcription) < 4:
            return True
        
        return False
    
    async def _process_user_input(self, transcription: str, confidence: float):
        """Process validated user input with multi-agent orchestration"""
        if self.is_processing:
            return
        
        self.is_processing = True
        start_time = time.time()
        
        try:
            # Check response cache first
            cache_key = self._generate_cache_key(transcription)
            cached_response = self._get_cached_response(cache_key)
            
            if cached_response:
                logger.info(f"📋 Cache hit for: '{transcription}'")
                self.stats["cache_hits"] += 1
                await self._send_response(cached_response["response"])
                return
            
            # Process through multi-agent orchestrator
            orchestration_start = time.time()
            
            result = await self.orchestrator.process_conversation_turn(
                session_id=self.session_id,
                user_input=transcription,
                call_sid=self.call_sid
            )
            
            orchestration_time = (time.time() - orchestration_start) * 1000
            
            if result.get("success", False):
                response_text = result.get("response", "")
                agent_used = result.get("agent_used", "unknown")
                tools_used = result.get("tools_used", [])
                total_latency = result.get("total_latency", 0)
                
                # Create conversation turn record
                turn = ConversationTurn(
                    turn_id=str(uuid.uuid4()),
                    timestamp=time.time(),
                    user_input=transcription,
                    agent_id=agent_used,
                    agent_response=response_text,
                    confidence=confidence,
                    latency_ms=total_latency,
                    tools_used=tools_used,
                    context_retrieved=len(result.get("context", [])),
                    success=True,
                    metadata={
                        "orchestration_time": orchestration_time,
                        "cache_used": False
                    }
                )
                
                # Update session context
                await self._update_session_context(turn)
                
                # Cache successful response
                self._cache_response(cache_key, {
                    "response": response_text,
                    "agent_id": agent_used,
                    "timestamp": time.time()
                })
                
                # Send response
                await self._send_response(response_text)
                
                # Track agent usage
                if agent_used != self.context.current_agent:
                    self.stats["agent_switches"] += 1
                    self.context.current_agent = agent_used
                    if agent_used not in self.context.agent_history:
                        self.context.agent_history.append(agent_used)
                
                self.stats["tools_executed"] += len(tools_used)
                
            else:
                # Handle error
                error_message = "I'm sorry, I had trouble processing that. Could you please try again?"
                await self._send_response(error_message)
                self.stats["errors"] += 1
            
            # Update performance metrics
            total_processing_time = (time.time() - start_time) * 1000
            await self._update_performance_metrics(total_processing_time)
            
        except Exception as e:
            logger.error(f"❌ Error processing user input: {e}")
            await self._send_response("I apologize, but I encountered an error. Please try again.")
            self.stats["errors"] += 1
        
        finally:
            self.is_processing = False
    
    def _generate_cache_key(self, transcription: str) -> str:
        """Generate cache key for response"""
        import hashlib
        
        # Include session context for personalized caching
        context_factors = [
            transcription.lower().strip(),
            self.context.current_agent or "none",
            str(len(self.context.conversation_turns))  # Conversation length
        ]
        
        cache_string = "|".join(context_factors)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if valid"""
        if cache_key not in self.response_cache:
            return None
        
        cached = self.response_cache[cache_key]
        
        # Check TTL
        if time.time() - cached["timestamp"] > self.cache_ttl:
            del self.response_cache[cache_key]
            return None
        
        return cached
    
    def _cache_response(self, cache_key: str, response_data: Dict[str, Any]):
        """Cache response for future use"""
        self.response_cache[cache_key] = response_data
        
        # Cleanup old cache entries
        current_time = time.time()
        keys_to_remove = [
            key for key, data in self.response_cache.items()
            if current_time - data["timestamp"] > self.cache_ttl
        ]
        
        for key in keys_to_remove:
            del self.response_cache[key]
    
    async def _update_session_context(self, turn: ConversationTurn):
        """Update session context with conversation turn"""
        # Add to conversation history
        self.context.conversation_turns.append(turn)
        
        # Update short-term memory
        self.context.short_term_memory.append({
            "user_input": turn.user_input,
            "agent_response": turn.agent_response,
            "agent_id": turn.agent_id,
            "timestamp": turn.timestamp
        })
        
        # Update long-term memory (extract key insights)
        if len(self.context.conversation_turns) % 5 == 0:  # Every 5 turns
            await self._extract_long_term_insights()
        
        # Update conversation summary
        if len(self.context.conversation_turns) % 10 == 0:  # Every 10 turns
            await self._update_conversation_summary()
    
    async def _extract_long_term_insights(self):
        """Extract insights for long-term memory"""
        recent_turns = list(self.context.short_term_memory)
        
        if len(recent_turns) < 3:
            return
        
        # Extract patterns and preferences
        insights = {
            "timestamp": time.time(),
            "turn_range": f"{len(self.context.conversation_turns)-4}-{len(self.context.conversation_turns)}",
            "dominant_agent": self._get_dominant_agent(recent_turns),
            "common_topics": self._extract_common_topics(recent_turns),
            "user_patterns": self._extract_user_patterns(recent_turns)
        }
        
        self.context.long_term_memory.append(insights)
        
        # Keep only last 20 insights
        if len(self.context.long_term_memory) > 20:
            self.context.long_term_memory = self.context.long_term_memory[-20:]
    
    def _get_dominant_agent(self, turns: List[Dict[str, Any]]) -> str:
        """Get the most used agent in recent turns"""
        agent_counts = {}
        for turn in turns:
            agent_id = turn.get("agent_id", "unknown")
            agent_counts[agent_id] = agent_counts.get(agent_id, 0) + 1
        
        return max(agent_counts, key=agent_counts.get) if agent_counts else "unknown"
    
    def _extract_common_topics(self, turns: List[Dict[str, Any]]) -> List[str]:
        """Extract common topics from recent conversation"""
        # Simple keyword extraction
        all_text = " ".join([
            turn.get("user_input", "") + " " + turn.get("agent_response", "")
            for turn in turns
        ]).lower()
        
        # Common topic keywords
        topic_keywords = {
            "pricing": ["price", "cost", "plan", "subscription", "payment"],
            "technical": ["error", "problem", "issue", "bug", "setup"],
            "features": ["feature", "capability", "function", "option"],
            "support": ["help", "assistance", "support", "question"]
        }
        
        detected_topics = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in all_text for keyword in keywords):
                detected_topics.append(topic)
        
        return detected_topics
    
    def _extract_user_patterns(self, turns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract user communication patterns"""
        user_inputs = [turn.get("user_input", "") for turn in turns]
        
        return {
            "avg_input_length": sum(len(inp.split()) for inp in user_inputs) / len(user_inputs),
            "question_ratio": sum(1 for inp in user_inputs if "?" in inp) / len(user_inputs),
            "politeness_indicators": sum(1 for inp in user_inputs if any(word in inp.lower() for word in ["please", "thank", "sorry"])) / len(user_inputs)
        }
    
    async def _update_conversation_summary(self):
        """Update high-level conversation summary"""
        if not self.context.conversation_turns:
            return
        
        # Generate summary based on conversation patterns
        recent_topics = []
        recent_agents = []
        
        for turn in self.context.conversation_turns[-10:]:  # Last 10 turns
            if turn.tools_used:
                recent_topics.extend(turn.tools_used)
            recent_agents.append(turn.agent_id)
        
        # Create summary
        dominant_agent = max(set(recent_agents), key=recent_agents.count) if recent_agents else "unknown"
        summary_parts = [
            f"Conversation with {len(self.context.conversation_turns)} turns",
            f"Primary agent: {dominant_agent}",
        ]
        
        if recent_topics:
            summary_parts.append(f"Tools used: {', '.join(set(recent_topics))}")
        
        self.context.conversation_summary = ". ".join(summary_parts)
    
    async def _update_performance_metrics(self, processing_time: float):
        """Update performance metrics"""
        # Update running average
        if self.stats["responses_generated"] == 0:
            self.stats["average_latency"] = processing_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats["average_latency"] = (
                alpha * processing_time + 
                (1 - alpha) * self.stats["average_latency"]
            )
        
        self.stats["responses_generated"] += 1
        self.context.total_latency += processing_time
        
        # Track with latency optimizer
        if self.latency_optimizer:
            await self.latency_optimizer.record_processing_time(
                self.session_id,
                "conversation_processing", 
                processing_time
            )
    
    async def _send_response(self, text: str):
        """Send response with enhanced delivery and tracking"""
        if not text.strip() or not self.conversation_active:
            return
        
        try:
            logger.info(f"💬 Sending response: '{text}'")
            
            # Track TTS for echo prevention
            self.last_tts_time = time.time()
            self.last_response_text = text
            
            # Convert to speech
            audio_data = await self.tts_client.synthesize(text)
            
            if audio_data:
                # Send audio in optimized chunks
                await self._send_audio_chunks(audio_data)
                logger.info(f"✅ Response sent successfully ({len(audio_data)} bytes)")
            else:
                logger.error("❌ No audio data generated")
                
        except Exception as e:
            logger.error(f"❌ Error sending response: {e}")
            self.stats["errors"] += 1
    
    async def _send_audio_chunks(self, audio_data: bytes):
        """Send audio data in optimized chunks"""
        if not self.stream_sid:
            logger.warning("❌ Cannot send audio: missing stream_sid")
            return
        
        chunk_size = 400  # 50ms chunks for smooth playback
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            
            try:
                import base64
                audio_base64 = base64.b64encode(chunk).decode('utf-8')
                
                message = {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": audio_base64}
                }
                
                await self.websocket.send_text(json.dumps(message))
                
                # Small delay for smooth playback
                await asyncio.sleep(0.025)  # 25ms delay
                
            except Exception as e:
                logger.error(f"❌ Error sending audio chunk: {e}")
                break
    
    def get_session_context(self) -> Dict[str, Any]:
        """Get comprehensive session context"""
        return {
            "session_id": self.context.session_id,
            "call_sid": self.context.call_sid,
            "duration": time.time() - self.context.start_time,
            "conversation_turns": len(self.context.conversation_turns),
            "current_agent": self.context.current_agent,
            "agent_history": self.context.agent_history,
            "conversation_summary": self.context.conversation_summary,
            "user_preferences": self.context.user_preferences,
            "performance_metrics": {
                "total_latency": self.context.total_latency,
                "average_latency": self.stats["average_latency"],
                "error_count": self.context.error_count
            },
            "memory_insights": {
                "short_term_items": len(self.context.short_term_memory),
                "long_term_insights": len(self.context.long_term_memory),
                "last_insight": self.context.long_term_memory[-1] if self.context.long_term_memory else None
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive conversation statistics"""
        current_time = time.time()
        session_duration = current_time - self.stats["session_start"]
        
        return {
            "session_id": self.context.session_id,
            "call_sid": self.context.call_sid,
            "session_duration": session_duration,
            "conversation_active": self.conversation_active,
            "current_agent": self.context.current_agent,
            
            # Processing statistics
            "audio_chunks_processed": self.stats["audio_chunks_processed"],
            "transcriptions_received": self.stats["transcriptions_received"],
            "valid_transcriptions": self.stats["valid_transcriptions"],
            "responses_generated": self.stats["responses_generated"],
            
            # Performance metrics
            "average_latency": self.stats["average_latency"],
            "total_latency": self.context.total_latency,
            "cache_hits": self.stats["cache_hits"],
            "cache_hit_rate": (self.stats["cache_hits"] / max(self.stats["valid_transcriptions"], 1)) * 100,
            
            # Agent and tool usage
            "agent_switches": self.stats["agent_switches"],
            "tools_executed": self.stats["tools_executed"],
            "agent_history": self.context.agent_history,
            
            # Quality metrics
            "echo_detections": self.stats["echo_detections"],
            "transcription_accuracy": (self.stats["valid_transcriptions"] / max(self.stats["transcriptions_received"], 1)) * 100,
            "error_count": self.stats["errors"],
            "success_rate": ((self.stats["responses_generated"] - self.stats["errors"]) / max(self.stats["responses_generated"], 1)) * 100,
            
            # Conversation insights
            "conversation_turns": len(self.context.conversation_turns),
            "conversation_summary": self.context.conversation_summary,
            "memory_items": {
                "short_term": len(self.context.short_term_memory),
                "long_term": len(self.context.long_term_memory)
            }
        }
    
    async def cleanup(self):
        """Enhanced cleanup with session preservation"""
        logger.info(f"🧹 Cleaning up conversation manager for {self.session_id}")
        
        try:
            self.conversation_active = False
            
            # Stop STT streaming
            if hasattr(self, 'stt_client') and self.stt_client:
                if self.stt_client.is_streaming:
                    await self.stt_client.stop_streaming()
            
            # Generate final session summary
            final_stats = self.get_stats()
            
            # Log session summary
            logger.info(f"📊 Session {self.session_id} summary:")
            logger.info(f"  Duration: {final_stats['session_duration']:.1f}s")
            logger.info(f"  Turns: {final_stats['conversation_turns']}")
            logger.info(f"  Average latency: {final_stats['average_latency']:.1f}ms")
            logger.info(f"  Success rate: {final_stats['success_rate']:.1f}%")
            logger.info(f"  Agents used: {', '.join(self.context.agent_history) if self.context.agent_history else 'None'}")
            
            # Store session data for analytics (if needed)
            await self._store_session_data()
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
        
        logger.info(f"✅ Cleanup complete for session {self.session_id}")
    
    async def _store_session_data(self):
        """Store session data for analytics and learning"""
        try:
            # Create comprehensive session record
            session_record = {
                "session_id": self.context.session_id,
                "call_sid": self.context.call_sid,
                "timestamp": time.time(),
                "duration": time.time() - self.context.start_time,
                "context": self.get_session_context(),
                "stats": self.get_stats(),
                "conversation_turns": [
                    {
                        "turn_id": turn.turn_id,
                        "timestamp": turn.timestamp,
                        "user_input": turn.user_input,
                        "agent_id": turn.agent_id,
                        "agent_response": turn.agent_response,
                        "confidence": turn.confidence,
                        "latency_ms": turn.latency_ms,
                        "tools_used": turn.tools_used,
                        "success": turn.success
                    }
                    for turn in self.context.conversation_turns
                ]
            }
            
            # In a production system, this would be stored in a database
            # For now, we'll log it for analysis
            logger.debug(f"📋 Session record created for {self.session_id}")
            
        except Exception as e:
            logger.error(f"❌ Error storing session data: {e}")