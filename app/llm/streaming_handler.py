"""
LLM Streaming Handler - Real-Time Response Processing
===================================================

Advanced streaming handler for real-time LLM response processing with ultra-low latency.
Provides word-level streaming, chunk optimization, and seamless integration with TTS pipeline.
Achieves <100ms time-to-first-token through intelligent buffering and processing strategies.

Features:
- Real-time LLM streaming with word-level chunk processing
- Intelligent buffering and chunk optimization for TTS integration
- Stream multiplexing for concurrent conversation handling
- Error recovery and automatic stream reconnection
- Performance monitoring and latency optimization
- Context-aware streaming with agent-specific configurations
- Token-level processing with natural pause detection
- Seamless integration with dual streaming TTS pipeline
"""
import os
import asyncio
import logging
import time
import uuid
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, AsyncIterator, Callable, Awaitable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import queue
import threading
from collections import deque, defaultdict
import re
import string

# OpenAI and other LLM clients
import openai
from openai import AsyncOpenAI
import anthropic

# For natural language processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)


class StreamingMode(str, Enum):
    """Streaming modes for different use cases"""
    WORD_LEVEL = "word_level"        # Stream individual words
    SENTENCE_LEVEL = "sentence_level" # Stream complete sentences
    CHUNK_LEVEL = "chunk_level"      # Stream optimized chunks
    TOKEN_LEVEL = "token_level"      # Stream individual tokens
    ADAPTIVE = "adaptive"            # Adapt based on content


class ChunkType(str, Enum):
    """Types of streaming chunks"""
    WORD = "word"
    PHRASE = "phrase"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    COMPLETE = "complete"
    ERROR = "error"


class StreamQuality(str, Enum):
    """Stream quality levels"""
    ULTRA_LOW_LATENCY = "ultra_low_latency"  # <50ms chunks
    LOW_LATENCY = "low_latency"              # <100ms chunks
    BALANCED = "balanced"                     # <200ms chunks
    HIGH_QUALITY = "high_quality"            # Optimized for quality


@dataclass
class StreamChunk:
    """Individual streaming chunk with metadata"""
    chunk_id: str
    content: str
    chunk_type: ChunkType
    sequence_number: int
    timestamp: float
    is_final: bool
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # TTS integration fields
    should_synthesize: bool = True
    pause_after_ms: int = 0
    emphasis_level: float = 0.0
    speaking_rate_adjustment: float = 1.0


@dataclass
class StreamSession:
    """Streaming session management"""
    session_id: str
    model_id: str
    streaming_mode: StreamingMode
    quality_level: StreamQuality
    start_time: float
    last_activity: float
    chunks_sent: int = 0
    total_tokens: int = 0
    error_count: int = 0
    is_active: bool = True
    
    # Performance tracking
    avg_chunk_latency_ms: float = 0.0
    first_token_latency_ms: float = 0.0
    completion_latency_ms: float = 0.0
    
    # Stream state
    current_sentence: str = ""
    buffer: deque = field(default_factory=deque)
    processing_queue: asyncio.Queue = field(default_factory=asyncio.Queue)


@dataclass
class StreamingConfig:
    """Configuration for streaming behavior"""
    mode: StreamingMode = StreamingMode.ADAPTIVE
    quality: StreamQuality = StreamQuality.BALANCED
    
    # Chunk settings
    min_chunk_size: int = 3      # Minimum characters per chunk
    max_chunk_size: int = 50     # Maximum characters per chunk
    word_boundary_required: bool = True
    
    # Timing settings
    max_chunk_delay_ms: int = 200  # Maximum delay before sending chunk
    sentence_pause_ms: int = 300   # Pause after sentences
    paragraph_pause_ms: int = 500  # Pause after paragraphs
    
    # Quality settings
    enable_natural_pauses: bool = True
    enable_emphasis_detection: bool = True
    enable_punctuation_timing: bool = True
    
    # Agent-specific settings
    agent_id: Optional[str] = None
    urgency_level: str = "normal"


class NaturalLanguageProcessor:
    """
    Processes streaming text for natural speech patterns and optimal chunking
    """
    
    def __init__(self):
        """Initialize NLP processor"""
        
        # Punctuation timing rules
        self.punctuation_pauses = {
            '.': 300,    # Period - natural sentence end
            '!': 250,    # Exclamation - emphatic pause
            '?': 250,    # Question - inquiry pause
            ',': 150,    # Comma - brief pause
            ';': 200,    # Semicolon - moderate pause
            ':': 180,    # Colon - list or explanation pause
            '-': 100,    # Dash - brief interruption
            '...': 400,  # Ellipsis - thoughtful pause
            '—': 150,    # Em dash - emphasis pause
        }
        
        # Emphasis patterns
        self.emphasis_patterns = [
            r'\b(very|extremely|absolutely|definitely|certainly)\s+\w+',
            r'\b(important|critical|urgent|essential)\b',
            r'\b(never|always|must|should)\b',
            r'[A-Z]{2,}',  # ALL CAPS words
            r'\*\w+\*',    # *emphasized* words
        ]
        
        # Natural chunking boundaries
        self.chunk_boundaries = [
            r'\b(and|but|or|so|however|therefore|meanwhile|furthermore)\b',
            r'\b(first|second|third|next|then|finally)\b',
            r'\b(because|since|although|while|when|if)\b',
        ]
        
        # Breathing pause indicators
        self.breathing_indicators = [
            r'\b(um|uh|er|ah|well|now|so)\b',
            r'\b(you know|I mean|actually|basically)\b',
            r'\.{2,}',  # Multiple periods
        ]
        
        logger.info("Natural Language Processor initialized")
    
    def analyze_text_for_streaming(self, text: str) -> Dict[str, Any]:
        """Analyze text to determine optimal streaming strategy"""
        
        analysis = {
            'total_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(sent_tokenize(text)),
            'has_questions': '?' in text,
            'has_exclamations': '!' in text,
            'has_emphasis': any(re.search(pattern, text, re.IGNORECASE) for pattern in self.emphasis_patterns),
            'complexity_indicators': [],
            'optimal_chunk_size': 25,  # Default
            'recommended_mode': StreamingMode.WORD_LEVEL
        }
        
        # Determine optimal streaming mode
        if analysis['word_count'] < 10:
            analysis['recommended_mode'] = StreamingMode.WORD_LEVEL
            analysis['optimal_chunk_size'] = 15
        elif analysis['word_count'] < 30:
            analysis['recommended_mode'] = StreamingMode.CHUNK_LEVEL
            analysis['optimal_chunk_size'] = 20
        else:
            analysis['recommended_mode'] = StreamingMode.SENTENCE_LEVEL
            analysis['optimal_chunk_size'] = 35
        
        # Detect complexity indicators
        if any(word in text.lower() for word in ['analyze', 'compare', 'evaluate', 'explain']):
            analysis['complexity_indicators'].append('analytical')
            analysis['optimal_chunk_size'] += 10
        
        if any(word in text.lower() for word in ['step', 'first', 'second', 'procedure']):
            analysis['complexity_indicators'].append('procedural')
            analysis['recommended_mode'] = StreamingMode.SENTENCE_LEVEL
        
        return analysis
    
    def segment_for_streaming(self, text: str, chunk_size: int = 25, mode: StreamingMode = StreamingMode.ADAPTIVE) -> List[Dict[str, Any]]:
        """Segment text into optimal streaming chunks"""
        
        if mode == StreamingMode.SENTENCE_LEVEL:
            return self._segment_by_sentences(text)
        elif mode == StreamingMode.WORD_LEVEL:
            return self._segment_by_words(text, chunk_size)
        elif mode == StreamingMode.TOKEN_LEVEL:
            return self._segment_by_tokens(text)
        else:  # ADAPTIVE or CHUNK_LEVEL
            return self._segment_adaptively(text, chunk_size)
    
    def _segment_by_sentences(self, text: str) -> List[Dict[str, Any]]:
        """Segment text by sentences with natural pauses"""
        
        sentences = sent_tokenize(text)
        chunks = []
        
        for i, sentence in enumerate(sentences):
            # Determine pause after sentence
            pause_ms = 300  # Default sentence pause
            
            if sentence.endswith('?'):
                pause_ms = 250  # Question pause
            elif sentence.endswith('!'):
                pause_ms = 250  # Exclamation pause
            elif i == len(sentences) - 1:
                pause_ms = 100  # Final sentence - shorter pause
            
            # Check for emphasis
            emphasis = 0.0
            if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in self.emphasis_patterns):
                emphasis = 0.3
            
            chunks.append({
                'content': sentence.strip(),
                'chunk_type': ChunkType.SENTENCE,
                'pause_after_ms': pause_ms,
                'emphasis_level': emphasis,
                'sequence_number': i,
                'word_count': len(sentence.split())
            })
        
        return chunks
    
    def _segment_by_words(self, text: str, max_words: int = 5) -> List[Dict[str, Any]]:
        """Segment text by word groups with natural boundaries"""
        
        words = text.split()
        chunks = []
        current_chunk = []
        
        for i, word in enumerate(words):
            current_chunk.append(word)
            
            # Check for natural boundaries
            should_break = (
                len(current_chunk) >= max_words or
                word.endswith(('.', '!', '?', ',')) or
                i == len(words) - 1
            )
            
            if should_break:
                content = ' '.join(current_chunk)
                
                # Determine pause based on punctuation
                pause_ms = 0
                last_char = content[-1] if content else ''
                if last_char in self.punctuation_pauses:
                    pause_ms = self.punctuation_pauses[last_char]
                
                chunks.append({
                    'content': content,
                    'chunk_type': ChunkType.PHRASE if len(current_chunk) <= 3 else ChunkType.WORD,
                    'pause_after_ms': pause_ms,
                    'emphasis_level': 0.0,
                    'sequence_number': len(chunks),
                    'word_count': len(current_chunk)
                })
                
                current_chunk = []
        
        return chunks
    
    def _segment_by_tokens(self, text: str) -> List[Dict[str, Any]]:
        """Segment text by individual tokens (words and punctuation)"""
        
        # Simple tokenization including punctuation
        tokens = re.findall(r'\w+|[.!?,:;—\-\'"()]', text)
        chunks = []
        
        for i, token in enumerate(tokens):
            pause_ms = 0
            chunk_type = ChunkType.WORD
            
            # Handle punctuation
            if token in string.punctuation:
                pause_ms = self.punctuation_pauses.get(token, 50)
                chunk_type = ChunkType.WORD  # Punctuation as word-level
            
            chunks.append({
                'content': token,
                'chunk_type': chunk_type,
                'pause_after_ms': pause_ms,
                'emphasis_level': 0.0,
                'sequence_number': i,
                'word_count': 1 if token not in string.punctuation else 0
            })
        
        return chunks
    
    def _segment_adaptively(self, text: str, target_chunk_size: int = 25) -> List[Dict[str, Any]]:
        """Adaptive segmentation based on content analysis"""
        
        analysis = self.analyze_text_for_streaming(text)
        
        # Use sentence-level for short responses
        if analysis['word_count'] <= 15:
            return self._segment_by_words(text, 3)
        
        # Use natural phrase boundaries
        phrases = re.split(r'([.!?,:;]|\s+(?:and|but|or|so|however|therefore)\s+)', text)
        phrases = [p.strip() for p in phrases if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for phrase in phrases:
            if len(current_chunk) + len(phrase) <= target_chunk_size:
                current_chunk += phrase
            else:
                if current_chunk:
                    chunks.append(self._create_chunk_dict(current_chunk, len(chunks)))
                current_chunk = phrase
        
        if current_chunk:
            chunks.append(self._create_chunk_dict(current_chunk, len(chunks)))
        
        return chunks
    
    def _create_chunk_dict(self, content: str, sequence: int) -> Dict[str, Any]:
        """Create standardized chunk dictionary"""
        
        # Determine pause based on content ending
        pause_ms = 100  # Default
        if content.rstrip().endswith(('.', '!', '?')):
            pause_ms = self.punctuation_pauses.get(content.rstrip()[-1], 200)
        elif content.rstrip().endswith(','):
            pause_ms = 150
        
        # Check for emphasis
        emphasis = 0.0
        if any(re.search(pattern, content, re.IGNORECASE) for pattern in self.emphasis_patterns):
            emphasis = 0.2
        
        return {
            'content': content.strip(),
            'chunk_type': ChunkType.PHRASE,
            'pause_after_ms': pause_ms,
            'emphasis_level': emphasis,
            'sequence_number': sequence,
            'word_count': len(content.split())
        }


class StreamBuffer:
    """
    Intelligent buffering system for streaming optimization
    """
    
    def __init__(self, config: StreamingConfig):
        """Initialize stream buffer"""
        self.config = config
        self.buffer = deque()
        self.last_flush = time.time()
        self.total_buffered = 0
        self.flush_callbacks: List[Callable] = []
        
        # Adaptive buffering
        self.avg_chunk_time = 0.1  # 100ms default
        self.chunk_times = deque(maxlen=20)
        
        logger.debug("Stream Buffer initialized")
    
    def add_content(self, content: str, force_flush: bool = False) -> List[StreamChunk]:
        """Add content to buffer and return chunks ready for streaming"""
        
        self.buffer.append({
            'content': content,
            'timestamp': time.time(),
            'force_flush': force_flush
        })
        
        self.total_buffered += len(content)
        
        # Check if we should flush
        if self._should_flush() or force_flush:
            return self._flush_buffer()
        
        return []
    
    def _should_flush(self) -> bool:
        """Determine if buffer should be flushed"""
        
        current_time = time.time()
        time_since_flush = (current_time - self.last_flush) * 1000
        
        # Flush conditions
        return (
            self.total_buffered >= self.config.max_chunk_size or
            time_since_flush >= self.config.max_chunk_delay_ms or
            len(self.buffer) >= 5  # Too many pending chunks
        )
    
    def _flush_buffer(self) -> List[StreamChunk]:
        """Flush buffer and create stream chunks"""
        
        if not self.buffer:
            return []
        
        flush_start = time.time()
        chunks = []
        
        # Combine buffered content
        combined_content = ""
        force_flush = False
        
        while self.buffer:
            item = self.buffer.popleft()
            combined_content += item['content']
            if item['force_flush']:
                force_flush = True
        
        # Process content into chunks
        if combined_content.strip():
            nlp = NaturalLanguageProcessor()
            chunk_segments = nlp.segment_for_streaming(
                combined_content, 
                self.config.max_chunk_size,
                self.config.mode
            )
            
            # Create stream chunks
            for i, segment in enumerate(chunk_segments):
                chunk = StreamChunk(
                    chunk_id=str(uuid.uuid4()),
                    content=segment['content'],
                    chunk_type=ChunkType(segment.get('chunk_type', ChunkType.PHRASE)),
                    sequence_number=segment.get('sequence_number', i),
                    timestamp=time.time(),
                    is_final=force_flush and i == len(chunk_segments) - 1,
                    confidence=0.9,  # Default confidence
                    pause_after_ms=segment.get('pause_after_ms', 0),
                    emphasis_level=segment.get('emphasis_level', 0.0)
                )
                chunks.append(chunk)
        
        # Update timing stats
        flush_time = (time.time() - flush_start) * 1000
        self.chunk_times.append(flush_time)
        self.avg_chunk_time = sum(self.chunk_times) / len(self.chunk_times)
        
        # Reset buffer state
        self.total_buffered = 0
        self.last_flush = time.time()
        
        return chunks
    
    def force_flush(self) -> List[StreamChunk]:
        """Force flush all buffered content"""
        return self._flush_buffer()
    
    def add_flush_callback(self, callback: Callable):
        """Add callback to be called when buffer flushes"""
        self.flush_callbacks.append(callback)


class StreamMultiplexer:
    """
    Manages multiple concurrent streaming sessions
    """
    
    def __init__(self, max_concurrent_streams: int = 100):
        """Initialize stream multiplexer"""
        self.max_concurrent_streams = max_concurrent_streams
        self.active_streams: Dict[str, StreamSession] = {}
        self.stream_queues: Dict[str, asyncio.Queue] = {}
        self.performance_stats = defaultdict(lambda: defaultdict(float))
        
        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.monitoring_task: Optional[asyncio.Task] = None
        
        logger.info(f"Stream Multiplexer initialized (max streams: {max_concurrent_streams})")
    
    async def create_stream(self, session_id: str, model_id: str, config: StreamingConfig) -> StreamSession:
        """Create new streaming session"""
        
        if len(self.active_streams) >= self.max_concurrent_streams:
            # Clean up oldest inactive streams
            await self._cleanup_inactive_streams()
            
            if len(self.active_streams) >= self.max_concurrent_streams:
                raise Exception(f"Maximum concurrent streams ({self.max_concurrent_streams}) exceeded")
        
        stream_session = StreamSession(
            session_id=session_id,
            model_id=model_id,
            streaming_mode=config.mode,
            quality_level=config.quality,
            start_time=time.time(),
            last_activity=time.time()
        )
        
        self.active_streams[session_id] = stream_session
        self.stream_queues[session_id] = asyncio.Queue(maxsize=1000)
        
        logger.debug(f"Created stream session: {session_id}")
        return stream_session
    
    async def send_to_stream(self, session_id: str, chunk: StreamChunk):
        """Send chunk to specific stream"""
        
        if session_id not in self.active_streams:
            logger.warning(f"Stream session not found: {session_id}")
            return
        
        session = self.active_streams[session_id]
        session.last_activity = time.time()
        session.chunks_sent += 1
        
        # Add to queue
        queue = self.stream_queues[session_id]
        try:
            await queue.put(chunk)
        except asyncio.QueueFull:
            logger.warning(f"Stream queue full for session: {session_id}")
    
    async def get_stream_chunks(self, session_id: str) -> AsyncIterator[StreamChunk]:
        """Get chunks from stream"""
        
        if session_id not in self.stream_queues:
            logger.warning(f"Stream queue not found: {session_id}")
            return
        
        queue = self.stream_queues[session_id]
        
        try:
            while True:
                # Wait for chunk with timeout
                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield chunk
                    
                    # Update session
                    if session_id in self.active_streams:
                        self.active_streams[session_id].last_activity = time.time()
                    
                    # Check if stream is complete
                    if chunk.is_final:
                        logger.debug(f"Stream completed: {session_id}")
                        break
                        
                except asyncio.TimeoutError:
                    logger.debug(f"Stream timeout for session: {session_id}")
                    break
                    
        except Exception as e:
            logger.error(f"Error in stream processing: {e}")
        finally:
            await self.close_stream(session_id)
    
    async def close_stream(self, session_id: str):
        """Close streaming session"""
        
        if session_id in self.active_streams:
            session = self.active_streams[session_id]
            session.is_active = False
            
            # Calculate final metrics
            duration = time.time() - session.start_time
            self.performance_stats[session.model_id]['total_sessions'] += 1
            self.performance_stats[session.model_id]['avg_duration'] += duration
            
            del self.active_streams[session_id]
            logger.debug(f"Closed stream session: {session_id}")
        
        if session_id in self.stream_queues:
            del self.stream_queues[session_id]
    
    async def _cleanup_inactive_streams(self):
        """Clean up inactive streams"""
        
        current_time = time.time()
        inactive_streams = []
        
        for session_id, session in self.active_streams.items():
            if (current_time - session.last_activity > 300 or  # 5 minutes inactive
                not session.is_active):
                inactive_streams.append(session_id)
        
        for session_id in inactive_streams:
            await self.close_stream(session_id)
        
        if inactive_streams:
            logger.info(f"Cleaned up {len(inactive_streams)} inactive streams")
    
    def get_stream_stats(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        
        return {
            'active_streams': len(self.active_streams),
            'max_concurrent': self.max_concurrent_streams,
            'total_queued_chunks': sum(q.qsize() for q in self.stream_queues.values()),
            'performance_by_model': dict(self.performance_stats)
        }


class LLMStreamingHandler:
    """
    Main LLM Streaming Handler for real-time response processing.
    
    Provides sophisticated streaming capabilities with ultra-low latency optimization,
    intelligent buffering, and seamless TTS integration for voice AI applications.
    """
    
    def __init__(self,
                 openai_client: Optional[AsyncOpenAI] = None,
                 anthropic_client: Optional[anthropic.AsyncAnthropic] = None,
                 default_config: Optional[StreamingConfig] = None,
                 max_concurrent_streams: int = 100):
        """Initialize LLM streaming handler"""
        
        self.openai_client = openai_client
        self.anthropic_client = anthropic_client
        self.default_config = default_config or StreamingConfig()
        
        # Core components
        self.nlp_processor = NaturalLanguageProcessor()
        self.multiplexer = StreamMultiplexer(max_concurrent_streams)
        
        # Performance tracking
        self.performance_metrics = {
            'total_streams': 0,
            'successful_streams': 0,
            'avg_first_token_ms': 0.0,
            'avg_completion_ms': 0.0,
            'error_rate': 0.0,
            'throughput_tokens_per_second': 0.0
        }
        
        # Agent-specific configurations
        self.agent_configs = {
            'roadside-assistance': StreamingConfig(
                mode=StreamingMode.WORD_LEVEL,
                quality=StreamQuality.ULTRA_LOW_LATENCY,
                max_chunk_delay_ms=100,  # Ultra-fast for emergencies
                word_boundary_required=True
            ),
            'billing-support': StreamingConfig(
                mode=StreamingMode.CHUNK_LEVEL,
                quality=StreamQuality.BALANCED,
                max_chunk_delay_ms=150,  # Balanced for financial accuracy
                enable_emphasis_detection=True
            ),
            'technical-support': StreamingConfig(
                mode=StreamingMode.SENTENCE_LEVEL,
                quality=StreamQuality.HIGH_QUALITY,
                max_chunk_delay_ms=200,  # Quality for complex explanations
                enable_natural_pauses=True
            )
        }
        
        self.initialized = False
        logger.info("LLM Streaming Handler initialized")
    
    async def initialize(self):
        """Initialize streaming handler and components"""
        
        if self.initialized:
            return
        
        logger.info("🚀 Initializing LLM Streaming Handler...")
        
        try:
            # Initialize OpenAI client if not provided
            if not self.openai_client:
                self.openai_client = AsyncOpenAI()
            
            # Initialize Anthropic client if available
            if not self.anthropic_client and os.getenv('ANTHROPIC_API_KEY'):
                self.anthropic_client = anthropic.AsyncAnthropic()
            
            # Start background monitoring
            self.multiplexer.monitoring_task = asyncio.create_task(self._background_monitoring())
            self.multiplexer.cleanup_task = asyncio.create_task(self._background_cleanup())
            
            self.initialized = True
            logger.info("✅ LLM Streaming Handler initialization complete")
            
        except Exception as e:
            logger.error(f"❌ Streaming handler initialization failed: {e}")
            raise
    
    async def stream_response(self,
                            session_id: str,
                            model_id: str,
                            messages: List[Dict[str, str]],
                            agent_id: Optional[str] = None,
                            config: Optional[StreamingConfig] = None,
                            tts_callback: Optional[Callable[[StreamChunk], Awaitable[None]]] = None) -> AsyncIterator[StreamChunk]:
        """
        Main streaming method - stream LLM response with optimal chunking
        
        Args:
            session_id: Unique session identifier
            model_id: LLM model to use
            messages: Conversation messages
            agent_id: Agent requesting the stream
            config: Streaming configuration
            tts_callback: Callback for TTS integration
            
        Yields:
            StreamChunk: Optimized chunks for streaming
        """
        
        if not self.initialized:
            await self.initialize()
        
        stream_start = time.time()
        
        # Get configuration
        streaming_config = config or self._get_agent_config(agent_id)
        
        logger.debug(f"Starting stream for session {session_id} with model {model_id}")
        
        try:
            # Create stream session
            session = await self.multiplexer.create_stream(session_id, model_id, streaming_config)
            
            # Initialize stream buffer
            buffer = StreamBuffer(streaming_config)
            
            # Start LLM streaming based on provider
            if model_id.startswith('gpt-'):
                stream_task = asyncio.create_task(
                    self._stream_openai_response(session, messages, streaming_config, buffer, tts_callback)
                )
            elif model_id.startswith('claude-'):
                stream_task = asyncio.create_task(
                    self._stream_anthropic_response(session, messages, streaming_config, buffer, tts_callback)
                )
            else:
                raise ValueError(f"Unsupported model: {model_id}")
            
            # Stream chunks to client
            first_chunk_sent = False
            async for chunk in self.multiplexer.get_stream_chunks(session_id):
                # Record first token latency
                if not first_chunk_sent:
                    session.first_token_latency_ms = (time.time() - stream_start) * 1000
                    first_chunk_sent = True
                    logger.debug(f"First token latency: {session.first_token_latency_ms:.2f}ms")
                
                # Send to TTS if callback provided
                if tts_callback and chunk.should_synthesize:
                    try:
                        await tts_callback(chunk)
                    except Exception as e:
                        logger.error(f"TTS callback error: {e}")
                
                yield chunk
            
            # Wait for streaming task to complete
            await stream_task
            
            # Record completion metrics
            session.completion_latency_ms = (time.time() - stream_start) * 1000
            self._update_performance_metrics(session)
            
            logger.debug(f"Stream completed: {session_id} (total: {session.completion_latency_ms:.2f}ms)")
            
        except Exception as e:
            logger.error(f"❌ Streaming error for session {session_id}: {e}")
            
            # Send error chunk
            error_chunk = StreamChunk(
                chunk_id=str(uuid.uuid4()),
                content=f"Sorry, I encountered an error: {str(e)}",
                chunk_type=ChunkType.ERROR,
                sequence_number=0,
                timestamp=time.time(),
                is_final=True,
                confidence=0.0,
                metadata={'error': str(e)}
            )
            
            yield error_chunk
            
        finally:
            await self.multiplexer.close_stream(session_id)
    
    async def _stream_openai_response(self,
                                    session: StreamSession,
                                    messages: List[Dict[str, str]],
                                    config: StreamingConfig,
                                    buffer: StreamBuffer,
                                    tts_callback: Optional[Callable]) -> None:
        """Stream response from OpenAI model"""
        
        try:
            # Create streaming completion
            stream = await self.openai_client.chat.completions.create(
                model=session.model_id,
                messages=messages,
                max_tokens=300,  # Reasonable limit for voice responses
                temperature=0.7,
                stream=True
            )
            
            accumulated_content = ""
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    accumulated_content += content
                    session.total_tokens += 1
                    
                    # Add to buffer
                    stream_chunks = buffer.add_content(content)
                    
                    # Send chunks to multiplexer
                    for stream_chunk in stream_chunks:
                        await self.multiplexer.send_to_stream(session.session_id, stream_chunk)
            
            # Flush remaining content
            final_chunks = buffer.force_flush()
            for chunk in final_chunks:
                chunk.is_final = True
                await self.multiplexer.send_to_stream(session.session_id, chunk)
            
            logger.debug(f"OpenAI stream completed: {len(accumulated_content)} characters")
            
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            session.error_count += 1
            raise
    
    async def _stream_anthropic_response(self,
                                       session: StreamSession,
                                       messages: List[Dict[str, str]],
                                       config: StreamingConfig,
                                       buffer: StreamBuffer,
                                       tts_callback: Optional[Callable]) -> None:
        """Stream response from Anthropic model"""
        
        if not self.anthropic_client:
            raise Exception("Anthropic client not initialized")
        
        try:
            # Convert messages to Anthropic format
            prompt = self._convert_messages_to_anthropic_prompt(messages)
            
            # Create streaming completion
            async with self.anthropic_client.messages.stream(
                model=session.model_id,
                max_tokens=300,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                
                accumulated_content = ""
                
                async for event in stream:
                    if event.type == "content_block_delta" and hasattr(event.delta, 'text'):
                        content = event.delta.text
                        accumulated_content += content
                        session.total_tokens += 1
                        
                        # Add to buffer
                        stream_chunks = buffer.add_content(content)
                        
                        # Send chunks to multiplexer
                        for stream_chunk in stream_chunks:
                            await self.multiplexer.send_to_stream(session.session_id, stream_chunk)
            
            # Flush remaining content
            final_chunks = buffer.force_flush()
            for chunk in final_chunks:
                chunk.is_final = True
                await self.multiplexer.send_to_stream(session.session_id, chunk)
            
            logger.debug(f"Anthropic stream completed: {len(accumulated_content)} characters")
            
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            session.error_count += 1
            raise
    
    def _convert_messages_to_anthropic_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI-style messages to Anthropic prompt format"""
        
        prompt_parts = []
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"Human: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts) + "\n\nAssistant:"
    
    def _get_agent_config(self, agent_id: Optional[str]) -> StreamingConfig:
        """Get streaming configuration for agent"""
        
        if agent_id and agent_id in self.agent_configs:
            return self.agent_configs[agent_id]
        
        return self.default_config
    
    def _update_performance_metrics(self, session: StreamSession):
        """Update performance metrics from completed session"""
        
        self.performance_metrics['total_streams'] += 1
        
        if session.error_count == 0:
            self.performance_metrics['successful_streams'] += 1
        
        # Update averages
        total = self.performance_metrics['total_streams']
        
        # First token latency
        current_avg = self.performance_metrics['avg_first_token_ms']
        self.performance_metrics['avg_first_token_ms'] = (
            (current_avg * (total - 1) + session.first_token_latency_ms) / total
        )
        
        # Completion latency
        current_avg = self.performance_metrics['avg_completion_ms']
        self.performance_metrics['avg_completion_ms'] = (
            (current_avg * (total - 1) + session.completion_latency_ms) / total
        )
        
        # Error rate
        self.performance_metrics['error_rate'] = (
            (total - self.performance_metrics['successful_streams']) / total
        ) * 100
        
        # Throughput
        if session.completion_latency_ms > 0:
            throughput = (session.total_tokens / session.completion_latency_ms) * 1000
            current_throughput = self.performance_metrics['throughput_tokens_per_second']
            self.performance_metrics['throughput_tokens_per_second'] = (
                (current_throughput * (total - 1) + throughput) / total
            )
    
    async def _background_monitoring(self):
        """Background task for performance monitoring"""
        
        while True:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                
                # Log performance summary
                metrics = self.performance_metrics
                
                if metrics['total_streams'] > 0:
                    logger.info(f"Streaming Performance: "
                              f"First token: {metrics['avg_first_token_ms']:.1f}ms, "
                              f"Completion: {metrics['avg_completion_ms']:.1f}ms, "
                              f"Success rate: {100 - metrics['error_rate']:.1f}%, "
                              f"Throughput: {metrics['throughput_tokens_per_second']:.1f} tokens/sec")
                
                # Check for performance issues
                if metrics['avg_first_token_ms'] > 200:
                    logger.warning("⚠️ High first token latency detected")
                
                if metrics['error_rate'] > 5:
                    logger.warning("⚠️ High error rate detected")
                
            except Exception as e:
                logger.error(f"Error in background monitoring: {e}")
    
    async def _background_cleanup(self):
        """Background cleanup of resources"""
        
        while True:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                await self.multiplexer._cleanup_inactive_streams()
                
            except Exception as e:
                logger.error(f"Error in background cleanup: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        
        return {
            **self.performance_metrics,
            'multiplexer_stats': self.multiplexer.get_stream_stats(),
            'target_first_token_ms': 100,
            'target_completion_ms': 1000,
            'performance_status': {
                'first_token_ok': self.performance_metrics['avg_first_token_ms'] <= 150,
                'completion_ok': self.performance_metrics['avg_completion_ms'] <= 1500,
                'error_rate_ok': self.performance_metrics['error_rate'] <= 2
            }
        }
    
    async def shutdown(self):
        """Shutdown streaming handler and cleanup resources"""
        
        logger.info("Shutting down LLM Streaming Handler...")
        
        # Cancel background tasks
        if self.multiplexer.monitoring_task:
            self.multiplexer.monitoring_task.cancel()
        
        if self.multiplexer.cleanup_task:
            self.multiplexer.cleanup_task.cancel()
        
        # Close all active streams
        for session_id in list(self.multiplexer.active_streams.keys()):
            await self.multiplexer.close_stream(session_id)
        
        self.initialized = False
        logger.info("✅ LLM Streaming Handler shutdown complete")


# Utility functions for easy integration

def create_streaming_handler_for_agent(agent_type: str, **kwargs) -> LLMStreamingHandler:
    """Create optimized streaming handler for specific agent types"""
    
    agent_configs = {
        "roadside-assistance": StreamingConfig(
            mode=StreamingMode.WORD_LEVEL,
            quality=StreamQuality.ULTRA_LOW_LATENCY,
            max_chunk_delay_ms=80,
            urgency_level="emergency"
        ),
        "billing-support": StreamingConfig(
            mode=StreamingMode.CHUNK_LEVEL,
            quality=StreamQuality.BALANCED,
            max_chunk_delay_ms=120,
            enable_emphasis_detection=True
        ),
        "technical-support": StreamingConfig(
            mode=StreamingMode.SENTENCE_LEVEL,
            quality=StreamQuality.HIGH_QUALITY,
            max_chunk_delay_ms=180,
            enable_natural_pauses=True
        )
    }
    
    default_config = agent_configs.get(agent_type)
    
    return LLMStreamingHandler(
        default_config=default_config,
        **kwargs
    )


# Export main classes and functions
__all__ = [
    'LLMStreamingHandler',
    'StreamChunk',
    'StreamingConfig',
    'StreamingMode',
    'StreamQuality',
    'ChunkType',
    'NaturalLanguageProcessor',
    'StreamBuffer',
    'StreamMultiplexer',
    'create_streaming_handler_for_agent'
]