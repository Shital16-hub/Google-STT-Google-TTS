# utils/logging.py

"""
Enhanced logging configuration with structured logging and error tracking.
"""
import logging
import json
import sys
import time
from typing import Dict, Any, Optional
from pathlib import Path
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger
import traceback

# Custom JSON formatter for structured logging
class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields."""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]):
        """Add custom fields to log record."""
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp
        log_record['timestamp'] = time.time()
        log_record['datetime'] = self.formatTime(record)
        
        # Add log level
        log_record['level'] = record.levelname
        log_record['name'] = record.name
        
        # Add caller information
        log_record['filename'] = record.filename
        log_record['funcName'] = record.funcName
        log_record['lineno'] = record.lineno
        
        # Add thread information
        log_record['threadName'] = record.threadName
        log_record['thread'] = record.thread
        
        # Add exception info if present
        if record.exc_info:
            log_record['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'stacktrace': traceback.format_exception(*record.exc_info)
            }

def setup_logging(
    log_dir: str = "./logs",
    log_level: str = "INFO",
    enable_json: bool = True,
    enable_console: bool = True,
    enable_file: bool = True,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5
):
    """
    Set up enhanced logging configuration.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
        enable_json: Whether to use JSON formatting
        enable_console: Whether to log to console
        enable_file: Whether to log to file
        max_bytes: Maximum log file size
        backup_count: Number of backup files to keep
    """
    # Create log directory
    if enable_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    if enable_json:
        json_formatter = CustomJsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s'
        )
    
    standard_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            json_formatter if enable_json else standard_formatter
        )
        root_logger.addHandler(console_handler)
    
    # File handlers
    if enable_file:
        # Main log file
        main_handler = RotatingFileHandler(
            log_path / "voice_ai.log",
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        main_handler.setFormatter(
            json_formatter if enable_json else standard_formatter
        )
        root_logger.addHandler(main_handler)
        
        # Error log file
        error_handler = RotatingFileHandler(
            log_path / "error.log",
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(
            json_formatter if enable_json else standard_formatter
        )
        root_logger.addHandler(error_handler)
        
        # Access log file
        access_handler = RotatingFileHandler(
            log_path / "access.log",
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        access_handler.setFormatter(
            json_formatter if enable_json else standard_formatter
        )
        root_logger.addHandler(access_handler)

class StructuredLogger:
    """Enhanced logger with structured logging capabilities."""
    
    def __init__(
        self,
        name: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            context: Optional context dictionary
        """
        self.logger = logging.getLogger(name)
        self.context = context or {}
    
    def _format_message(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format message with context."""
        log_data = {
            "message": message,
            "context": self.context.copy()
        }
        
        if extra:
            log_data["context"].update(extra)
            
        return log_data
    
    def info(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None
    ):
        """Log info message."""
        self.logger.info(
            json.dumps(self._format_message(message, extra)),
            extra=extra
        )
    
    def error(
        self,
        message: str,
        exc_info: bool = True,
        extra: Optional[Dict[str, Any]] = None
    ):
        """Log error message."""
        self.logger.error(
            json.dumps(self._format_message(message, extra)),
            exc_info=exc_info,
            extra=extra
        )
    
    def warning(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None
    ):
        """Log warning message."""
        self.logger.warning(
            json.dumps(self._format_message(message, extra)),
            extra=extra
        )
    
    def debug(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None
    ):
        """Log debug message."""
        self.logger.debug(
            json.dumps(self._format_message(message, extra)),
            extra=extra
        )
    
    def set_context(self, context: Dict[str, Any]):
        """Update logger context."""
        self.context.update(context)
    
    def clear_context(self):
        """Clear logger context."""
        self.context.clear()

class RequestLogger(StructuredLogger):
    """Logger specialized for API requests."""
    
    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration: float,
        extra: Optional[Dict[str, Any]] = None
    ):
        """Log API request."""
        context = {
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration": duration
        }
        
        if extra:
            context.update(extra)
            
        self.info(
            f"{method} {path} {status_code}",
            extra=context
        )

class ServiceLogger(StructuredLogger):
    """Logger specialized for service operations."""
    
    def log_service_start(
        self,
        service_type: str,
        session_id: str,
        extra: Optional[Dict[str, Any]] = None
    ):
        """Log service start."""
        context = {
            "service_type": service_type,
            "session_id": session_id,
            "event": "service_start",
            "timestamp": time.time()
        }
        
        if extra:
            context.update(extra)
            
        self.info(
            f"Starting {service_type} service for session {session_id}",
            extra=context
        )
    
    def log_service_end(
        self,
        service_type: str,
        session_id: str,
        duration: float,
        success: bool,
        extra: Optional[Dict[str, Any]] = None
    ):
        """Log service completion."""
        context = {
            "service_type": service_type,
            "session_id": session_id,
            "event": "service_end",
            "duration": duration,
            "success": success,
            "timestamp": time.time()
        }
        
        if extra:
            context.update(extra)
            
        self.info(
            f"Completed {service_type} service for session {session_id} "
            f"(success: {success}, duration: {duration:.2f}s)",
            extra=context
        )
    
    def log_service_error(
        self,
        service_type: str,
        session_id: str,
        error: Exception,
        extra: Optional[Dict[str, Any]] = None
    ):
        """Log service error."""
        context = {
            "service_type": service_type,
            "session_id": session_id,
            "event": "service_error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": time.time()
        }
        
        if extra:
            context.update(extra)
            
        self.error(
            f"Error in {service_type} service for session {session_id}: {error}",
            exc_info=True,
            extra=context
        )
    
    def log_handoff(
        self,
        service_type: str,
        session_id: str,
        reason: str,
        extra: Optional[Dict[str, Any]] = None
    ):
        """Log service handoff."""
        context = {
            "service_type": service_type,
            "session_id": session_id,
            "event": "handoff",
            "reason": reason,
            "timestamp": time.time()
        }
        
        if extra:
            context.update(extra)
            
        self.info(
            f"Handoff in {service_type} service for session {session_id}: {reason}",
            extra=context
        )

class ConversationLogger(StructuredLogger):
    """Logger specialized for conversation tracking."""
    
    def log_user_message(
        self,
        session_id: str,
        message: str,
        extra: Optional[Dict[str, Any]] = None
    ):
        """Log user message."""
        context = {
            "session_id": session_id,
            "event": "user_message",
            "message_length": len(message),
            "timestamp": time.time()
        }
        
        if extra:
            context.update(extra)
            
        self.info(
            f"User message in session {session_id}: {message[:100]}...",
            extra=context
        )
    
    def log_assistant_response(
        self,
        session_id: str,
        response: str,
        response_time: float,
        extra: Optional[Dict[str, Any]] = None
    ):
        """Log assistant response."""
        context = {
            "session_id": session_id,
            "event": "assistant_response",
            "response_length": len(response),
            "response_time": response_time,
            "timestamp": time.time()
        }
        
        if extra:
            context.update(extra)
            
        self.info(
            f"Assistant response in session {session_id} "
            f"(time: {response_time:.2f}s): {response[:100]}...",
            extra=context
        )
    
    def log_state_change(
        self,
        session_id: str,
        old_state: str,
        new_state: str,
        extra: Optional[Dict[str, Any]] = None
    ):
        """Log conversation state change."""
        context = {
            "session_id": session_id,
            "event": "state_change",
            "old_state": old_state,
            "new_state": new_state,
            "timestamp": time.time()
        }
        
        if extra:
            context.update(extra)
            
        self.info(
            f"State change in session {session_id}: {old_state} -> {new_state}",
            extra=context
        )
    
    def log_intent_detection(
        self,
        session_id: str,
        intent: str,
        confidence: float,
        extra: Optional[Dict[str, Any]] = None
    ):
        """Log intent detection."""
        context = {
            "session_id": session_id,
            "event": "intent_detection",
            "intent": intent,
            "confidence": confidence,
            "timestamp": time.time()
        }
        
        if extra:
            context.update(extra)
            
        self.info(
            f"Detected intent in session {session_id}: "
            f"{intent} (confidence: {confidence:.2f})",
            extra=context
        )

# Initialize loggers
request_logger = RequestLogger("request")
service_logger = ServiceLogger("service")
conversation_logger = ConversationLogger("conversation")

# Set up logging on import
setup_logging()