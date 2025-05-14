# telephony/websocket/connection_manager.py

"""
WebSocket connection management for Twilio streams.
Handles connection lifecycle, keep-alive, and stream state management.
"""
import asyncio
import logging
import json
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConnectionManager:
    """
    Manages WebSocket connection state and lifecycle for Twilio media streams.
    """
    
    def __init__(self, call_sid: str):
        """
        Initialize connection manager.
        
        Args:
            call_sid: Twilio call SID for this connection
        """
        self.call_sid = call_sid
        self.stream_sid = None
        self.connected = False
        self.connection_active = asyncio.Event()
        self.connection_active.clear()
        
        # Keep-alive management
        self.keep_alive_task = None
        self.keep_alive_interval = 10  # seconds
        self.last_ping_time = 0
        self.ping_count = 0
        self.pong_count = 0
        
        # Connection timing
        self.connect_time = None
        self.disconnect_time = None
        
        # Connection statistics
        self.total_bytes_sent = 0
        self.total_bytes_received = 0
        self.messages_sent = 0
        self.messages_received = 0
        
        logger.info(f"ConnectionManager initialized for call {call_sid}")
    
    async def handle_connected(self, data: Dict[str, Any], ws) -> None:
        """
        Handle WebSocket connected event.
        
        Args:
            data: Connected event data from Twilio
            ws: WebSocket connection object
        """
        logger.info(f"WebSocket connected for call {self.call_sid}")
        logger.info(f"Connection data: {data}")
        
        # Set connection state
        self.connected = True
        self.connect_time = time.time()
        self.connection_active.set()
        
        # Extract connection info
        protocol = data.get('protocol', 'unknown')
        version = data.get('version', 'unknown')
        logger.info(f"Connected - Protocol: {protocol}, Version: {version}")
        
        # Start keep-alive task
        await self._start_keep_alive(ws)
    
    async def handle_start(self, data: Dict[str, Any], ws) -> None:
        """
        Handle stream start event.
        
        Args:
            data: Start event data from Twilio
            ws: WebSocket connection object
        """
        self.stream_sid = data.get('streamSid')
        logger.info(f"Stream started - SID: {self.stream_sid}, Call: {self.call_sid}")
        
        # Extract and log stream configuration
        start_data = data.get('start', {})
        media_format = start_data.get('mediaFormat', {})
        custom_params = start_data.get('customParameters', {})
        
        logger.info(f"Media format: {media_format}")
        logger.info(f"Custom parameters: {custom_params}")
        
        # Update connection state
        self.connection_active.set()
        
        # Send acknowledgment
        await self._send_ack(ws)
    
    async def handle_stop(self, data: Dict[str, Any]) -> None:
        """
        Handle stream stop event.
        
        Args:
            data: Stop event data from Twilio
        """
        logger.info(f"Stream stopped - SID: {self.stream_sid}")
        logger.info(f"Stop data: {data}")
        
        # Update connection state
        self.connected = False
        self.disconnect_time = time.time()
        self.connection_active.clear()
        
        # Stop keep-alive task
        await self._stop_keep_alive()
        
        # Log connection duration
        if self.connect_time:
            duration = self.disconnect_time - self.connect_time
            logger.info(f"Connection duration: {duration:.2f} seconds")
        
        # Log final statistics
        self._log_connection_stats()
    
    async def _start_keep_alive(self, ws) -> None:
        """Start the keep-alive task."""
        if self.keep_alive_task and not self.keep_alive_task.done():
            logger.debug("Keep-alive task already running")
            return
        
        self.keep_alive_task = asyncio.create_task(self._keep_alive_loop(ws))
        logger.info("Started keep-alive task")
    
    async def _stop_keep_alive(self) -> None:
        """Stop the keep-alive task."""
        if self.keep_alive_task:
            self.keep_alive_task.cancel()
            try:
                await self.keep_alive_task
            except asyncio.CancelledError:
                logger.debug("Keep-alive task cancelled")
            self.keep_alive_task = None
        logger.info("Stopped keep-alive task")
    
    async def _keep_alive_loop(self, ws) -> None:
        """
        Send periodic keep-alive messages to maintain the connection.
        
        Args:
            ws: WebSocket connection object
        """
        try:
            while self.connected and self.stream_sid:
                await asyncio.sleep(self.keep_alive_interval)
                
                # Check if still connected
                if not self.connected or not self.stream_sid:
                    break
                
                try:
                    # Send ping message
                    ping_message = {
                        "event": "ping",
                        "streamSid": self.stream_sid,
                        "timestamp": time.time()
                    }
                    
                    await self._send_message(ws, ping_message)
                    self.last_ping_time = time.time()
                    self.ping_count += 1
                    
                    logger.debug(f"Sent keep-alive ping #{self.ping_count}")
                    
                except Exception as e:
                    logger.error(f"Error sending keep-alive ping: {e}")
                    
                    # Check if it's a connection error
                    if "Connection closed" in str(e) or "WebSocket" in str(e):
                        logger.warning("WebSocket connection lost during keep-alive")
                        self.connected = False
                        self.connection_active.clear()
                        break
                    
        except asyncio.CancelledError:
            logger.debug("Keep-alive loop cancelled")
        except Exception as e:
            logger.error(f"Unexpected error in keep-alive loop: {e}")
        finally:
            logger.info("Keep-alive loop ended")
    
    async def _send_ack(self, ws) -> None:
        """Send acknowledgment message for stream start."""
        try:
            ack_message = {
                "event": "ack",
                "streamSid": self.stream_sid,
                "timestamp": time.time()
            }
            await self._send_message(ws, ack_message)
            logger.debug("Sent stream start acknowledgment")
        except Exception as e:
            logger.error(f"Error sending acknowledgment: {e}")
    
    async def _send_message(self, ws, message: Dict[str, Any]) -> None:
        """
        Send a message through the WebSocket with error handling.
        
        Args:
            ws: WebSocket connection object
            message: Message to send
        """
        try:
            message_str = json.dumps(message)
            ws.send(message_str)
            
            # Update statistics
            self.messages_sent += 1
            self.total_bytes_sent += len(message_str)
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise
    
    def handle_pong(self, data: Dict[str, Any]) -> None:
        """
        Handle pong response from Twilio.
        
        Args:
            data: Pong event data
        """
        self.pong_count += 1
        logger.debug(f"Received pong #{self.pong_count}")
        
        # Calculate ping latency if timestamp is available
        timestamp = data.get('timestamp')
        if timestamp and self.last_ping_time:
            latency = time.time() - timestamp
            logger.debug(f"Ping latency: {latency*1000:.2f}ms")
    
    def update_received_stats(self, message_size: int) -> None:
        """
        Update statistics for received messages.
        
        Args:
            message_size: Size of the received message in bytes
        """
        self.messages_received += 1
        self.total_bytes_received += message_size
    
    def is_connected(self) -> bool:
        """Check if the connection is active."""
        return self.connected and self.stream_sid is not None
    
    def wait_for_connection(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the connection to be established.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if connected within timeout, False otherwise
        """
        try:
            # Use asyncio.wait_for with the event
            asyncio.wait_for(self.connection_active.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Connection timeout after {timeout} seconds")
            return False
        except Exception as e:
            logger.error(f"Error waiting for connection: {e}")
            return False
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        current_time = time.time()
        
        stats = {
            "call_sid": self.call_sid,
            "stream_sid": self.stream_sid,
            "connected": self.connected,
            "connect_time": self.connect_time,
            "disconnect_time": self.disconnect_time,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "total_bytes_sent": self.total_bytes_sent,
            "total_bytes_received": self.total_bytes_received,
            "ping_count": self.ping_count,
            "pong_count": self.pong_count,
            "last_ping_time": self.last_ping_time
        }
        
        # Add duration if connected
        if self.connect_time:
            if self.disconnect_time:
                stats["connection_duration"] = self.disconnect_time - self.connect_time
            else:
                stats["connection_duration"] = current_time - self.connect_time
        
        # Add ping success rate
        if self.ping_count > 0:
            stats["ping_success_rate"] = (self.pong_count / self.ping_count) * 100
        else:
            stats["ping_success_rate"] = 0
        
        return stats
    
    def _log_connection_stats(self) -> None:
        """Log comprehensive connection statistics."""
        stats = self.get_connection_stats()
        
        logger.info("=== Connection Statistics ===")
        logger.info(f"Call SID: {stats['call_sid']}")
        logger.info(f"Stream SID: {stats['stream_sid']}")
        logger.info(f"Duration: {stats.get('connection_duration', 0):.2f} seconds")
        logger.info(f"Messages sent: {stats['messages_sent']}")
        logger.info(f"Messages received: {stats['messages_received']}")
        logger.info(f"Bytes sent: {stats['total_bytes_sent']}")
        logger.info(f"Bytes received: {stats['total_bytes_received']}")
        logger.info(f"Ping/Pong: {stats['ping_count']}/{stats['pong_count']}")
        logger.info(f"Ping success rate: {stats['ping_success_rate']:.1f}%")
        logger.info("=============================")
    
    async def close(self) -> None:
        """Clean up connection manager resources."""
        logger.info(f"Closing connection manager for call {self.call_sid}")
        
        # Stop keep-alive
        await self._stop_keep_alive()
        
        # Update state
        self.connected = False
        self.connection_active.clear()
        
        # Log final stats
        self._log_connection_stats()