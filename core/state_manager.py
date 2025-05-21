# core/state_manager.py

"""
Enhanced conversation state management with slot tracking.
"""
from enum import Enum
from typing import Dict, Any, List, Optional, Set
import logging
import time

logger = logging.getLogger(__name__)

class ConversationState(str, Enum):
    """Enhanced conversation states for roadside assistance."""
    GREETING = "greeting"
    COLLECTING_NAME = "collecting_name"
    COLLECTING_PHONE = "collecting_phone"
    COLLECTING_LOCATION = "collecting_location"
    COLLECTING_VEHICLE = "collecting_vehicle"
    CONFIRMING_SERVICE = "confirming_service"
    PROVIDING_PRICE = "providing_price"
    HANDOFF = "handoff"
    ENDED = "ended"

class StateManager:
    """Manages conversation state and slot filling."""
    
    def __init__(self):
        """Initialize the state manager."""
        self.current_state = ConversationState.GREETING
        self.slots: Dict[str, Any] = {}
        self.confirmed_slots: Set[str] = set()
        
        # Track state transitions
        self.state_history: List[Dict[str, Any]] = []
        self.last_state_change = time.time()
        
        # Track slot updates
        self.slot_updates: List[Dict[str, Any]] = []
        
        logger.info("Initialized state manager")
    
    def update_state(self, new_state: ConversationState) -> bool:
        """
        Update conversation state with tracking.
        
        Args:
            new_state: New state to transition to
            
        Returns:
            True if state was changed
        """
        if new_state != self.current_state:
            # Record state transition
            self.state_history.append({
                "from_state": self.current_state,
                "to_state": new_state,
                "timestamp": time.time(),
                "duration": time.time() - self.last_state_change
            })
            
            # Update state
            self.current_state = new_state
            self.last_state_change = time.time()
            
            logger.info(f"State transition: {self.current_state}")
            return True
        return False
    
    def set_slot_value(
        self,
        slot: str,
        value: Any,
        confidence: float = 1.0,
        source: str = "direct"
    ) -> bool:
        """
        Set a slot value with metadata.
        
        Args:
            slot: Slot name
            value: Slot value
            confidence: Confidence in the value
            source: Source of the value
            
        Returns:
            True if slot was updated
        """
        # Record update attempt
        update = {
            "slot": slot,
            "old_value": self.slots.get(slot),
            "new_value": value,
            "confidence": confidence,
            "source": source,
            "timestamp": time.time()
        }
        
        # Check if value is different
        if slot not in self.slots or self.slots[slot] != value:
            self.slots[slot] = value
            update["changed"] = True
            logger.info(f"Updated slot {slot}: {value}")
        else:
            update["changed"] = False
        
        # Record update
        self.slot_updates.append(update)
        
        return update["changed"]
    
    def confirm_slot(self, slot: str):
        """
        Mark a slot as confirmed.
        
        Args:
            slot: Slot name to confirm
        """
        self.confirmed_slots.add(slot)
        logger.info(f"Confirmed slot: {slot}")
    
    def get_missing_slots(self, required_slots: List[str]) -> List[str]:
        """
        Get list of missing required slots.
        
        Args:
            required_slots: List of required slot names
            
        Returns:
            List of missing slot names
        """
        return [
            slot for slot in required_slots
            if slot not in self.slots or not self.slots[slot]
        ]
    
    def get_unconfirmed_slots(self, required_slots: List[str]) -> List[str]:
        """
        Get list of unconfirmed required slots.
        
        Args:
            required_slots: List of required slot names
            
        Returns:
            List of unconfirmed slot names
        """
        return [
            slot for slot in required_slots
            if slot not in self.confirmed_slots
        ]
    
    def clear_state(self):
        """Reset the state manager."""
        self.current_state = ConversationState.GREETING
        self.slots.clear()
        self.confirmed_slots.clear()
        logger.info("Reset state manager")
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get comprehensive state information."""
        return {
            "current_state": self.current_state,
            "slots": self.slots,
            "confirmed_slots": list(self.confirmed_slots),
            "last_state_change": self.last_state_change,
            "state_changes": len(self.state_history),
            "slot_updates": len(self.slot_updates)
        }
    
    def get_slot_history(self, slot: str) -> List[Dict[str, Any]]:
        """
        Get history of updates for a specific slot.
        
        Args:
            slot: Slot name
            
        Returns:
            List of slot updates
        """
        return [
            update for update in self.slot_updates
            if update["slot"] == slot
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get state manager statistics."""
        total_duration = time.time() - self.state_history[0]["timestamp"] if self.state_history else 0
        
        stats = {
            "total_duration": total_duration,
            "state_changes": len(self.state_history),
            "slot_updates": len(self.slot_updates),
            "filled_slots": len(self.slots),
            "confirmed_slots": len(self.confirmed_slots)
        }
        
        # Add state durations
        if self.state_history:
            state_durations = {}
            for transition in self.state_history:
                state = transition["from_state"]
                duration = transition["duration"]
                if state not in state_durations:
                    state_durations[state] = []
                state_durations[state].append(duration)
            
            # Calculate average durations
            stats["avg_state_durations"] = {
                state: sum(durations) / len(durations)
                for state, durations in state_durations.items()
            }
        
        return stats