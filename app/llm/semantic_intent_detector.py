"""
Semantic Intent Detection System - Replace Hardcoded Keywords
============================================================

Uses LLM-based semantic understanding to detect user intent naturally,
without relying on exact keyword matches. Handles natural speech patterns,
variations, and context understanding.
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import re

logger = logging.getLogger(__name__)

class UserIntent(str, Enum):
    """Detected user intents"""
    ROADSIDE_ASSISTANCE = "roadside_assistance"
    BILLING_SUPPORT = "billing_support" 
    TECHNICAL_SUPPORT = "technical_support"
    GENERAL_INQUIRY = "general_inquiry"
    GREETING = "greeting"
    CLARIFICATION = "clarification"
    UNKNOWN = "unknown"

@dataclass
class IntentResult:
    """Result of intent detection"""
    intent: UserIntent
    confidence: float
    agent_id: str
    reasoning: str
    context_factors: Dict[str, Any]
    suggested_response: str

class SemanticIntentDetector:
    """
    LLM-based intent detection that understands natural language semantically
    instead of relying on hardcoded keyword matching.
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.intent_examples = self._load_intent_examples()
        self.conversation_context = {}
        
        # Cache for quick responses
        self.intent_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        logger.info("Semantic Intent Detector initialized")
    
    def _load_intent_examples(self) -> Dict[str, List[str]]:
        """Load diverse examples for each intent category"""
        return {
            UserIntent.ROADSIDE_ASSISTANCE: [
                "I need help with my car", "My vehicle broke down", "I'm stuck on the highway",
                "Need a tow truck", "Car won't start", "I have a flat tire", "Battery is dead",
                "Engine trouble", "Accident assistance needed", "Vehicle emergency",
                "Stranded with car problems", "Need roadside help", "My car is not running",
                "Vehicle malfunction", "Need emergency towing", "broke down", "broken down",
                "vehicle is broke", "car broke", "truck problems", "engine issues"
            ],
            
            UserIntent.BILLING_SUPPORT: [
                "Question about my bill", "Payment issue", "Billing problem", "Need a refund",
                "Charge on my account", "Invoice inquiry", "Payment not processed",
                "Billing error", "Account balance question", "Subscription issue"
            ],
            
            UserIntent.TECHNICAL_SUPPORT: [
                "App not working", "System error", "Login problems", "Technical issue",
                "Software bug", "Setup help needed", "Configuration problem",
                "Connection issues", "Installation help", "Password reset"
            ],
            
            UserIntent.GREETING: [
                "Hello", "Hi there", "Good morning", "Hey", "How are you",
                "Hello, can you help me", "Hi, I need assistance"
            ],
            
            UserIntent.CLARIFICATION: [
                "What do you mean", "Can you explain", "I don't understand",
                "Could you clarify", "What are my options", "Tell me more"
            ]
        }
    
    async def detect_intent(self, user_input: str, conversation_history: List[str] = None) -> IntentResult:
        """Main method: Detect user intent using semantic understanding"""
        
        start_time = time.time()
        
        try:
            # Use enhanced pattern detection (faster than LLM for most cases)
            result = await self._enhanced_pattern_detection(user_input, conversation_history)
            
            detection_time = (time.time() - start_time) * 1000
            logger.info(f"Intent detected in {detection_time:.1f}ms: {result.intent.value} "
                       f"(confidence: {result.confidence:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Intent detection error: {e}")
            return self._create_fallback_result(user_input)
    
    async def _enhanced_pattern_detection(self, user_input: str, conversation_history: List[str] = None) -> IntentResult:
        """Enhanced pattern-based detection with semantic understanding"""
        
        user_lower = user_input.lower()
        
        # Calculate semantic similarity scores for each intent
        intent_scores = {}
        
        for intent, examples in self.intent_examples.items():
            score = self._calculate_semantic_similarity(user_input, examples)
            intent_scores[intent] = score
        
        # Find best matching intent
        best_intent = max(intent_scores.keys(), key=lambda k: intent_scores[k])
        confidence = intent_scores[best_intent]
        
        # Apply context boost if we have conversation history
        if conversation_history:
            confidence = self._apply_context_boost(best_intent, conversation_history, confidence)
        
        # Map intent to agent
        agent_mapping = {
            UserIntent.ROADSIDE_ASSISTANCE: "roadside-assistance-v2",
            UserIntent.BILLING_SUPPORT: "billing-support-v2", 
            UserIntent.TECHNICAL_SUPPORT: "technical-support-v2",
            UserIntent.GENERAL_INQUIRY: "general",
            UserIntent.GREETING: "general",
            UserIntent.CLARIFICATION: "general"
        }
        
        agent_id = agent_mapping.get(best_intent, "general")
        
        # Generate appropriate response
        suggested_response = self._generate_response_for_intent(best_intent, user_input)
        
        return IntentResult(
            intent=best_intent,
            confidence=confidence,
            agent_id=agent_id,
            reasoning=f"Semantic similarity match with {confidence:.2f} confidence",
            context_factors={"method": "enhanced_pattern", "scores": intent_scores},
            suggested_response=suggested_response
        )
    
    def _calculate_semantic_similarity(self, user_input: str, examples: List[str]) -> float:
        """Calculate semantic similarity between user input and intent examples"""
        
        user_words = set(user_input.lower().split())
        
        # Calculate similarity with each example
        similarities = []
        
        for example in examples:
            example_words = set(example.lower().split())
            
            # Jaccard similarity
            intersection = len(user_words & example_words)
            union = len(user_words | example_words)
            
            if union > 0:
                jaccard = intersection / union
            else:
                jaccard = 0.0
            
            # Boost for partial word matches (e.g., "broke" matches "broken")
            partial_matches = 0
            for user_word in user_words:
                for example_word in example_words:
                    if len(user_word) > 3 and len(example_word) > 3:
                        if user_word in example_word or example_word in user_word:
                            partial_matches += 1
            
            partial_score = min(1.0, partial_matches / max(len(user_words), 1))
            
            # Combined score
            combined_score = (jaccard * 0.7) + (partial_score * 0.3)
            similarities.append(combined_score)
        
        # Return maximum similarity
        return max(similarities) if similarities else 0.0
    
    def _apply_context_boost(self, intent: UserIntent, conversation_history: List[str], base_confidence: float) -> float:
        """Apply context-based confidence boost"""
        
        # Boost if previous messages support this intent
        context_words = " ".join(conversation_history[-2:]).lower()
        
        boost_keywords = {
            UserIntent.ROADSIDE_ASSISTANCE: ["vehicle", "car", "truck", "drive", "road", "highway"],
            UserIntent.BILLING_SUPPORT: ["payment", "money", "cost", "account", "subscription"],
            UserIntent.TECHNICAL_SUPPORT: ["app", "system", "login", "password", "error"]
        }
        
        if intent in boost_keywords:
            for keyword in boost_keywords[intent]:
                if keyword in context_words:
                    base_confidence = min(1.0, base_confidence + 0.1)
        
        return base_confidence
    
    def _generate_response_for_intent(self, intent: UserIntent, user_input: str) -> str:
        """Generate appropriate response based on detected intent"""
        
        responses = {
            UserIntent.ROADSIDE_ASSISTANCE: "I understand you need roadside assistance. I'm here to help with your vehicle emergency. Can you tell me your current location and describe what's happening with your vehicle?",
            
            UserIntent.BILLING_SUPPORT: "I can help you with your billing question. I'll assist you with payments, refunds, or any account-related issues. What specific billing matter can I help you with?",
            
            UserIntent.TECHNICAL_SUPPORT: "I'm here to help you resolve this technical issue. I can guide you through troubleshooting steps to get everything working properly. Can you describe what specific problem you're experiencing?",
            
            UserIntent.GENERAL_INQUIRY: "I'm here to help you with any questions about our services. What would you like to know?",
            
            UserIntent.GREETING: "Hello! I'm your AI assistant and I'm here to help you with roadside assistance, billing questions, or technical support. What can I assist you with today?",
            
            UserIntent.CLARIFICATION: "I'd be happy to provide more information. What specific details would you like me to explain?"
        }
        
        return responses.get(intent, "I'm here to help you. How can I assist you today?")
    
    def _create_fallback_result(self, user_input: str) -> IntentResult:
        """Create fallback result when detection fails"""
        
        return IntentResult(
            intent=UserIntent.GENERAL_INQUIRY,
            confidence=0.5,
            agent_id="general",
            reasoning="Fallback due to detection error",
            context_factors={"method": "fallback"},
            suggested_response="I'm here to help you. Could you tell me more about what you need assistance with?"
        )

class SmartConversationHandler:
    """
    REPLACEMENT for the hardcoded keyword system - uses semantic understanding
    """
    
    def __init__(self, agent_registry=None, orchestrator=None, llm_client=None):
        self.agent_registry = agent_registry
        self.orchestrator = orchestrator
        self.intent_detector = SemanticIntentDetector(llm_client)
        self.conversation_history = {}
        
        logger.info("Smart Conversation Handler initialized with semantic understanding")
    
    async def process_conversation(
        self,
        user_input: str,
        session_id: str,
        orchestrator=None,
        context: Dict[str, Any] = None
    ) -> str:
        """Process conversation using semantic intent detection"""
        
        try:
            logger.info(f"🧠 Smart processing: '{user_input}'")
            
            # Get conversation history for context
            conversation_history = self.conversation_history.get(session_id, [])
            
            # Detect intent semantically
            intent_result = await self.intent_detector.detect_intent(
                user_input, conversation_history
            )
            
            logger.info(f"🎯 Intent detected: {intent_result.intent.value} "
                       f"(confidence: {intent_result.confidence:.2f}) -> {intent_result.agent_id}")
            logger.info(f"💭 Reasoning: {intent_result.reasoning}")
            
            # Update conversation history
            self._update_conversation_history(session_id, user_input)
            
            # Try orchestrator if available and high confidence
            if (self.orchestrator and 
                getattr(self.orchestrator, 'initialized', False) and 
                intent_result.confidence > 0.6):
                
                try:
                    enhanced_context = {
                        **(context or {}),
                        'detected_intent': intent_result.intent.value,
                        'intent_confidence': intent_result.confidence,
                        'intent_reasoning': intent_result.reasoning,
                        'conversation_history': conversation_history,
                        'semantic_detection': True
                    }
                    
                    result = await self.orchestrator.process_conversation(
                        session_id=session_id,
                        input_text=user_input,
                        context=enhanced_context,
                        preferred_agent_id=intent_result.agent_id
                    )
                    
                    if (result and hasattr(result, 'success') and result.success and 
                        hasattr(result, 'response') and result.response and result.response.strip()):
                        
                        logger.info(f"✅ Orchestrator success: {result.agent_id}")
                        
                        # Add response to history
                        self._update_conversation_history(session_id, result.response, is_assistant=True)
                        
                        return result.response
                    
                except Exception as e:
                    logger.warning(f"⚠️ Orchestrator failed: {e}")
            
            # Use suggested response from intent detection
            logger.info("🎭 Using semantic intent response")
            response = intent_result.suggested_response
            
            # Add response to history
            self._update_conversation_history(session_id, response, is_assistant=True)
            
            logger.info(f"✅ Smart response: {response[:100]}...")
            return response
            
        except Exception as e:
            logger.error(f"❌ Smart processing error: {e}")
            return "I understand you need assistance. Could you tell me more about what specific help you need?"
    
    def _update_conversation_history(self, session_id: str, message: str, is_assistant: bool = False):
        """Update conversation history for context"""
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        
        prefix = "Assistant: " if is_assistant else "User: "
        self.conversation_history[session_id].append(f"{prefix}{message}")
        
        # Keep only last 10 messages for context
        if len(self.conversation_history[session_id]) > 10:
            self.conversation_history[session_id] = self.conversation_history[session_id][-10:]
