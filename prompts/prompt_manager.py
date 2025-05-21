# prompts/prompt_manager.py

"""
Centralized prompt management system for specialized agents.
"""
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json

from agents.base_agent import AgentType
from core.state_manager import ConversationState

logger = logging.getLogger(__name__)

class PromptManager:
    """Manages and customizes prompts for different agents and states."""
    
    def __init__(self, prompt_dir: str = "prompts"):
        """
        Initialize prompt manager.
        
        Args:
            prompt_dir: Directory containing prompt templates
        """
        self.prompt_dir = Path(prompt_dir)
        self.templates: Dict[str, str] = {}
        self.load_templates()
        
        # Common response patterns
        self.confirmation_patterns = {
            "name": "Thank you, {name}. ",
            "phone": "I've got your phone number as {phone}. ",
            "location": "You're located at {location}. ",
            "vehicle": "You have a {year} {make} {model}. "
        }
        
        logger.info("Initialized prompt manager")
    
    def load_templates(self):
        """Load prompt templates from files."""
        try:
            # Load base prompts
            base_path = self.prompt_dir / "base_prompts.json"
            if base_path.exists():
                with open(base_path) as f:
                    self.templates.update(json.load(f))
            
            # Load agent-specific prompts
            for agent_type in AgentType:
                agent_path = self.prompt_dir / f"agent_prompts/{agent_type.value}.json"
                if agent_path.exists():
                    with open(agent_path) as f:
                        self.templates.update(json.load(f))
            
            logger.info(f"Loaded {len(self.templates)} prompt templates")
            
        except Exception as e:
            logger.error(f"Error loading prompt templates: {e}")
            raise
    
    def get_prompt(
        self,
        agent_type: AgentType,
        state: ConversationState,
        collected_info: Dict[str, Any]
    ) -> str:
        """
        Get appropriate prompt for the current context.
        
        Args:
            agent_type: Type of agent
            state: Current conversation state
            collected_info: Information collected so far
            
        Returns:
            Formatted prompt string
        """
        # Get base template
        template_key = f"{agent_type.value}_{state.value}"
        template = self.templates.get(template_key)
        
        if not template:
            # Fall back to generic template
            template = self.templates.get(f"generic_{state.value}")
            if not template:
                logger.warning(f"No template found for {template_key}")
                return self._get_fallback_prompt(state)
        
        # Add confirmations for collected info
        confirmations = self._get_confirmations(collected_info)
        
        # Build system context
        context = self._build_context(agent_type, collected_info)
        
        # Format final prompt
        try:
            prompt = template.format(
                confirmations=confirmations,
                context=context,
                **collected_info
            )
            return prompt
        except KeyError as e:
            logger.error(f"Error formatting prompt: missing key {e}")
            return self._get_fallback_prompt(state)
    
    def _get_confirmations(self, collected_info: Dict[str, Any]) -> str:
        """
        Get confirmation messages for collected information.
        
        Args:
            collected_info: Collected information
            
        Returns:
            Formatted confirmation string
        """
        confirmations = []
        
        for field, pattern in self.confirmation_patterns.items():
            if field in collected_info and collected_info[field]:
                try:
                    confirmation = pattern.format(**collected_info)
                    confirmations.append(confirmation)
                except KeyError:
                    continue
        
        return " ".join(confirmations)
    
    def _build_context(
        self,
        agent_type: AgentType,
        collected_info: Dict[str, Any]
    ) -> str:
        """
        Build context string for prompt.
        
        Args:
            agent_type: Type of agent
            collected_info: Collected information
            
        Returns:
            Formatted context string
        """
        context_parts = [
            f"You are a specialized {agent_type.value} service agent.",
            "Be concise and professional in your responses.",
            "Ask for one piece of information at a time.",
            "Always confirm received information."
        ]
        
        # Add service-specific context
        if agent_type == AgentType.TOWING:
            if "vehicle_condition" in collected_info:
                context_parts.append(
                    f"The vehicle is in the following condition: {collected_info['vehicle_condition']}"
                )
        elif agent_type == AgentType.TIRE:
            if "tire_location" in collected_info:
                context_parts.append(
                    f"The flat tire is on the {collected_info['tire_location']}"
                )
        
        return "\n".join(context_parts)
    
    def _get_fallback_prompt(self, state: ConversationState) -> str:
        """
        Get fallback prompt for when no template is found.
        
        Args:
            state: Current conversation state
            
        Returns:
            Fallback prompt string
        """
        if state == ConversationState.GREETING:
            return "You are a roadside assistance agent. Greet the customer and ask how you can help."
        elif state == ConversationState.COLLECTING_NAME:
            return "Ask for the customer's name politely."
        elif state == ConversationState.COLLECTING_PHONE:
            return "Ask for the customer's phone number."
        elif state == ConversationState.COLLECTING_LOCATION:
            return "Ask for the customer's current location."
        else:
            return "Continue helping the customer professionally and efficiently."
    
    def add_template(
        self,
        key: str,
        template: str,
        persist: bool = True
    ):
        """
        Add a new prompt template.
        
        Args:
            key: Template key
            template: Template string
            persist: Whether to save to file
        """
        self.templates[key] = template
        
        if persist:
            try:
                # Determine appropriate file
                if "_" in key:
                    agent_type = key.split("_")[0]
                    file_path = self.prompt_dir / f"agent_prompts/{agent_type}.json"
                else:
                    file_path = self.prompt_dir / "base_prompts.json"
                
                # Load existing templates
                templates = {}
                if file_path.exists():
                    with open(file_path) as f:
                        templates = json.load(f)
                
                # Add new template
                templates[key] = template
                
                # Save back to file
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, 'w') as f:
                    json.dump(templates, f, indent=2)
                
                logger.info(f"Added new template: {key}")
            except Exception as e:
                logger.error(f"Error persisting template: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get prompt manager statistics."""
        stats = {
            "total_templates": len(self.templates),
            "templates_by_type": {}
        }
        
        # Count templates by type
        for key in self.templates:
            if "_" in key:
                agent_type = key.split("_")[0]
                if agent_type not in stats["templates_by_type"]:
                    stats["templates_by_type"][agent_type] = 0
                stats["templates_by_type"][agent_type] += 1
        
        return stats