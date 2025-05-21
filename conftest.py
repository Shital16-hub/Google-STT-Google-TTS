# tests/conftest.py

"""Test configuration and fixtures."""
import pytest
import asyncio
from typing import Dict, Any
from pathlib import Path

from core.config import Settings
from core.conversation_manager import ConversationManager
from knowledge_base.query_engine import QueryEngine
from prompts.prompt_manager import PromptManager
from agents.router import AgentRouter
from services.dispatcher import DispatcherService

@pytest.fixture
def settings():
    """Test settings."""
    return Settings(
        debug=True,
        environment="test",
        base_url="http://test.local",
        storage_dir="./test_storage"
    )

@pytest.fixture
async def query_engine():
    """Test query engine."""
    engine = QueryEngine()
    await engine.init()
    yield engine
    await engine.cleanup()

@pytest.fixture
def prompt_manager():
    """Test prompt manager."""
    return PromptManager(prompt_dir="./test_prompts")

@pytest.fixture
async def conversation_manager(query_engine, prompt_manager):
    """Test conversation manager."""
    manager = ConversationManager(
        query_engine=query_engine,
        prompt_manager=prompt_manager
    )
    await manager.init()
    yield manager
    await manager.cleanup()

@pytest.fixture
async def agent_router(conversation_manager, query_engine, prompt_manager):
    """Test agent router."""
    return AgentRouter(
        conversation_manager=conversation_manager,
        query_engine=query_engine,
        prompt_manager=prompt_manager
    )

@pytest.fixture
def dispatcher_service():
    """Test dispatcher service."""
    return DispatcherService()

# tests/test_agents/test_base_agent.py

"""Tests for base agent functionality."""
import pytest
from agents.base_agent import BaseAgent, AgentType

async def test_agent_initialization(conversation_manager, query_engine, prompt_manager):
    """Test agent initialization."""
    agent = BaseAgent(
        agent_type=AgentType.TOWING,
        conversation_manager=conversation_manager,
        query_engine=query_engine,
        prompt_manager=prompt_manager
    )
    assert agent.agent_type == AgentType.TOWING
    assert agent.conversation_manager == conversation_manager

async def test_agent_handle_message(conversation_manager, query_engine, prompt_manager):
    """Test message handling."""
    agent = BaseAgent(
        agent_type=AgentType.TOWING,
        conversation_manager=conversation_manager,
        query_engine=query_engine,
        prompt_manager=prompt_manager
    )
    response = await agent.handle_message("Hello")
    assert "response" in response
    assert "state" in response

# tests/test_agents/test_router.py

"""Tests for agent routing functionality."""
import pytest
from agents.router import AgentRouter
from agents.base_agent import AgentType

async def test_router_intent_detection(agent_router):
    """Test intent detection."""
    intent = await agent_router.determine_intent("I need a tow truck")
    assert intent == AgentType.TOWING

async def test_router_agent_creation(agent_router):
    """Test agent creation."""
    agent = agent_router.get_agent("test_session", AgentType.TOWING)
    assert agent.agent_type == AgentType.TOWING

# tests/test_core/test_conversation_manager.py

"""Tests for conversation management."""
import pytest
from core.conversation_manager import ConversationManager
from core.state_manager import ConversationState

async def test_conversation_processing(conversation_manager):
    """Test conversation processing."""
    response = await conversation_manager.process_message("Hello")
    assert "response" in response
    assert "state" in response

async def test_state_transitions(conversation_manager):
    """Test state transitions."""
    await conversation_manager.process_message("Hello")
    assert conversation_manager.state_manager.current_state != ConversationState.GREETING

# tests/test_services/test_dispatcher.py

"""Tests for dispatcher service."""
import pytest
from services.dispatcher import DispatcherService

async def test_service_request_creation(dispatcher_service):
    """Test service request creation."""
    request_id = await dispatcher_service.create_service_request(
        session_id="test_session",
        agent_type=AgentType.TOWING,
        customer_info={"name": "Test User"},
        service_requirements={},
        handoff_reason="test"
    )
    assert request_id is not None

# tests/test_utils/test_audio_utils.py

"""Tests for audio utilities."""
import pytest
import numpy as np
from utils.audio_utils import normalize_audio, detect_silence

def test_audio_normalization():
    """Test audio normalization."""
    audio = np.random.randn(1000)
    normalized = normalize_audio(audio)
    assert np.max(np.abs(normalized)) <= 1.0

def test_silence_detection():
    """Test silence detection."""
    audio = np.zeros(1000)
    audio[300:700] = 1.0
    regions = detect_silence(audio)
    assert len(regions) == 2  # Start and end silence regions

# Create test directories
DIRS = [
    'tests/test_agents',
    'tests/test_core',
    'tests/test_prompts',
    'tests/test_api',
    'tests/test_services',
    'tests/test_utils'
]

# Create __init__.py files in each test directory
INIT_CONTENT = '"""Tests for {} module."""'

for dir_name in DIRS:
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    init_file = Path(dir_name) / '__init__.py'
    init_file.write_text(INIT_CONTENT.format(dir_name.split('/')[-1]))