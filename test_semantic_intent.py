#!/usr/bin/env python3
"""
Test script for semantic intent detection
"""
import asyncio
import sys
import os

# Add the project path
sys.path.insert(0, '/workspace/Google-STT-Google-TTS')

from app.llm.semantic_intent_detector import SemanticIntentDetector, UserIntent

async def test_semantic_detection():
    """Test the semantic intent detection with problematic cases"""
    
    detector = SemanticIntentDetector()
    
    # These were FAILING with the old keyword system
    failing_test_cases = [
        "My vehicle is broke down",          # ❌ Old system failed
        "Uh, my vehicle is break broke down", # ❌ Old system failed  
        "Car won't start this morning",      # ❌ Old system failed
        "I'm stuck on the side of the road", # ❌ Old system failed
        "Engine is making strange noises",   # ❌ Old system failed
        "Battery completely dead",           # ❌ Old system failed
        "Flat tire on highway",             # ❌ Old system failed
        "Vehicle malfunction",               # ❌ Old system failed
        "Stranded and need assistance",      # ❌ Old system failed
    ]
    
    # These were working with the old system
    working_test_cases = [
        "Need roadside help",               # ✅ Old system worked
        "I need roadside help",             # ✅ Old system worked
    ]
    
    print("🧪 Testing Semantic Intent Detection System")
    print("=" * 60)
    
    print("\n📊 Cases that FAILED with old keyword system:")
    print("-" * 50)
    
    all_correct = True
    
    for test_input in failing_test_cases:
        result = await detector.detect_intent(test_input)
        
        is_correct = result.intent == UserIntent.ROADSIDE_ASSISTANCE
        status = "✅ FIXED" if is_correct else "❌ STILL BROKEN"
        
        if not is_correct:
            all_correct = False
        
        print(f"{status} '{test_input}'")
        print(f"    Intent: {result.intent.value}")
        print(f"    Agent: {result.agent_id}")
        print(f"    Confidence: {result.confidence:.2f}")
        print()
    
    print("\n📊 Cases that worked with old keyword system:")
    print("-" * 50)
    
    for test_input in working_test_cases:
        result = await detector.detect_intent(test_input)
        
        is_correct = result.intent == UserIntent.ROADSIDE_ASSISTANCE
        status = "✅ STILL WORKS" if is_correct else "❌ BROKEN"
        
        if not is_correct:
            all_correct = False
        
        print(f"{status} '{test_input}'")
        print(f"    Intent: {result.intent.value}")
        print(f"    Agent: {result.agent_id}")
        print(f"    Confidence: {result.confidence:.2f}")
        print()
    
    print("=" * 60)
    if all_correct:
        print("🎉 ALL TESTS PASSED! Semantic system is working correctly.")
        print("   The problematic phrases should now route to roadside assistance.")
    else:
        print("⚠️ Some tests failed. Check the semantic detection logic.")
    
    return all_correct

if __name__ == "__main__":
    success = asyncio.run(test_semantic_detection())
    sys.exit(0 if success else 1)
