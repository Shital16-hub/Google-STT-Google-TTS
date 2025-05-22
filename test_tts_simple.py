# test_tts_simple.py - Simple TTS test without custom voice parameters

import asyncio
import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_simple_tts():
    """Test Google Cloud TTS with simple configuration."""
    
    print("🗣️ Testing Google Cloud TTS (Simple Configuration)...")
    
    try:
        from google.cloud import texttospeech
        
        # Test 1: Basic client initialization
        print("1️⃣ Testing client initialization...")
        client = texttospeech.TextToSpeechClient()
        print("✅ TTS client initialized successfully")
        
        # Test 2: List available voices
        print("2️⃣ Testing voice listing...")
        try:
            voices_request = texttospeech.ListVoicesRequest(language_code="en-US")
            voices_response = client.list_voices(request=voices_request)
            
            neural2_voices = [v for v in voices_response.voices if "Neural2" in v.name]
            print(f"✅ Found {len(neural2_voices)} Neural2 voices for en-US")
            
            # Print first few voices for debugging
            for i, voice in enumerate(neural2_voices[:3]):
                print(f"   Voice {i+1}: {voice.name}")
                
        except Exception as e:
            print(f"⚠️ Voice listing failed: {e}")
        
        # Test 3: Simple synthesis with standard voice
        print("3️⃣ Testing synthesis with standard voice...")
        try:
            # Use a simple standard voice first
            synthesis_input = texttospeech.SynthesisInput(text="Hello, this is a test.")
            
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
            )
            
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16
            )
            
            response = client.synthesize_speech(
                input=synthesis_input, 
                voice=voice, 
                audio_config=audio_config
            )
            
            print(f"✅ Standard voice synthesis successful - {len(response.audio_content)} bytes")
            
        except Exception as e:
            print(f"❌ Standard voice synthesis failed: {e}")
            return False
        
        # Test 4: Neural2 voice synthesis
        print("4️⃣ Testing Neural2 voice synthesis...")
        try:
            synthesis_input = texttospeech.SynthesisInput(text="Hello, this is a Neural2 test.")
            
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name="en-US-Neural2-C"  # Specific Neural2 voice
                # No gender parameter for Neural2 voices
            )
            
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16
            )
            
            response = client.synthesize_speech(
                input=synthesis_input, 
                voice=voice, 
                audio_config=audio_config
            )
            
            print(f"✅ Neural2 voice synthesis successful - {len(response.audio_content)} bytes")
            
        except Exception as e:
            print(f"❌ Neural2 voice synthesis failed: {e}")
            print(f"   Error details: {str(e)}")
            
            # Try with a different Neural2 voice
            print("   Trying alternative Neural2 voice...")
            try:
                voice = texttospeech.VoiceSelectionParams(
                    language_code="en-US",
                    name="en-US-Neural2-A"  # Try different Neural2 voice
                )
                
                response = client.synthesize_speech(
                    input=synthesis_input, 
                    voice=voice, 
                    audio_config=audio_config
                )
                
                print(f"✅ Alternative Neural2 voice successful - {len(response.audio_content)} bytes")
                
            except Exception as e2:
                print(f"❌ Alternative Neural2 voice also failed: {e2}")
                return False
        
        # Test 5: MULAW format for Twilio
        print("5️⃣ Testing MULAW format for Twilio...")
        try:
            synthesis_input = texttospeech.SynthesisInput(text="Testing MULAW format.")
            
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name="en-US-Neural2-C"
            )
            
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MULAW,
                sample_rate_hertz=8000,
                effects_profile_id=["telephony-class-application"]
            )
            
            response = client.synthesize_speech(
                input=synthesis_input, 
                voice=voice, 
                audio_config=audio_config
            )
            
            print(f"✅ MULAW format synthesis successful - {len(response.audio_content)} bytes")
            
        except Exception as e:
            print(f"❌ MULAW format synthesis failed: {e}")
            return False
        
        print("🎉 All TTS tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ TTS test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_integration_tts():
    """Test the integration TTS class."""
    
    print("\n🔧 Testing Integration TTS Class...")
    
    try:
        sys.path.append('.')
        from integration.tts_integration import TTSIntegration
        
        # Initialize with safe parameters
        tts = TTSIntegration(
            voice_name="en-US-Neural2-C",
            language_code="en-US",
            container_format="linear16",  # Use LINEAR16 instead of MULAW for testing
            sample_rate=16000,  # Use 16kHz instead of 8kHz for testing
            enable_caching=True,
            credentials_file=os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        )
        
        await tts.init()
        print("✅ Integration TTS initialized")
        
        # Test synthesis
        audio_data = await tts.synthesize("Hello from integration TTS!")
        print(f"✅ Integration TTS synthesis successful - {len(audio_data)} bytes")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration TTS failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting TTS Tests...")
    
    # Test simple TTS first
    success1 = asyncio.run(test_simple_tts())
    
    if success1:
        # Test integration TTS
        success2 = asyncio.run(test_integration_tts())
        
        if success2:
            print("\n✅ All TTS tests passed! TTS should work in your application.")
        else:
            print("\n⚠️ Integration TTS failed, but basic TTS works.")
    else:
        print("\n❌ Basic TTS tests failed. Check your Google Cloud setup.")
        
    print("\nNext steps:")
    print("1. Replace your text_to_speech/google_cloud_tts.py with the fixed version")
    print("2. Restart your application")
    print("3. Test again")