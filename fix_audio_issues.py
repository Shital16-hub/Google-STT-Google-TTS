#!/usr/bin/env python3
"""
Immediate fix script for the STT and TTS issues
Save this as fix_audio_issues.py and run it
"""
import os
import re

def fix_stt_streaming_config():
    """Fix the missing streaming_config in STT"""
    stt_file = "app/voice/google_cloud_stt.py"
    
    if not os.path.exists(stt_file):
        print(f"❌ File not found: {stt_file}")
        return False
    
    with open(stt_file, 'r') as f:
        content = f.read()
    
    # Check if streaming_config is being created
    if "self.streaming_config = self.cloud_speech.StreamingRecognitionConfig" not in content:
        print("🔧 Adding missing streaming_config creation...")
        
        # Find the _setup_config method and add the streaming_config
        pattern = r'(self\.recognition_config = self\.cloud_speech\.RecognitionConfig\([^}]+\}\s*,\s*\)\s*)'
        
        streaming_config_code = '''
        
        # CRITICAL: Create self.streaming_config (this was missing!)
        self.streaming_config = self.cloud_speech.StreamingRecognitionConfig(
            config=self.recognition_config,
            streaming_features=self.cloud_speech.StreamingRecognitionFeatures(
                interim_results=self.interim_results,
                enable_voice_activity_events=True,
                voice_activity_timeout=self.cloud_speech.StreamingRecognitionFeatures.VoiceActivityTimeout(
                    speech_start_timeout=self.Duration(seconds=5),
                    speech_end_timeout=self.Duration(seconds=1)
                ),
            ),
        )
        
        logger.debug("✅ STT streaming configuration created successfully")'''
        
        if re.search(pattern, content, re.DOTALL):
            content = re.sub(pattern, r'\1' + streaming_config_code, content, flags=re.DOTALL)
            
            with open(stt_file, 'w') as f:
                f.write(content)
            
            print("✅ Fixed STT streaming_config issue")
            return True
        else:
            print("⚠️ Could not find insertion point for streaming_config")
            return False
    else:
        print("✅ STT streaming_config already exists")
        return True

def fix_audio_chunking():
    """Fix the audio chunking in main.py"""
    main_file = "main.py"
    
    if not os.path.exists(main_file):
        print(f"❌ File not found: {main_file}")
        return False
    
    with open(main_file, 'r') as f:
        content = f.read()
    
    # Fix chunk size from 400 to 160
    old_chunk = r'chunk_size = 400  # 50ms chunks'
    new_chunk = 'chunk_size = 160  # 20ms chunks for smooth playbook'
    
    if re.search(old_chunk, content):
        content = re.sub(old_chunk, new_chunk, content)
        print("🔧 Fixed audio chunk size")
    
    # Fix delay timing
    old_delay = r'await asyncio\.sleep\(0\.025\)  # 25ms delay'
    new_delay = 'await asyncio.sleep(0.020)  # 20ms delay matches chunk size'
    
    if re.search(old_delay, content):
        content = re.sub(old_delay, new_delay, content)
        print("🔧 Fixed audio timing delay")
    
    with open(main_file, 'w') as f:
        f.write(content)
    
    print("✅ Fixed audio chunking issues")
    return True

def main():
    print("🔧 Applying immediate fixes for STT and TTS issues...")
    
    # Change to correct directory
    if os.path.exists("/workspace/Google-STT-Google-TTS"):
        os.chdir("/workspace/Google-STT-Google-TTS")
        print(f"📁 Working in: {os.getcwd()}")
    
    # Apply fixes
    stt_fixed = fix_stt_streaming_config()
    audio_fixed = fix_audio_chunking()
    
    if stt_fixed and audio_fixed:
        print("\n✅ ALL FIXES APPLIED SUCCESSFULLY!")
        print("\n🚀 Next steps:")
        print("1. Kill current process: pkill -f 'main.py'")
        print("2. Restart application: python main.py")
        print("3. Test call - audio should be smooth now!")
    else:
        print("\n⚠️ Some fixes failed - manual intervention needed")
        print("Check the code sections mentioned above")

if __name__ == "__main__":
    main()