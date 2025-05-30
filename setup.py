#!/usr/bin/env python3
"""
Setup script to fix OpenAI authentication and install NLTK data
"""
import os
import nltk
import openai

def fix_openai_auth():
    """Fix OpenAI authentication issues"""
    print("🔧 Fixing OpenAI Authentication...")
    
    # Remove problematic organization header
    if 'OPENAI_ORGANIZATION' in os.environ:
        print("   Removing OPENAI_ORGANIZATION environment variable")
        del os.environ['OPENAI_ORGANIZATION']
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        print("   Please set: export OPENAI_API_KEY='your-api-key-here'")
        return False
    
    print(f"✅ OpenAI API key found: {api_key[:10]}...")
    return True

def install_nltk_data():
    """Install required NLTK data"""
    print("📦 Installing NLTK data...")
    
    try:
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        print("✅ NLTK data installed successfully")
        return True
    except Exception as e:
        print(f"❌ Error installing NLTK data: {e}")
        return False

def test_openai_connection():
    """Test OpenAI connection"""
    print("🧪 Testing OpenAI connection...")
    
    try:
        from openai import OpenAI
        client = OpenAI()
        
        # Simple test call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        
        print("✅ OpenAI connection successful")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Setting up fixes for RunPod deployment...\n")
    
    # Fix authentication
    auth_ok = fix_openai_auth()
    
    # Install NLTK data
    nltk_ok = install_nltk_data()
    
    # Test connection
    if auth_ok:
        connection_ok = test_openai_connection()
    else:
        connection_ok = False
    
    print(f"\n📋 Setup Summary:")
    print(f"   OpenAI Auth: {'✅' if auth_ok else '❌'}")
    print(f"   NLTK Data: {'✅' if nltk_ok else '❌'}")
    print(f"   OpenAI Connection: {'✅' if connection_ok else '❌'}")
    
    if auth_ok and nltk_ok and connection_ok:
        print("\n🎉 All fixes applied successfully!")
    else:
        print("\n⚠️ Some issues remain. Check the errors above.")