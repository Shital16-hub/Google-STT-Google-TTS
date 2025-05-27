import os
import re

def patch_file(filepath, patterns):
    """Patch a file with given patterns."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original_content = content
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        if content != original_content:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"✅ Patched {filepath}")
            return True
        else:
            print(f"ℹ️ No changes needed in {filepath}")
            return False
            
    except Exception as e:
        print(f"❌ Error patching {filepath}: {e}")
        return False

# Patterns to fix Qdrant connection
patterns = [
    (r'"prefer_grpc": True', '"prefer_grpc": False'),
    (r'prefer_grpc: bool = True', 'prefer_grpc: bool = False'),
    (r'prefer_grpc=True', 'prefer_grpc=False'),
    (r'self\.prefer_grpc = prefer_grpc', 'self.prefer_grpc = False  # Force HTTP for RunPod'),
]

# Files to patch
files_to_patch = [
    'app/main.py',
    'app/vector_db/qdrant_manager.py',
    'app/vector_db/hybrid_vector_system.py'
]

print("🔧 Patching Qdrant connection to use HTTP...")

for filepath in files_to_patch:
    if os.path.exists(filepath):
        patch_file(filepath, patterns)
    else:
        print(f"⚠️ File not found: {filepath}")

print("✅ Patching complete!")
