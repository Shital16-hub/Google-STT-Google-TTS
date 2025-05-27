#!/usr/bin/env python3
"""
Configuration Validator and Tester
Validate YAML configurations and test agent deployment before running main system.
"""
import sys
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List

class ConfigValidator:
    """Validate agent configurations."""
    
    def __init__(self):
        self.required_fields = [
            "agent_id", "version", "specialization", 
            "voice_settings", "tools", "routing"
        ]
        
        self.required_specialization_fields = [
            "domain_expertise", "personality_profile"
        ]
        
        self.required_voice_fields = [
            "tts_voice"
        ]
        
        self.required_routing_fields = [
            "primary_keywords"
        ]
    
    def validate_config(self, config: Dict[str, Any], config_name: str = "unknown") -> Dict[str, Any]:
        """Validate a single configuration."""
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "config_name": config_name
        }
        
        # Check required top-level fields
        for field in self.required_fields:
            if field not in config:
                result["valid"] = False
                result["errors"].append(f"Missing required field: {field}")
        
        # Validate specialization
        if "specialization" in config:
            spec = config["specialization"]
            if not isinstance(spec, dict):
                result["valid"] = False
                result["errors"].append("specialization must be a dictionary")
            else:
                for field in self.required_specialization_fields:
                    if field not in spec:
                        result["valid"] = False
                        result["errors"].append(f"Missing specialization field: {field}")
        
        # Validate voice_settings
        if "voice_settings" in config:
            voice = config["voice_settings"]
            if not isinstance(voice, dict):
                result["valid"] = False
                result["errors"].append("voice_settings must be a dictionary")
            else:
                for field in self.required_voice_fields:
                    if field not in voice:
                        result["warnings"].append(f"Missing voice_settings field: {field}")
        
        # Validate tools
        if "tools" in config:
            if not isinstance(config["tools"], list):
                result["valid"] = False
                result["errors"].append("tools must be a list")
            elif len(config["tools"]) == 0:
                result["warnings"].append("No tools specified")
        
        # Validate routing
        if "routing" in config:
            routing = config["routing"]
            if not isinstance(routing, dict):
                result["valid"] = False
                result["errors"].append("routing must be a dictionary")
            else:
                for field in self.required_routing_fields:
                    if field not in routing:
                        result["warnings"].append(f"Missing routing field: {field}")
                
                # Check keywords
                if "primary_keywords" in routing:
                    keywords = routing["primary_keywords"]
                    if not isinstance(keywords, list):
                        result["valid"] = False
                        result["errors"].append("primary_keywords must be a list")
                    elif len(keywords) == 0:
                        result["warnings"].append("No primary_keywords specified")
        
        return result

def load_yaml_file(file_path: Path) -> Dict[str, Any]:
    """Load and parse YAML file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return {}

def main():
    """Main validation function."""
    print("🔍 Configuration Validator for Multi-Agent Voice AI System")
    print("=" * 60)
    
    config_dir = Path("app/config/agents")
    
    if not config_dir.exists():
        print(f"❌ Config directory not found: {config_dir}")
        print("   Please ensure you're running from the project root directory")
        return 1
    
    # Find all YAML files
    yaml_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))
    yaml_files = [f for f in yaml_files if "template" not in f.name.lower()]
    
    if not yaml_files:
        print(f"❌ No YAML config files found in {config_dir}")
        return 1
    
    print(f"📄 Found {len(yaml_files)} configuration files")
    print()
    
    validator = ConfigValidator()
    results = []
    
    for yaml_file in yaml_files:
        print(f"🔍 Validating: {yaml_file.name}")
        
        config = load_yaml_file(yaml_file)
        if not config:
            print(f"   ❌ Failed to load configuration")
            continue
        
        result = validator.validate_config(config, yaml_file.name)
        results.append(result)
        
        if result["valid"]:
            print(f"   ✅ Valid configuration")
            agent_id = config.get("agent_id", "unknown")
            version = config.get("version", "unknown")
            domain = config.get("specialization", {}).get("domain_expertise", "unknown")
            print(f"      Agent ID: {agent_id}")
            print(f"      Version: {version}")
            print(f"      Domain: {domain}")
        else:
            print(f"   ❌ Invalid configuration")
            for error in result["errors"]:
                print(f"      ERROR: {error}")
        
        if result["warnings"]:
            for warning in result["warnings"]:
                print(f"      WARNING: {warning}")
        
        print()
    
    # Summary
    valid_configs = [r for r in results if r["valid"]]
    invalid_configs = [r for r in results if not r["valid"]]
    
    print("=" * 60)
    print("📊 VALIDATION SUMMARY")
    print(f"✅ Valid configurations: {len(valid_configs)}")
    print(f"❌ Invalid configurations: {len(invalid_configs)}")
    
    if invalid_configs:
        print("\n❌ INVALID CONFIGURATIONS:")
        for result in invalid_configs:
            print(f"   {result['config_name']}")
            for error in result["errors"]:
                print(f"      - {error}")
    
    # Check for duplicate agent IDs
    agent_ids = []
    for yaml_file in yaml_files:
        config = load_yaml_file(yaml_file)
        if config and "agent_id" in config:
            agent_ids.append(config["agent_id"])
    
    if len(agent_ids) != len(set(agent_ids)):
        print("\n⚠️  WARNING: Duplicate agent IDs detected!")
        duplicates = [aid for aid in set(agent_ids) if agent_ids.count(aid) > 1]
        for dup in duplicates:
            print(f"   Duplicate: {dup}")
    
    print("\n🎯 RECOMMENDATIONS:")
    if len(valid_configs) == 0:
        print("   1. Fix configuration errors before running the system")
        print("   2. Ensure all required fields are present")
        print("   3. Check the agent_template.yaml for reference")
    elif len(invalid_configs) > 0:
        print("   1. Fix invalid configurations to enable those agents")
        print("   2. Valid configurations will be deployed successfully")
    else:
        print("   🎉 All configurations are valid! System should start successfully.")
    
    return 1 if invalid_configs else 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)