#!/usr/bin/env python3
"""
Setup script for HS Code Validation System
Downloads and organizes the required model files
"""

import os
import requests
import zipfile
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    directories = [
        "models",
        "backups",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"[SUCCESS] Created directory: {directory}")

def download_model():
    """Download the Mistral model if not present"""
    model_name = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    model_path = Path("models") / model_name
    
    if model_path.exists():
        print(f"[SUCCESS] Model already exists: {model_path}")
        return str(model_path)
    
    print(f"📥 Model not found: {model_path}")
    print("Please download the model manually:")
    print("1. Go to: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF")
    print("2. Download: mistral-7b-instruct-v0.1.Q4_K_M.gguf")
    print("3. Place it in the 'models' directory")
    print(f"4. Expected path: {model_path}")
    
    return None

def check_files():
    """Check if required files exist"""
    required_files = [
        "Automating_HS_Code_validation.xlsx",
        "uk-tariff-2021-01-01--v4.0.1060--commodities-report.ods"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
        else:
            print(f"✅ Found: {file}")
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all required files are in the current directory.")
        return False
    
    return True

def main():
    """Main setup function"""
    print("🚀 Setting up HS Code Validation System")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Check required files
    print("\n📋 Checking required files...")
    if not check_files():
        print("\n❌ Setup incomplete. Please add missing files and run again.")
        return False
    
    # Check model
    print("\n🤖 Checking model...")
    model_path = download_model()
    
    if model_path:
        print(f"\n✅ Setup complete!")
        print(f"Model path: {model_path}")
        print("\nYou can now run: python ZenCargo_Production_Ready.py")
        return True
    else:
        print("\n⚠️ Setup incomplete. Please download the model and run again.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 