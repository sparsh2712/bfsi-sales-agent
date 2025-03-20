"""
Script to create necessary directories for the BFSI Sales Agent
"""
import os
import sys

def create_directories():
    """Create all required directories for the application"""
    directories = [
        'data/documents',
        'data/indices',
        'data/audio_cache',
        'logs',
        'config/use_cases'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    print("Directory setup complete!")

if __name__ == "__main__":
    create_directories()