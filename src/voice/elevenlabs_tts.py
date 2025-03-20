"""
ElevenLabs Text-to-Speech integration
"""
import os
import logging
import requests
import tempfile
import json

class ElevenLabsTTS:
    """
    Text-to-Speech integration with ElevenLabs API
    Supports female voice with South Indian accent
    """
    
    def __init__(self, config):
        """
        Initialize the ElevenLabs TTS component
        
        Args:
            config (dict): ElevenLabs configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Get API key from environment or config
        self.api_key = os.environ.get("ELEVENLABS_API_KEY", config.get("api_key"))
        
        if not self.api_key:
            self.logger.warning("ElevenLabs API key not configured")
        
        # Configuration
        self.voice_id = config.get("voice_id", "21m00Tcm4TlvDq8ikWAM")  # Default voice
        self.model_id = config.get("model_id", "eleven_multilingual_v2")
        self.base_url = "https://api.elevenlabs.io/v1"
        
        # Voice settings for Indian accent
        self.voice_settings = {
            "stability": config.get("stability", 0.5),
            "similarity_boost": config.get("similarity_boost", 0.75),
            "style": config.get("style", 0.0),
            "use_speaker_boost": config.get("use_speaker_boost", True)
        }
        
        self.logger.info("ElevenLabs TTS initialized")
    
    def synthesize_speech(self, text):
        """
        Synthesize speech from text
        
        Args:
            text (str): Text to synthesize
            
        Returns:
            str: Path to audio file or None if failed
        """
        if not self.api_key:
            self.logger.error("ElevenLabs API key not configured")
            return None
        
        self.logger.info(f"Synthesizing speech: '{text[:50]}...'")
        
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        data = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": self.voice_settings
        }
        
        try:
            url = f"{self.base_url}/text-to-speech/{self.voice_id}"
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                # Save to temporary file
                audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                audio_file.write(response.content)
                audio_file.close()
                
                self.logger.info(f"Speech synthesized successfully, saved to {audio_file.name}")
                return audio_file.name
            else:
                self.logger.error(f"Failed to synthesize speech: {response.status_code} - {response.text}")
                return None
        
        except Exception as e:
            self.logger.error(f"Error synthesizing speech: {str(e)}")
            return None
    
    def get_available_voices(self):
        """
        Get available voices from ElevenLabs
        
        Returns:
            list: Available voices or None if failed
        """
        if not self.api_key:
            self.logger.error("ElevenLabs API key not configured")
            return None
        
        headers = {
            "xi-api-key": self.api_key
        }
        
        try:
            url = f"{self.base_url}/voices"
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                voices = response.json().get("voices", [])
                self.logger.info(f"Retrieved {len(voices)} available voices")
                return voices
            else:
                self.logger.error(f"Failed to get available voices: {response.status_code} - {response.text}")
                return None
        
        except Exception as e:
            self.logger.error(f"Error getting available voices: {str(e)}")
            return None
    
    def create_voice(self, name, description, files):
        """
        Create a new voice (for custom South Indian accent)
        
        Args:
            name (str): Voice name
            description (str): Voice description
            files (list): List of audio file paths for samples
            
        Returns:
            str: Voice ID or None if failed
        """
        if not self.api_key:
            self.logger.error("ElevenLabs API key not configured")
            return None
        
        headers = {
            "xi-api-key": self.api_key
        }
        
        # Prepare files for upload
        files_data = []
        for file_path in files:
            with open(file_path, "rb") as f:
                files_data.append(("files", f))
        
        data = {
            "name": name,
            "description": description
        }
        
        try:
            url = f"{self.base_url}/voices/add"
            response = requests.post(url, headers=headers, data=data, files=files_data)
            
            if response.status_code == 200:
                voice_id = response.json().get("voice_id")
                self.logger.info(f"Voice created successfully, ID: {voice_id}")
                return voice_id
            else:
                self.logger.error(f"Failed to create voice: {response.status_code} - {response.text}")
                return None
        
        except Exception as e:
            self.logger.error(f"Error creating voice: {str(e)}")
            return None