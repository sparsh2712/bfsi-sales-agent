"""
Enhanced call handling module for integrating Twilio with ElevenLabs TTS
"""
import os
import logging
import time
import tempfile
from flask import Flask, request, Response, send_file
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.rest import Client
from threading import Thread
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .elevenlabs_tts import ElevenLabsTTS

class CallHandler:
    """
    Enhanced call handler that uses ElevenLabs for high-quality TTS
    and Twilio for voice call management.
    """
    
    def __init__(self, config, sales_agent):
        """
        Initialize the call handler
        
        Args:
            config (dict): Voice configuration
            sales_agent (SalesAgent): Sales agent instance
        """
        self.config = config
        self.sales_agent = sales_agent
        self.logger = logging.getLogger(__name__)
        
        # Initialize Twilio client
        self.account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        self.auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
        self.twilio_number = os.environ.get("TWILIO_PHONE_NUMBER")
        
        if not all([self.account_sid, self.auth_token, self.twilio_number]):
            self.logger.warning("Twilio credentials not fully configured")
        else:
            self.twilio_client = Client(self.account_sid, self.auth_token)
        
        # Initialize TTS
        elevenlabs_config = config.get("elevenlabs", {})
        self.tts = ElevenLabsTTS(elevenlabs_config)
        
        # Configure webhook URL
        self.webhook_base_url = config.get("webhook_base_url", os.environ.get("WEBHOOK_BASE_URL", ""))
        if not self.webhook_base_url:
            self.logger.warning("Webhook base URL not configured. Call handling may not work correctly.")
        
        # Initialize Flask app for webhooks
        self.app = Flask(__name__)
        self.setup_routes()
        
        # Audio cache for previously synthesized responses
        self.audio_cache = {}
        self.audio_cache_dir = elevenlabs_config.get("output_dir", "data/audio_cache")
        os.makedirs(self.audio_cache_dir, exist_ok=True)
        
        self.logger.info("Enhanced call handler initialized")
    
    def setup_routes(self):
        """Set up Flask routes for Twilio webhooks"""
        
        @self.app.route("/", methods=["GET"])
        def index():
            """Simple health check endpoint"""
            return "BFSI Sales Agent is running"
        
        @self.app.route("/call", methods=["POST"])
        def incoming_call():
            """Handle incoming calls"""
            caller = request.values.get("From", "unknown")
            self.logger.info(f"Incoming call from: {caller}")
            
            # Create TwiML response
            response = VoiceResponse()
            
            # Generate greeting using sales agent
            greeting = self.sales_agent.process_input(
                request.values.get("CallSid"), 
                "hello"
            )
            
            # Use ElevenLabs for higher quality TTS if possible
            try:
                # Generate speech with ElevenLabs
                audio_url = self._get_or_create_audio(greeting, request.values.get("CallSid"))
                
                if audio_url:
                    # Use ElevenLabs audio
                    response.play(audio_url)
                    
                    # Set up gathering after the greeting plays
                    self._add_speech_gathering(response)
                else:
                    # Fallback to Twilio TTS
                    gather = Gather(
                        input="speech",
                        action="/process_speech",
                        method="POST",
                        language="en-IN",
                        speech_timeout="auto",
                        speech_model="phone_call"
                    )
                    gather.say(greeting, voice="alice")
                    response.append(gather)
            except Exception as e:
                self.logger.error(f"Error with ElevenLabs TTS: {str(e)}")
                # Fallback to Twilio TTS
                gather = Gather(
                    input="speech",
                    action="/process_speech",
                    method="POST",
                    language="en-IN",
                    speech_timeout="auto",
                    speech_model="phone_call"
                )
                gather.say(greeting, voice="alice")
                response.append(gather)
            
            # If user doesn't say anything, redirect
            response.redirect("/call")
            
            return Response(str(response), mimetype="text/xml")
        
        @self.app.route("/process_speech", methods=["POST"])
        def process_speech():
            """Process user speech input"""
            # Get the user's speech input
            user_input = request.values.get("SpeechResult", "")
            call_sid = request.values.get("CallSid")
            
            self.logger.info(f"Received speech: '{user_input}' from call {call_sid}")
            
            # Process through sales agent
            agent_response = self.sales_agent.process_input(call_sid, user_input)
            
            # Create TwiML response
            response = VoiceResponse()
            
            # Check if we need to transfer to a human
            if "human" in agent_response.lower() and "transfer" in agent_response.lower():
                response.say("Transferring you to a human agent. Please hold.", voice="alice")
                
                # Use call transfer settings from use case config if available
                transfer_to = self.sales_agent.use_case_config.get('transfers', {}).get('default_agent')
                if transfer_to:
                    response.dial(transfer_to)
                else:
                    # Fallback to a placeholder
                    response.dial("+1234567890")  # Replace with actual agent number in production
                
                return Response(str(response), mimetype="text/xml")
            
            # Use ElevenLabs for TTS if possible
            try:
                audio_url = self._get_or_create_audio(agent_response, call_sid)
                
                if audio_url:
                    # Play the ElevenLabs audio
                    response.play(audio_url)
                    
                    # Add gathering after the response
                    self._add_speech_gathering(response)
                else:
                    # Fallback to Twilio TTS
                    gather = Gather(
                        input="speech",
                        action="/process_speech",
                        method="POST",
                        language="en-IN",
                        speech_timeout="auto",
                        speech_model="phone_call"
                    )
                    gather.say(agent_response, voice="alice")
                    response.append(gather)
            except Exception as e:
                self.logger.error(f"Error with ElevenLabs TTS: {str(e)}")
                # Fallback to Twilio TTS
                gather = Gather(
                    input="speech",
                    action="/process_speech",
                    method="POST",
                    language="en-IN",
                    speech_timeout="auto",
                    speech_model="phone_call"
                )
                gather.say(agent_response, voice="alice")
                response.append(gather)
            
            # If user doesn't say anything, redirect
            response.redirect("/process_speech")
            
            return Response(str(response), mimetype="text/xml")
        
        @self.app.route("/audio/<filename>", methods=["GET"])
        def serve_audio(filename):
            """Serve audio files from cache"""
            file_path = os.path.join(self.audio_cache_dir, filename)
            if os.path.exists(file_path):
                return send_file(file_path, mimetype="audio/mpeg")
            else:
                return "Audio file not found", 404
    
    def _add_speech_gathering(self, response):
        """Add speech gathering to a TwiML response"""
        # Create a new gather
        gather = Gather(
            input="speech",
            action="/process_speech",
            method="POST",
            language="en-IN",
            speech_timeout="auto",
            speech_model="phone_call",
            enhanced=True  # Use enhanced speech recognition
        )
        
        # Add a prompt (optional - usually ElevenLabs audio has already played)
        gather.say("", voice="alice")
        
        # Add the gather to the response
        response.append(gather)
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=3))
    def _get_or_create_audio(self, text, call_sid):
        """
        Get or create audio for text using ElevenLabs
        
        Args:
            text (str): Text to synthesize
            call_sid (str): Call SID for tracking
            
        Returns:
            str: URL to the audio file
        """
        # Create a deterministic filename based on the text
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        audio_filename = f"{text_hash}.mp3"
        audio_path = os.path.join(self.audio_cache_dir, audio_filename)
        
        # Check if we already have this audio cached
        if os.path.exists(audio_path):
            self.logger.info(f"Using cached audio for: '{text[:30]}...'")
            # Return the URL to the audio file
            return f"{self.webhook_base_url}/audio/{audio_filename}"
        
        # Generate new audio
        try:
            # Synthesize with ElevenLabs
            audio_file = self.tts.synthesize_speech(text)
            
            if audio_file:
                # Copy to our cache directory with the hashed filename
                import shutil
                shutil.copy(audio_file, audio_path)
                
                # Delete the temporary file
                os.unlink(audio_file)
                
                # Return the URL
                return f"{self.webhook_base_url}/audio/{audio_filename}"
            else:
                self.logger.warning(f"Failed to synthesize speech with ElevenLabs")
                return None
        
        except Exception as e:
            self.logger.error(f"Error synthesizing speech: {str(e)}")
            return None
    
    def start(self):
        """Start the call handler server"""
        port = self.config.get("port", 5000)
        host = self.config.get("host", "0.0.0.0")
        debug = self.config.get("debug", True)
        
        # Run in a separate thread to not block the main thread
        server_thread = Thread(target=lambda: self.app.run(host=host, port=port, debug=debug))
        server_thread.daemon = True
        server_thread.start()
        
        self.logger.info(f"Call handler started on {host}:{port}")
    
    def make_outbound_call(self, to_number, callback_url=None):
        """
        Make an outbound call
        
        Args:
            to_number (str): Number to call
            callback_url (str, optional): Custom callback URL
            
        Returns:
            str: Call SID
        """
        if not hasattr(self, "twilio_client"):
            self.logger.error("Twilio client not initialized")
            raise ValueError("Twilio client not initialized")
        
        callback = callback_url or f"{self.webhook_base_url}/call"
        
        try:
            call = self.twilio_client.calls.create(
                to=to_number,
                from_=self.twilio_number,
                url=callback
            )
            
            self.logger.info(f"Outbound call initiated to {to_number}, SID: {call.sid}")
            return call.sid
        
        except Exception as e:
            self.logger.error(f"Failed to make outbound call: {str(e)}")
            raise