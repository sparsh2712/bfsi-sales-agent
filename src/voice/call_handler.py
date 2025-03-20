"""
Call handling module for interacting with Twilio
"""
import os
import logging
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.rest import Client
from threading import Thread

from .elevenlabs_tts import ElevenLabsTTS

class CallHandler:
    """
    Handles voice calls using Twilio and ElevenLabs for text-to-speech
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
        
        # Initialize Flask app for webhooks
        self.app = Flask(__name__)
        self.setup_routes()
        
        self.logger.info("Call handler initialized")
    
    def setup_routes(self):
        """Set up Flask routes for Twilio webhooks"""
        
        @self.app.route("/", methods=["GET"])
        def index():
            """Simple health check endpoint"""
            return "BFSI Sales Agent is running"
        
        @self.app.route("/call", methods=["POST"])
        def incoming_call():
            """Handle incoming calls"""
            self.logger.info(f"Incoming call from: {request.values.get('From', 'unknown')}")
            
            # Create TwiML response
            response = VoiceResponse()
            
            # Generate greeting
            greeting = self.sales_agent.process_input(
                request.values.get("CallSid"), 
                "hello"
            )
            
            # Create a gathering of user speech
            gather = Gather(
                input="speech",
                action="/process_speech",
                method="POST",
                language="en-IN",  # Indian English
                speech_timeout="auto",
                speech_model="phone_call"
            )
            
            # Add the greeting to the response
            gather.say(greeting, voice="alice")
            
            # Add the gather to the response
            response.append(gather)
            
            # If user doesn't say anything, try again
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
                
                # In a real implementation, you would use Twilio's Dial verb to transfer
                # This is just a placeholder example
                response.dial("+1234567890")  # Replace with actual agent number
                return Response(str(response), mimetype="text/xml")
            
            # Create a gathering of user speech
            gather = Gather(
                input="speech",
                action="/process_speech",
                method="POST",
                language="en-IN",  # Indian English
                speech_timeout="auto",
                speech_model="phone_call"
            )
            
            # Add the response to the gather
            # In production, you would use ElevenLabs TTS instead of Twilio's built-in TTS
            gather.say(agent_response, voice="alice")
            
            # Add the gather to the response
            response.append(gather)
            
            # If user doesn't say anything, try again
            response.redirect("/process_speech")
            
            return Response(str(response), mimetype="text/xml")
    
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
        
        callback = callback_url or f"{self.config.get('webhook_base_url', '')}/call"
        
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