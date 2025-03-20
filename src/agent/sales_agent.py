"""
Core sales agent implementation
"""
import logging
import re
import json
from datetime import datetime

class SalesAgent:
    """
    Main sales agent class that processes user input and generates responses
    using the knowledge base and configured use case.
    """
    
    def __init__(self, agent_config, use_case_config, knowledge_base):
        """
        Initialize the sales agent
        
        Args:
            agent_config (dict): Agent configuration
            use_case_config (dict): Use case specific configuration
            knowledge_base (KnowledgeBase): Knowledge base instance
        """
        self.config = agent_config
        self.use_case_config = use_case_config
        self.knowledge_base = knowledge_base
        self.logger = logging.getLogger(__name__)
        
        # Track conversation state for each session
        self.sessions = {}
        
        # Load responses from use case config
        self.responses = use_case_config.get("responses", {})
        
        # Initialize intent patterns
        self._init_intent_patterns()
        
        self.logger.info(f"Sales agent initialized for use case: {use_case_config.get('name', 'unnamed')}")
    
    def _init_intent_patterns(self):
        """Initialize the intent recognition patterns"""
        self.intent_patterns = {
            "greeting": [r"hello", r"hi", r"hey", r"good morning", r"good afternoon", r"namaste"],
            "goodbye": [r"bye", r"goodbye", r"see you", r"thank you", r"thanks"],
            "product_info": [r"product", r"tell me about", r"what is", r"information", r"details"],
            "pricing": [r"price", r"cost", r"fee", r"how much", r"rate", r"interest"],
            "eligibility": [r"eligible", r"qualify", r"can i", r"requirements", r"criteria"],
            "process": [r"process", r"how to", r"steps", r"procedure", r"what do i need to do"],
            "speak_to_human": [r"human", r"agent", r"person", r"representative", r"speak to someone"],
            "fund_performance": [r"performance", r"return", r"yield", r"how good", r"net gain"],
            "risk": [r"risk", r"safe", r"secure", r"guarantee", r"lose money"],
            "investment_duration": [r"duration", r"how long", r"term", r"maturity", r"period"],
            "contact": [r"contact", r"phone", r"email", r"office", r"branch", r"location"]
        }
    
    def process_input(self, session_id, user_input):
        """
        Process user input and generate a response
        
        Args:
            session_id (str): Unique session identifier
            user_input (str): User's input text
            
        Returns:
            str: Agent's response
        """
        # Ensure session exists
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "history": [],
                "context": {},
                "created_at": datetime.now().isoformat()
            }
        
        session = self.sessions[session_id]
        
        # Log the input
        self.logger.info(f"Session {session_id}: Received input: '{user_input}'")
        
        # Identify intent
        intent = self._identify_intent(user_input)
        
        # Record in history
        session["history"].append({
            "user": user_input,
            "intent": intent,
            "timestamp": datetime.now().isoformat()
        })
        
        # Get response based on intent
        response = self._generate_response(intent, user_input, session)
        
        # Add response to history
        session["history"][-1]["agent"] = response
        
        # Update context based on interaction
        self._update_context(session, intent, user_input, response)
        
        return response
    
    def _identify_intent(self, text):
        """
        Identify the user's intent from their input
        
        Args:
            text (str): User's input text
            
        Returns:
            str: Identified intent
        """
        text = text.lower()
        
        # Check against all patterns
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(r'\b' + pattern + r'\b', text):
                    self.logger.debug(f"Matched intent: {intent} with pattern: {pattern}")
                    return intent
        
        # Default to general inquiry if no specific intent is detected
        return "general_inquiry"
    
    def _generate_response(self, intent, user_input, session):
        """
        Generate a response based on the identified intent
        
        Args:
            intent (str): Identified intent
            user_input (str): User's input text
            session (dict): Session information
            
        Returns:
            str: Generated response
        """
        # Special handling for certain intents
        if intent == "greeting":
            return self._get_greeting_response(session)
        
        if intent == "goodbye":
            return self._get_random_response("goodbye")
        
        if intent == "speak_to_human":
            return self._get_random_response("speak_to_human")
        
        # Query the knowledge base for other intents
        knowledge_response = self.knowledge_base.query(user_input, intent)
        
        if knowledge_response:
            # If we have intent-specific templates, use them
            if intent in self.responses:
                response_template = self._get_random_response(intent)
                return response_template.format(answer=knowledge_response)
            return knowledge_response
        
        # Fallback response if no relevant information found
        return self._get_random_response("fallback")
    
    def _get_greeting_response(self, session):
        """
        Get an appropriate greeting response based on session state
        
        Args:
            session (dict): Session information
            
        Returns:
            str: Greeting response
        """
        # Check if this is the first interaction
        if len(session["history"]) <= 1:
            return self._get_random_response("initial_greeting")
        
        # Returning user
        return self._get_random_response("return_greeting")
    
    def _get_random_response(self, response_type):
        """
        Get a random response of the specified type
        
        Args:
            response_type (str): Type of response to get
            
        Returns:
            str: Selected response
        """
        import random
        
        responses = self.responses.get(response_type, [])
        
        # Fallback to default responses if none configured for this type
        if not responses:
            default_responses = {
                "initial_greeting": [
                    "Hello! I'm your financial advisor assistant. How may I help you today?",
                    "Namaste! I'm here to assist you with information about our financial products. What would you like to know?"
                ],
                "return_greeting": [
                    "Welcome back! What can I help you with today?",
                    "Hello again! How can I assist you further?"
                ],
                "goodbye": [
                    "Thank you for your time. Have a great day!",
                    "It was nice speaking with you. If you have more questions, feel free to reach out."
                ],
                "speak_to_human": [
                    "I understand you'd like to speak with a human agent. Let me transfer you to one of our representatives.",
                    "I'll connect you with a human representative who can assist you further."
                ],
                "fallback": [
                    "I apologize, but I don't have specific information about that. Could you please ask something else about our products or services?",
                    "I'm sorry, I couldn't find relevant information. Would you like to know about our financial products instead?"
                ]
            }
            responses = default_responses.get(response_type, ["I'm sorry, I don't understand. Could you rephrase that?"])
        
        return random.choice(responses)
    
    def _update_context(self, session, intent, user_input, response):
        """
        Update the session context based on the current interaction
        
        Args:
            session (dict): Session information
            intent (str): Identified intent
            user_input (str): User's input text
            response (str): Agent's response
        """
        context = session.get("context", {})
        
        # Update last interaction time
        context["last_interaction"] = datetime.now().isoformat()
        
        # Track topics discussed
        if "topics_discussed" not in context:
            context["topics_discussed"] = []
        
        if intent not in context["topics_discussed"]:
            context["topics_discussed"].append(intent)
        
        # Extract and store entities if applicable
        entities = self._extract_entities(user_input, intent)
        if entities:
            if "entities" not in context:
                context["entities"] = {}
            context["entities"].update(entities)
        
        session["context"] = context
    
    def _extract_entities(self, text, intent):
        """
        Extract relevant entities from user input based on intent
        
        Args:
            text (str): User's input text
            intent (str): Identified intent
            
        Returns:
            dict: Extracted entities
        """
        entities = {}
        
        # Extract based on intent
        if intent == "product_info":
            # Try to identify specific products mentioned
            products = ["loan", "insurance", "investment", "mutual fund", "fixed deposit"]
            for product in products:
                if product in text.lower():
                    entities["product"] = product
                    break
        
        elif intent == "pricing":
            # Look for amounts
            amount_match = re.search(r'(\d+)(?:\s*(?:rs|rupees|lakhs?|k|crores?|cr))?', text.lower())
            if amount_match:
                entities["amount"] = amount_match.group(1)
        
        return entities

    def get_session_summary(self, session_id):
        """
        Get a summary of the conversation session
        
        Args:
            session_id (str): Session identifier
            
        Returns:
            dict: Session summary
        """
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        session = self.sessions[session_id]
        return {
            "session_id": session_id,
            "created_at": session["created_at"],
            "interaction_count": len(session["history"]),
            "topics_discussed": session.get("context", {}).get("topics_discussed", []),
            "last_interaction": session.get("context", {}).get("last_interaction", "")
        }